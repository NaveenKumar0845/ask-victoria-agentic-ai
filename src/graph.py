from __future__ import annotations

from time import perf_counter
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from .data import load_products, load_reviews
from .evaluation import groundedness_score
from .guardrails import check_input, check_output, redact_pii
from .llm import gemini_answer
from .retrieval import RetailRetriever
from .tools import RetailTools


class AgentState(TypedDict, total=False):
    query: str
    sanitized_query: str
    intent: str
    context: dict
    conversation: list[dict]
    selected_product_ids: list[str]
    products: list[dict]
    evidence: list[str]
    draft_answer: str
    final_answer: str
    grounded: bool
    groundedness_score: float
    retry_count: int
    blocked: bool
    safety_category: str
    trace: list[str]
    latency_ms: float


PRODUCTS = load_products()
REVIEWS = load_reviews()
RETRIEVER = RetailRetriever(PRODUCTS, REVIEWS)
TOOLS = RetailTools(PRODUCTS, REVIEWS, RETRIEVER)


def _trace(state: AgentState, message: str) -> list[str]:
    return [*state.get("trace", []), message]


def _active_product_ids(state: AgentState) -> list[str]:
    context = state.get("context") or {}
    active = context.get("active_product_id")
    recent = context.get("recent_products") or []
    ids = [p for p in [active, *recent] if p]
    return list(dict.fromkeys(ids))


def _query_refers_to_context(query: str) -> bool:
    q = query.lower()
    return any(token in q.split() for token in ["it", "this", "that"]) or any(
        phrase in q for phrase in ["this one", "that one", "the first", "the second"]
    )


def route_intent(query: str) -> Literal["recommendation", "comparison", "review", "product"]:
    q = query.lower()
    if any(k in q for k in ["recommend", "find me", "looking for", "under ₹", "below ₹", "best option"]):
        return "recommendation"
    if any(k in q for k in ["compare", "versus", " vs ", "which of", "difference between"]):
        return "comparison"
    if any(k in q for k in ["review", "customers", "comfortable", "comfort", "fit", "run small", "complain", "love", "rating"]):
        return "review"
    return "product"


def input_guard_node(state: AgentState) -> AgentState:
    decision = check_input(state["query"])
    sanitized = redact_pii(state["query"])
    if not decision.allowed:
        return {
            "sanitized_query": sanitized,
            "blocked": True,
            "safety_category": decision.category,
            "draft_answer": decision.message,
            "grounded": True,
            "trace": _trace(state, f"Input guardrail blocked request: {decision.category}"),
        }
    return {
        "sanitized_query": sanitized,
        "blocked": False,
        "trace": _trace(state, "Input guardrail passed"),
    }


def router_node(state: AgentState) -> AgentState:
    intent = route_intent(state.get("sanitized_query", state["query"]))
    return {
        "intent": intent,
        "retry_count": state.get("retry_count", 0),
        "trace": _trace(state, f"Supervisor routed query to {intent} workflow"),
    }


def product_node(state: AgentState) -> AgentState:
    query = state.get("sanitized_query", state["query"])
    context_ids = _active_product_ids(state)
    if context_ids and _query_refers_to_context(query):
        chosen = PRODUCTS[PRODUCTS["product_id"].isin(context_ids[:1])]
        products = chosen.to_dict("records")
        evidence = [
            f"{r['product_id']} | {r['name']} | ₹{int(r['price'])} | {r['color']} | {r['material']} | "
            f"{r['support']} | padding={r['padding']} | {r['description']}"
            for _, r in chosen.iterrows()
        ]
        tool_name = "get_active_product"
    else:
        products, evidence = TOOLS.search_products(query, top_k=5)
        tool_name = "search_products"
    return {
        "products": products,
        "selected_product_ids": [p["product_id"] for p in products],
        "evidence": evidence,
        "trace": _trace(state, f"Product Agent called {tool_name} and retrieved {len(products)} products"),
    }


def review_node(state: AgentState) -> AgentState:
    query = state.get("sanitized_query", state["query"])
    product_ids = state.get("selected_product_ids") or []
    context_ids = _active_product_ids(state)
    if not product_ids and context_ids and _query_refers_to_context(query):
        product_ids = context_ids[:1]
    if not product_ids:
        products, _ = TOOLS.search_products(query, top_k=3)
        product_ids = [p["product_id"] for p in products]
    review_evidence = TOOLS.review_intelligence(query, product_ids, top_k=10)
    evidence = [*state.get("evidence", []), *review_evidence]
    return {
        "selected_product_ids": product_ids,
        "evidence": evidence,
        "trace": _trace(state, f"Review Intelligence Agent retrieved {len(review_evidence)} evidence items"),
    }


def recommendation_node(state: AgentState) -> AgentState:
    query = state.get("sanitized_query", state["query"])
    products, product_evidence = TOOLS.search_products(query, top_k=5)
    product_ids = [p["product_id"] for p in products]
    review_evidence = TOOLS.review_intelligence(query, product_ids[:3], top_k=8) if product_ids else []
    return {
        "products": products,
        "selected_product_ids": product_ids,
        "evidence": [*product_evidence, *review_evidence],
        "trace": _trace(
            state,
            f"Recommendation Agent called product + review tools for {len(product_ids)} candidates",
        ),
    }


def comparison_node(state: AgentState) -> AgentState:
    query = state.get("sanitized_query", state["query"])
    products, evidence = TOOLS.compare_products(query, top_k=3)
    return {
        "products": products,
        "selected_product_ids": [p["product_id"] for p in products],
        "evidence": evidence,
        "trace": _trace(state, f"Comparison Agent assembled evidence for {len(products)} products"),
    }


def _conversation_text(state: AgentState) -> str:
    conversation = state.get("conversation") or []
    if not conversation:
        return ""
    recent = conversation[-4:]
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent)


def answer_node(state: AgentState) -> AgentState:
    evidence = state.get("evidence", [])
    prompt = (
        "You are Ask Victoria, an objective retail shopping assistant. Answer ONLY from the supplied evidence. "
        "Never invent sizes, inventory, health claims, product specifications or customer claims. "
        "Clearly distinguish official product facts from customer-review evidence. If evidence is insufficient, say so.\n\n"
        f"Intent: {state.get('intent')}\n"
        f"Question: {state.get('sanitized_query', state['query'])}\n"
        f"Recent conversation:\n{_conversation_text(state)}\n"
        "Evidence:\n- " + "\n- ".join(evidence)
    )
    llm = gemini_answer(prompt)
    draft = llm.strip() if llm else deterministic_answer(state)
    return {
        "draft_answer": draft,
        "trace": _trace(state, "Answer Agent synthesized a grounded response" + (" with Gemini" if llm else " with zero-cost fallback")),
    }


def deterministic_answer(state: AgentState) -> str:
    intent = state.get("intent")
    products = state.get("products", [])
    evidence = state.get("evidence", [])
    if intent == "recommendation" and products:
        lines = ["I found these strong matches based on your constraints and the available product/review evidence:"]
        for p in products[:3]:
            lines.append(f"- **{p['name']}** — ₹{int(p['price'])}, {p['color']}, {p['support']}. {p['description']}")
        lines.append("I can compare comfort, fit, support or customer feedback for these options next.")
        return "\n".join(lines)
    if intent == "comparison" and evidence:
        return "Here is the evidence-based comparison:\n\n" + "\n\n".join(f"- {e}" for e in evidence[:3])
    if intent == "review" and evidence:
        return "Based on the available customer-review evidence:\n\n" + "\n\n".join(f"- {e}" for e in evidence[:5])
    if products:
        p = products[0]
        return (
            f"**{p['name']}** costs ₹{int(p['price'])}. It is {p['color']}, uses {p['material']}, "
            f"and is designed for {p['support'].lower()} use. {p['description']}"
        )
    return "I don't have enough product evidence to answer that reliably. Try asking about a product, reviews, comparison, or recommendation."


def judge_node(state: AgentState) -> AgentState:
    evidence = state.get("evidence", [])
    answer = state.get("draft_answer", "")
    decision = check_output(answer, evidence)
    score = groundedness_score(answer, evidence)
    return {
        "grounded": decision.allowed,
        "groundedness_score": score,
        "safety_category": decision.category if not decision.allowed else state.get("safety_category", "ok"),
        "trace": _trace(state, f"Judge Agent: {'PASS' if decision.allowed else 'FAIL'} | groundedness proxy={score:.2f}"),
    }


def retry_node(state: AgentState) -> AgentState:
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "draft_answer": (
            "I couldn't verify that claim from the available product and customer-review evidence, so I won't guess. "
            "Please ask about fit, comfort, support, material, price, reviews, or a product comparison."
        ),
        "grounded": True,
        "trace": _trace(state, "Self-correction path replaced an unsupported answer with a safe fallback"),
    }


def finalize_node(state: AgentState) -> AgentState:
    return {
        "final_answer": state.get("draft_answer", ""),
        "trace": _trace(state, "Workflow finalized response"),
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("router", router_node)
    graph.add_node("product", product_node)
    graph.add_node("review", review_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("comparison", comparison_node)
    graph.add_node("answer", answer_node)
    graph.add_node("judge", judge_node)
    graph.add_node("retry", retry_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("input_guard")
    graph.add_conditional_edges(
        "input_guard",
        lambda s: "blocked" if s.get("blocked") else "allowed",
        {"blocked": "finalize", "allowed": "router"},
    )
    graph.add_conditional_edges(
        "router",
        lambda s: s["intent"],
        {
            "product": "product",
            "review": "review",
            "recommendation": "recommendation",
            "comparison": "comparison",
        },
    )
    graph.add_edge("product", "answer")
    graph.add_edge("review", "answer")
    graph.add_edge("recommendation", "answer")
    graph.add_edge("comparison", "answer")
    graph.add_edge("answer", "judge")
    graph.add_conditional_edges(
        "judge",
        lambda s: "finalize" if s.get("grounded") or s.get("retry_count", 0) >= 1 else "retry",
        {"finalize": "finalize", "retry": "retry"},
    )
    graph.add_edge("retry", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


GRAPH = build_graph()


def ask_victoria(query: str, context: dict | None = None, conversation: list[dict] | None = None) -> dict:
    start = perf_counter()
    result = GRAPH.invoke(
        {
            "query": query,
            "context": context or {},
            "conversation": conversation or [],
            "retry_count": 0,
            "trace": [],
        }
    )
    result["latency_ms"] = round((perf_counter() - start) * 1000, 1)
    return result
