from __future__ import annotations

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from .data import load_products, load_reviews
from .retrieval import RetailRetriever, extract_constraints
from .intelligence import summarize_reviews
from .llm import gemini_answer


class AgentState(TypedDict, total=False):
    query: str
    intent: str
    selected_product_ids: list[str]
    products: list[dict]
    evidence: list[str]
    draft_answer: str
    final_answer: str
    grounded: bool
    retry_count: int


PRODUCTS = load_products()
REVIEWS = load_reviews()
RETRIEVER = RetailRetriever(PRODUCTS, REVIEWS)


def route_intent(query: str) -> Literal["recommendation","comparison","review","product"]:
    q = query.lower()
    if any(k in q for k in ["recommend","find me","looking for","under ₹","below ₹"]):
        return "recommendation"
    if any(k in q for k in ["compare","versus","vs ","which of"]):
        return "comparison"
    if any(k in q for k in ["review","customers","comfortable","fit","run small","complain","love"]):
        return "review"
    return "product"


def router_node(state: AgentState) -> AgentState:
    return {"intent": route_intent(state["query"]), "retry_count": state.get("retry_count", 0)}


def product_node(state: AgentState) -> AgentState:
    constraints = extract_constraints(state["query"])
    chosen = RETRIEVER.search_products(state["query"], top_k=5, **constraints)
    products = chosen.to_dict("records")
    evidence = [f"{r['product_id']} | {r['name']} | ₹{int(r['price'])} | {r['color']} | {r['material']} | {r['support']} | padding={r['padding']} | {r['description']}" for _, r in chosen.iterrows()]
    return {"products": products, "selected_product_ids": [p["product_id"] for p in products], "evidence": evidence}


def review_node(state: AgentState) -> AgentState:
    product_ids = state.get("selected_product_ids")
    if not product_ids:
        chosen = RETRIEVER.search_products(state["query"], top_k=3)
        product_ids = chosen["product_id"].tolist()
    review_hits = RETRIEVER.search_reviews(state["query"], product_ids=product_ids, top_k=10)
    evidence = list(state.get("evidence", []))
    for pid in product_ids[:3]:
        subset = REVIEWS[REVIEWS["product_id"] == pid]
        if subset.empty:
            continue
        summary = summarize_reviews(subset)
        name = PRODUCTS.loc[PRODUCTS["product_id"] == pid, "name"].iloc[0]
        evidence.append(f"Review intelligence for {pid} {name}: rating {summary['average_rating']}/5 from {summary['review_count']} demo reviews. {summary['love']} {summary['mention']}")
    for _, r in review_hits.head(5).iterrows():
        evidence.append(f"Customer review for {r['product_id']} ({int(r['rating'])}/5): {r['review_text']}")
    return {"selected_product_ids": product_ids, "evidence": evidence}


def comparison_node(state: AgentState) -> AgentState:
    chosen = RETRIEVER.search_products(state["query"], top_k=3)
    products = chosen.to_dict("records")
    evidence = []
    for p in products:
        subset = REVIEWS[REVIEWS["product_id"] == p["product_id"]]
        summary = summarize_reviews(subset)
        evidence.append(f"{p['name']} ({p['product_id']}): ₹{int(p['price'])}, {p['support']}, {p['material']}, rating {summary['average_rating']}/5. {summary['love']} {summary['mention']}")
    return {"products": products, "selected_product_ids": [p["product_id"] for p in products], "evidence": evidence}


def answer_node(state: AgentState) -> AgentState:
    evidence = state.get("evidence", [])
    prompt = (
        "You are Ask Victoria, an objective retail shopping assistant. Answer ONLY from the evidence. "
        "Never invent sizes, inventory, health claims, or product facts. If evidence is insufficient, say so. "
        f"Intent: {state.get('intent')}\nQuestion: {state['query']}\nEvidence:\n- " + "\n- ".join(evidence)
    )
    llm = gemini_answer(prompt)
    return {"draft_answer": llm.strip() if llm else deterministic_answer(state)}


def deterministic_answer(state: AgentState) -> str:
    intent = state.get("intent")
    products = state.get("products", [])
    evidence = state.get("evidence", [])
    if intent == "recommendation" and products:
        lines = ["I found these strong matches based on your constraints and the demo product catalogue:"]
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
        return f"**{p['name']}** costs ₹{int(p['price'])}. It is {p['color']}, uses {p['material']}, and is designed for {p['support'].lower()} use. {p['description']}"
    return "I don't have enough product evidence to answer that reliably. Try asking about a product, reviews, comparison, or recommendation."


def judge_node(state: AgentState) -> AgentState:
    answer = state.get("draft_answer", "")
    unsafe = any(k in answer.lower() for k in ["cure", "treats pain", "guaranteed medical"])
    return {"grounded": bool(state.get("evidence")) and not unsafe}


def retry_node(state: AgentState) -> AgentState:
    return {"retry_count": state.get("retry_count", 0) + 1, "draft_answer": "I couldn't verify that claim from the available product and customer-review evidence, so I won't guess. Please ask about fit, comfort, support, material, price, or a product comparison.", "grounded": True}


def finalize_node(state: AgentState) -> AgentState:
    return {"final_answer": state.get("draft_answer", "")}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("product", product_node)
    graph.add_node("review", review_node)
    graph.add_node("recommendation", product_node)
    graph.add_node("comparison", comparison_node)
    graph.add_node("answer", answer_node)
    graph.add_node("judge", judge_node)
    graph.add_node("retry", retry_node)
    graph.add_node("finalize", finalize_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", lambda s: s["intent"], {"product":"product","review":"review","recommendation":"recommendation","comparison":"comparison"})
    graph.add_edge("product", "answer")
    graph.add_edge("recommendation", "review")
    graph.add_edge("review", "answer")
    graph.add_edge("comparison", "answer")
    graph.add_edge("answer", "judge")
    graph.add_conditional_edges("judge", lambda s: "finalize" if s.get("grounded") or s.get("retry_count", 0) >= 1 else "retry", {"finalize":"finalize","retry":"retry"})
    graph.add_edge("retry", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


GRAPH = build_graph()


def ask_victoria(query: str) -> dict:
    return GRAPH.invoke({"query": query, "retry_count": 0})
