# Career Package — Resume, LinkedIn & Interview Positioning

## 1. Resume project title

**Ask Victoria — Agentic Product Intelligence & Conversational Commerce Platform**

## 2. Recommended resume bullets

Choose the three below that best fit the role:

- Built an end-to-end **agentic retail copilot using LangGraph, Streamlit and FastAPI**, orchestrating product, review-intelligence, recommendation and comparison workflows with session memory, tool use, guardrails, judge-based validation and self-correction.
- Designed **constraint-aware hybrid retrieval** combining TF-IDF, dense LSA vectors and structured commerce reranking across a synthetic **40-product / 800-review** corpus; achieved **100% Recall@3, 94% Top-1 accuracy and 97% MRR** on a controlled retrieval benchmark.
- Developed an **explainable recommendation engine** combining retrieval relevance, review-aspect sentiment, rating, evidence confidence and price/value fit, with transparent score decomposition and aspect-level customer-review evidence.
- Implemented a reproducible **12-case end-to-end evaluation and observability framework** covering routing, product selection, safety, grounding, multi-turn context, retries and latency; achieved **100% task/routing/product-selection/safety/grounding pass** on the controlled synthetic benchmark.
- Added **Responsible AI controls** for prompt injection, PII redaction, medical claims, unsupported commerce claims and numeric-claim verification, with a LangGraph Judge → self-correction path for rejected outputs.

### Recommended 3-bullet version for AI Product / GenAI Solutioning

- Built **Ask Victoria**, an end-to-end agentic retail copilot using LangGraph, Streamlit and FastAPI with specialized product, review, recommendation and comparison workflows, tool use, memory, guardrails and judge-based self-correction.
- Designed constraint-aware hybrid retrieval and explainable recommendation ranking over a **40-product / 800-review** synthetic corpus, achieving **100% Recall@3, 94% Top-1 and 97% MRR** on a controlled retrieval benchmark.
- Created evaluation/observability across routing, product selection, safety, grounding, retries and latency; achieved **100% end-to-end task success across 12 controlled synthetic scenarios**, with transparent limitations and Responsible AI checks.

## 3. LinkedIn project description

**Ask Victoria | Agentic Product Intelligence & Conversational Commerce Platform**

Designed and built a zero-cost, end-to-end Agentic AI retail assistant that helps shoppers discover, compare and understand products using grounded catalogue and customer-review evidence.

The platform uses LangGraph to orchestrate specialized Product, Review Intelligence, Recommendation and Comparison workflows; constraint-aware hybrid retrieval (TF-IDF + dense LSA + structured reranking); aspect-level review intelligence; explainable recommendation scoring; multi-turn product memory; input/output guardrails; a Judge/self-correction path; FastAPI; Streamlit; and transparent evaluation/observability.

Controlled synthetic benchmark: 40 products, 800 reviews; retrieval Recall@3 100%, Top-1 94%, MRR 97%; 12-case end-to-end suite with 100% task, routing, product-selection, safety and grounding pass.

Benchmark figures are controlled portfolio results on synthetic data, not production-user metrics.

## 4. 30-second interview explanation

“Ask Victoria is an agentic conversational-commerce platform I built to solve e-commerce decision overload. Instead of sending every question directly to an LLM, a LangGraph supervisor routes requests to specialized product, review, recommendation or comparison workflows. Those agents use grounded product/review tools, hybrid retrieval and an explainable recommendation ranker. I also added conversational memory, safety guardrails, a judge/self-correction path and an evaluation/observability layer. The project runs without a paid API and has a controlled benchmark so I can quantify routing, retrieval, safety and end-to-end quality rather than making subjective claims.”

## 5. 90-second architecture answer

“The system starts with input guardrails for prompt injection, obvious PII and medical claims. A LangGraph supervisor then routes the query into one of four workflows: product facts, review intelligence, recommendation or comparison.

For retrieval I wanted something semantic but lightweight enough for free deployment, so I combined TF-IDF with dense LSA vectors and structured commerce-field reranking, while keeping explicit price/color/category constraints as hard filters. Review intelligence aggregates aspect-level signals like comfort, fit and support. For recommendations, I rank the retrieved candidate set using retrieval relevance, requested review-aspect evidence, rating, evidence confidence and value fit, and expose the score components in the UI.

The answer is generated only from the evidence context. A Judge checks unsupported phrases, commerce numbers and evidence overlap; failures can route to self-correction. Session state preserves the active product for follow-ups. Finally, evaluation and observability expose routing, retrieval, product-selection, safety, grounding, retries and latency.”

## 6. Why LangGraph instead of a simple chain?

A simple chain would be sufficient for one fixed RAG workflow. This system has conditional branches, specialized workflows, shared state, context-aware follow-ups and a Judge → retry edge. LangGraph makes those state transitions explicit, inspectable and testable.

## 7. Why is this not just RAG?

RAG is one component. Ask Victoria also performs intent routing, tool selection, structured constraints, aspect-level review analysis, explainable recommendation ranking, conversational state, output judging, safety gating, self-correction and observability. Retrieval augments the agents; it does not define the whole architecture.

## 8. Why LSA instead of SentenceTransformers/FAISS?

The public demo prioritizes zero cost, low memory use and reliable Streamlit cold starts. Dense LSA gives a real vector-retrieval layer without downloading a large transformer model. The retriever is modular, so in production I would replace this implementation with transformer embeddings and a managed vector store or FAISS depending on scale and infrastructure requirements.

## 9. Why not let the LLM rank recommendations?

I wanted recommendation logic to be transparent and reproducible. The weighted ranker exposes which signals influenced the order and can be evaluated independently. An LLM can still explain the ranking, but it does not get to invent the score.

## 10. How did you evaluate the system?

“I separated evaluation by layer. Router tests evaluate intent classification. Retrieval tests measure Recall@3, Top-1 and MRR. Recommendation tests evaluate the post-retrieval ranking layer. The E2E suite then evaluates routing, product selection, safety, answer completion, grounding, context follow-up, retries and latency together. I also expose traces so failures can be diagnosed instead of just counted.”

## 11. What does 100% task success mean?

It means all 12 scenarios in the current controlled synthetic E2E benchmark pass their expected routing/product/safety/answer/grounding checks. It does **not** mean the assistant is 100% accurate on arbitrary real-world queries. A production evaluation set would need substantially broader real queries, human labels, adversarial tests and online metrics.

## 12. Why was self-correction important if retry rate is 0% now?

The architecture must be able to reject unsupported output even when the current controlled cases happen to pass. During development the judge did trigger retries; calibrating those failures helped improve the numeric-grounding logic. A 0% retry rate on the final 12 cases means those cases now pass the judge, not that the retry path was removed.

## 13. How would you scale to 1M products?

Move catalogue/reviews into durable stores, generate transformer embeddings offline, use managed ANN/vector search, separate structured filtering from semantic retrieval, cache frequent queries and product intelligence, precompute review summaries/aspects, add asynchronous pipelines, introduce service-level tracing and model routing, and evaluate retrieval by category/market/intent rather than one global score.

## 14. What would you measure in production?

Technical metrics: retrieval Precision/Recall/MRR, groundedness, hallucination rate, router/tool accuracy, context accuracy, latency, cost/query, retries and errors.

Product/business metrics: search success, helpfulness, PDP engagement, add-to-cart, conversion, repeat usage, support deflection and potentially return-rate impact for fit-related use cases.

## 15. What was the hardest design decision?

A strong answer is:

“The most important decision was separating what should be deterministic from what should be generative. Product filters, ranking signals, safety checks and evaluation are more reliable when they are explicit and testable; the LLM is most useful for synthesis and explanation. That made the system more transparent, cheaper and easier to debug.”

## 16. What would you change with an enterprise budget?

Add real-time catalogue/review ingestion, transformer embeddings + managed vector search, persistent shopper memory, enterprise IAM, consent/privacy controls, centralized model gateway, semantic safety classifiers, full tracing, feature flags, A/B experimentation, human evaluation, online KPI dashboards, caching and production SLOs.