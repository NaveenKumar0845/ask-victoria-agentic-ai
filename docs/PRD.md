# Product Requirements Document (PRD)

## Product name
Ask Victoria — Agentic Product Intelligence & Conversational Commerce Platform

## 1. Problem statement
Online shoppers must combine structured product specifications with unstructured customer-review evidence to answer simple but high-value questions: Which option fits my needs? Does it run small? Which product is more comfortable? Is a recommendation supported by evidence?

Traditional filters handle explicit attributes but not nuanced review themes or conversational follow-ups. Generic LLM chatbots can answer naturally but may fabricate specifications, inventory or claims.

## 2. Product vision
Create a transparent shopping copilot that understands natural-language intent, routes requests to specialized workflows, grounds answers in catalogue and review evidence, explains recommendation rationale and exposes measurable safety/evaluation behavior.

## 3. Primary users

### Decision-focused shopper
Wants a fast shortlist based on constraints such as price, category, color, activity and comfort.

### Evidence-seeking shopper
Wants to understand recurring review signals about fit, comfort, support, material or complaints.

### Comparison shopper
Wants a concise comparison between shortlisted products and expects follow-up questions to retain context.

### Product / AI reviewer
Wants transparency into routing, tool calls, evidence, ranking, safety and quality metrics.

## 4. Jobs to be done

- Find products that satisfy explicit and semantic constraints.
- Understand what customers consistently praise or criticize.
- Compare products using both specifications and review evidence.
- Ask follow-up questions without restating the product.
- Receive recommendations with an understandable rationale.
- Avoid unsafe, unsupported or hallucinated commerce claims.

## 5. Functional requirements

### FR1 — Intent routing
The system shall route queries to product, review, recommendation or comparison workflows.

### FR2 — Product retrieval
The system shall retrieve product facts from the catalogue rather than relying on LLM memory.

### FR3 — Structured constraints
The system shall recognize and apply explicit price, color and category constraints where supported.

### FR4 — Review intelligence
The system shall aggregate review evidence into aspect-level signals and representative quotes.

### FR5 — Recommendation ranking
The system shall rank eligible candidates using transparent retrieval, review, rating, confidence and value signals.

### FR6 — Contextual follow-up
The system shall preserve active-product context for follow-up questions within a session.

### FR7 — Safety
The system shall block or safely redirect prompt-injection and medical-claim requests, redact detected PII and reject unsupported output claims.

### FR8 — Self-correction
The system shall provide a fallback path when the output judge rejects an answer.

### FR9 — Evaluation
The system shall expose routing, retrieval, recommendation, safety and end-to-end evaluation.

### FR10 — Observability
The system shall expose run-level traces, latency, evidence, retry and judge telemetry.

### FR11 — Zero-cost operation
The base experience shall remain functional without a paid model API.

## 6. Non-functional requirements

- Reproducible behavior over a deterministic synthetic benchmark.
- Clear distinction between product facts and customer-review evidence.
- No dependency on proprietary retailer data.
- Lightweight enough for Streamlit Community Cloud.
- Modular retrieval/model layer for future replacement.
- Transparent limitations and benchmark scope.

## 7. Current portfolio success metrics

Controlled synthetic benchmark:

- Task success: 100%
- Routing success: 100%
- Product selection: 100%
- Safety success: 100%
- Grounding pass: 100%
- Retrieval Recall@3: 100%
- Retrieval Top-1: 94%
- Retrieval MRR: 97%
- Average deterministic-path latency: 48 ms
- P95 deterministic-path latency: 81 ms

These figures are engineering benchmark results, not production or customer outcomes.

## 8. Product KPIs for a real deployment

### User value
- Recommendation task success
- Search-to-PDP engagement
- Comparison completion
- Helpful-answer rate
- CSAT
- Repeat usage

### Business value
- Conversion uplift
- Add-to-cart rate
- Product discovery depth
- Support/contact deflection
- Return-rate impact from improved fit understanding

### AI quality
- Retrieval Recall@K / Precision@K / MRR
- Grounded answer rate
- Hallucination / unsupported-claim rate
- Router accuracy
- Tool-selection accuracy
- Context-follow-up accuracy
- Safety pass rate

### Operational
- P50/P95 latency
- Cost/query
- Model calls/query
- Retry rate
- Error rate
- Cache hit rate

## 9. Out of scope for the portfolio build

- Real inventory or checkout transactions
- Personalized recommendations from private customer history
- Medical or health guidance
- Production authentication/authorization
- Real retailer integrations
- Long-term cross-session identity memory
- Claims of real conversion or CSAT improvement

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Hallucinated product facts | Evidence-only generation + output judge |
| Misleading recommendation score | Explicitly label score as within-candidate ranking |
| Prompt injection | Input pattern guard + scope restriction |
| Health claims | Medical-claim blocking |
| Sensitive data | PII redaction; synthetic corpus |
| Benchmark overfitting | Keep benchmark transparent; add human/external evaluation in production |
| Retrieval degradation at scale | Replace lightweight index with transformer + managed vector search |

## 11. Product principle

**Ground first, explain second, generate last.** The assistant should prefer a transparent evidence-backed answer—or abstention—over an impressive but unsupported response.