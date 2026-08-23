# System Architecture

## 1. Objective
Ask Victoria is an agentic retail-assistance system that converts natural-language shopping questions into grounded product discovery, review intelligence, recommendation, comparison and follow-up answers.

## 2. Logical architecture

```text
User / Streamlit / API
        │
        ▼
Input Guardrails
        │
        ▼
Supervisor Router (LangGraph)
        │
 ┌──────┼────────────────────┐
 ▼      ▼                    ▼
Product Review          Recommendation /
Agent   Intelligence    Comparison Agent
        Agent
 └──────┴──────────┬─────────┘
                   ▼
       Retail Tool Abstraction
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
Product Retrieval          Review Retrieval
      │                         │
      └────────────┬────────────┘
                   ▼
 Constraint-aware Hybrid Retrieval
 TF-IDF + dense LSA + structured reranking
                   │
                   ▼
          Evidence Context
                   │
                   ▼
             Answer Agent
                   │
                   ▼
              Output Judge
              /          \
           PASS          FAIL
            │              │
            ▼              ▼
         Response     Self-Correction
            │
            ▼
    Evaluation + Observability
```

## 3. Agent responsibilities

### Supervisor / Router
Classifies the request into product, review, recommendation or comparison workflows. Routing is deterministic in the zero-cost build so behavior is testable and reproducible.

### Product Agent
Retrieves structured catalogue facts such as price, color, material, support and description. For context-dependent follow-ups it can resolve the active product from session memory.

### Review Intelligence Agent
Retrieves relevant reviews and produces aspect-level evidence for comfort, fit, support, material, activity, padding, durability and style. It exposes representative review quotes and a fit signal.

### Recommendation Agent
Retrieves eligible candidates, applies hard constraints, then uses explainable reranking based on retrieval relevance, review aspects, ratings, confidence and value.

### Comparison Agent
Builds side-by-side evidence from product attributes and aggregated review signals.

### Answer Agent
Synthesizes only from supplied evidence. Gemini can be used optionally; otherwise a deterministic grounded fallback is used.

### Judge / Self-Correction
Checks output for unsupported claims, unsupported numeric claims and insufficient evidence overlap. A failed answer is replaced by a safe fallback rather than passed to the user.

## 4. Retrieval architecture

The live build combines three signals:

1. **Lexical relevance** using TF-IDF n-grams.
2. **Dense latent-semantic similarity** using TruncatedSVD/LSA over the TF-IDF matrix.
3. **Structured commerce reranking** over high-value fields such as name, category, color, material and support.

Hard filters for explicit category, color and maximum price are applied before final ranking.

This design keeps the demo lightweight enough for free Streamlit deployment while preserving a clean interface that could later be backed by transformer embeddings and a managed vector store.

## 5. Recommendation architecture

The final ranking score combines:

- 42% retrieval relevance
- 25% requested review-aspect sentiment
- 15% average rating
- 10% review-evidence confidence
- 8% value/price fit

The score is intentionally explainable and local to the retrieved candidate set.

## 6. State and memory

LangGraph state carries the query, sanitized query, intent, selected products, evidence, context, conversation, answer, judge status, retry count, safety category, trace and latency.

Session memory stores the active product and recent products so follow-up questions such as “What material is it made from?” can resolve the correct entity.

## 7. Safety architecture

Input controls cover prompt-injection patterns, medical-claim requests and PII redaction. Output controls verify unsupported phrases, commerce numbers and minimum evidence overlap. The system is intentionally designed to abstain rather than fabricate unavailable facts.

## 8. Observability

Every run emits an explicit trace. Telemetry derives intent, blocked state, safety category, evidence count, selected product count, groundedness proxy, retry count, latency, trace steps, inferred tool events and judge status.

## 9. Interfaces

- **Streamlit:** product-detail and conversational portfolio experience.
- **FastAPI:** programmatic health/product/chat interface.
- **Evaluation pages:** retrieval, recommendation, safety, observability and end-to-end benchmark inspection.

## 10. Production boundary

The current build is a portfolio reference architecture over synthetic data. Production would require authenticated services, persistent data stores, managed vector search, privacy controls, rate limiting, monitoring, human evaluation, experimentation and business KPI instrumentation.