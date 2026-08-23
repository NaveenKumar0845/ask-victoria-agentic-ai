# Production Roadmap

## Goal
Evolve the public zero-cost portfolio architecture into an enterprise-grade conversational-commerce platform without losing its core principles: grounded evidence, explainability, safety, evaluation and observability.

## Phase 1 — Data and retrieval hardening

### Catalogue and review ingestion
- Connect authenticated product-information, pricing, inventory and review feeds.
- Add schema validation, deduplication, freshness metadata and data-quality monitoring.
- Separate real-time fields such as price/inventory from slower-changing product content.

### Semantic retrieval
- Replace dense LSA with transformer embeddings.
- Use a managed vector database or ANN service appropriate to scale.
- Keep structured filters for category, market, availability, price and compliance.
- Add query rewriting, hybrid lexical/vector retrieval and reranking.
- Evaluate by product category, language, market and intent.

## Phase 2 — Intelligence pipelines

### Offline review intelligence
- Incrementally process new reviews.
- Extract aspects/topics with scalable NLP/LLM pipelines.
- Store product-level review summaries, sentiment distributions, representative evidence and freshness timestamps.
- Trigger recomputation only when sufficient new evidence arrives.

### Recommendation services
- Separate candidate generation from ranking.
- Add shopper/context features only with consent and appropriate governance.
- Calibrate ranking weights using offline labels and online experiments.
- Keep explanation signals available even if the ranking model becomes learned.

## Phase 3 — Agent platform

### Orchestration
- Keep LangGraph or an equivalent explicit state-machine abstraction for multi-step workflows.
- Introduce typed tool contracts, schema validation, timeouts, retries and circuit breakers.
- Add model routing for cost/latency/quality tiers.

### Memory
- Replace session-only memory with consent-aware persistent preference memory.
- Separate ephemeral conversation context from durable user preferences.
- Add retention and deletion controls.

### Tool security
- Apply least-privilege authorization per tool.
- Require explicit confirmation for any future transactional action such as cart, order or account changes.
- Isolate read-only product intelligence from write-capable commerce actions.

## Phase 4 — Responsible AI and security

- Enterprise authentication and authorization.
- DLP/PII detection and redaction.
- Prompt-injection and data-exfiltration classifiers.
- Policy engine for prohibited claims and regulated markets.
- Audit logs and trace retention.
- Adversarial/red-team test suites.
- Human escalation for ambiguous high-risk cases.
- Model/provider data-governance controls.

## Phase 5 — Reliability and observability

### Platform telemetry
- Distributed tracing across router, retrieval, tools, model calls and downstream services.
- P50/P95/P99 latency by workflow.
- Error rate and timeout rate.
- Model calls/query and tokens/query.
- Cache hit rate.
- Retrieval latency and index freshness.
- Guardrail/retry/abstention rates.

### SLOs
Define separate SLOs for product fact Q&A, recommendation, review intelligence and comparison instead of one generic chatbot SLA.

## Phase 6 — Evaluation

### Offline
- Large human-labeled query set.
- Router/tool-selection accuracy.
- Retrieval Precision@K, Recall@K, nDCG and MRR.
- Recommendation relevance and diversity.
- Groundedness/factuality via deterministic checks + model judge + human review.
- Context-follow-up accuracy.
- Safety/adversarial pass rates.

### Online
- Helpfulness feedback.
- Search/recommendation task success.
- PDP engagement.
- Add-to-cart / conversion lift.
- Return-rate and support-contact impact where causally appropriate.
- A/B experiments with guardrails and holdouts.

## Phase 7 — Scaling patterns

### At ~100K products
- Precomputed transformer embeddings.
- Managed vector index.
- Cached product/review intelligence.
- Batch + incremental ingestion.

### At ~1M+ products / multiple markets
- Partition/shard by market/category as appropriate.
- Distributed retrieval and reranking.
- Event-driven refresh pipelines.
- Regional data residency.
- Localization and multilingual retrieval/evaluation.
- Feature store or ranking-service architecture if personalization is introduced.

## Reference production architecture

```text
Channels
Web / App / API
      │
      ▼
API Gateway + Auth
      │
      ▼
Agent Orchestrator
      │
 ┌────┼───────────────┬─────────────┐
 ▼    ▼               ▼             ▼
Search Review         Recommender    Comparison
Tool   Intelligence   Service        Tool
 │        │               │             │
 └────────┴───────┬───────┴─────────────┘
                  ▼
       Retrieval / Evidence Platform
 structured search + vector search + reranking
                  │
     ┌────────────┴────────────┐
     ▼                         ▼
Catalogue / Inventory      Review Intelligence
Stores                     + Vector Stores
                  │
                  ▼
          Model Gateway / LLM
                  │
                  ▼
       Safety + Output Judge
                  │
                  ▼
              Response

Cross-cutting: IAM · consent · audit · caching · tracing · evaluation · feature flags · experimentation
```

## Key production principle
Do not scale the demo by simply giving a larger LLM more context. Scale the **data contracts, retrieval, evidence, ranking, governance and evaluation layers** independently, and use the model where generation adds value.