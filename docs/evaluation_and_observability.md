# Evaluation, Observability and Self-Correction

Ask Victoria uses transparent, controlled synthetic benchmarks so system behavior can be measured without making unsupported production claims.

## Evaluation layers

1. **Router evaluation** — intent classification across product, review, recommendation and comparison requests.
2. **Retrieval evaluation** — Recall@3, Top-1 accuracy and MRR over the constraint-aware hybrid retrieval layer.
3. **Recommendation evaluation** — ranking quality after retrieval and review-aware reranking.
4. **End-to-end evaluation** — routing, product selection, safety, answer completion, grounding, context behavior, retries and latency together.

## Final validated portfolio benchmark

### Retrieval

| Metric | Result |
|---|---:|
| Recall@3 | **100%** |
| Top-1 accuracy | **94%** |
| MRR | **97%** |

### End-to-end — 12 controlled cases

| Metric | Result |
|---|---:|
| Task success | **100%** |
| Routing success | **100%** |
| Product selection | **100%** |
| Safety success | **100%** |
| Grounding pass | **100%** |
| Average latency | **48 ms** |
| P95 latency | **81 ms** |
| Retry rate | **0%** |

The 12-case suite includes constrained recommendations, semantic recommendations, product facts, review questions, comparison, context-aware follow-up, price parsing, cross-category retrieval, prompt injection and medical-claim safety.

## What “100%” means

It means every case in this specific deterministic synthetic benchmark satisfied its declared checks. It does **not** mean the system is universally 100% accurate or safe. The benchmark is intentionally visible in code so reviewers can inspect its coverage and limitations.

The latency figures measure the current deterministic local execution path. They should not be described as Gemini API or production distributed-system latency.

## Observability

Every LangGraph run emits an explicit trace. The observability layer derives:

- intent
- block/safety category
- evidence count
- selected product count
- groundedness proxy
- retry count
- latency
- trace step count
- inferred tool events
- judge status

The Streamlit Observability page aggregates average/P95 latency, block rate, retry rate, judge pass rate, average evidence count and tool activity. Individual benchmark traces can be inspected step-by-step.

## Output judge

The Judge applies deterministic checks before an answer is accepted:

1. **Unsupported-claim phrase detection**.
2. **Numeric claim verification** — prices, ratings, percentages and recommendation scores are normalized and must be supported by retrieved evidence.
3. **Minimum evidence-overlap gate** — extremely low lexical support triggers rejection.

A rejected answer can activate the LangGraph self-correction path and return a safe grounded fallback.

The final 12 controlled cases currently produce a 0% retry rate after judge calibration. The retry path remains implemented and can activate on unsupported answers outside the benchmark.

## Development lesson: evaluate the evaluator

Earlier in development, the end-to-end page appeared stuck at 92% task/routing success and 33% retries. Two separate issues were identified:

- one recommendation phrase (“Best …”) was not covered by the deterministic router;
- Streamlit had cached the previous benchmark for ten minutes, making updated code look unchanged.

The router was broadened, numeric normalization was improved, diagnostic categories were added and the benchmark pages were changed to recompute fresh results. This is why observability and reproducibility are treated as first-class system features rather than presentation-only dashboards.

## Production evaluation extensions

A production program should add:

- larger human-labeled datasets;
- Precision@K, nDCG and category/market segmented retrieval metrics;
- tool-selection accuracy;
- semantic factuality/entailment evaluation;
- LLM-as-Judge with calibration against human labels;
- adversarial red-team suites;
- multilingual evaluation where relevant;
- online helpfulness/task-success metrics;
- A/B experimentation and business KPI measurement.

## Interpretation limits

All published project metrics are generated against the controlled synthetic retail catalogue/review corpus. They demonstrate implementation quality, reproducibility and debugging discipline; they are **not production traffic metrics, external customer-study results, or universal model-accuracy claims**.