# Evaluation, Observability and Self-Correction

Ask Victoria uses a controlled synthetic benchmark so system behavior can be measured without claiming production performance.

## Evaluation layers

1. **Router evaluation** — verifies intent classification across product, review, recommendation and comparison requests.
2. **Retrieval evaluation** — measures Recall@3, Top-1 accuracy and MRR over the constraint-aware hybrid retrieval layer.
3. **Recommendation evaluation** — measures ranking quality after product retrieval and review-aware reranking.
4. **End-to-end evaluation** — measures routing, product selection, safety, answer completion, grounding, retries and latency together.

The end-to-end benchmark includes product facts, review questions, recommendations, comparisons, context-aware follow-ups, prompt-injection attempts and medical-claim requests.

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
- judge pass/fail

The Streamlit Observability page aggregates average and P95 latency, block rate, retry rate, judge pass rate, average evidence count and average tool activity.

## Stronger output judge

The output guard applies three deterministic checks before an answer is accepted:

1. Unsupported-claim phrase detection.
2. Numeric claim verification — explicit prices, ratings, percentages and recommendation scores must occur in the retrieved evidence.
3. Minimum evidence-overlap gate — answers with extremely low lexical support are rejected.

A rejected answer activates the LangGraph self-correction path and returns a safe grounded fallback rather than allowing an unsupported claim through.

## Interpretation limits

All benchmark numbers are generated against a controlled synthetic retail catalogue and review corpus. They demonstrate architecture quality, reproducibility and debugging discipline; they are **not production traffic metrics, external customer-study results or universal model-accuracy claims**.
