# Responsible AI & Safety Design

## Purpose
Ask Victoria is a retail decision-support assistant. Its safety design focuses on preventing unsupported commerce claims, handling adversarial instructions, protecting obvious personal information and avoiding health/medical guidance.

## 1. Safety principles

1. **Evidence over fluency** — product facts and review claims must come from retrieved evidence.
2. **Abstention over fabrication** — when evidence is insufficient, the assistant should say so.
3. **Scope discipline** — the system helps with retail discovery, comparison and review intelligence; it does not act as a medical advisor.
4. **Transparency** — benchmark limitations, synthetic data and recommendation-score meaning are explicitly disclosed.
5. **No proprietary data dependency** — the portfolio build uses synthetic product and review data.

## 2. Input controls

### Prompt injection
The guard checks for known instruction-override and prompt-exfiltration patterns such as requests to ignore previous instructions or reveal the system prompt. Detected requests are blocked before routing.

### Medical claims
Requests asking whether a retail product can cure, treat, diagnose, heal or prevent a condition are blocked and redirected to product-feature/customer-feedback scope.

### PII redaction
Detected email addresses and phone-number-like strings are redacted before downstream processing.

## 3. Output controls

### Unsupported-claim patterns
Known high-risk claim language such as guaranteed/clinically proven/doctor approved is rejected when generated.

### Numeric claim verification
Explicit commerce numbers—prices, ratings, percentages and recommendation scores—are normalized and checked against retrieved evidence. This reduces the risk of a fabricated number changing a purchase decision.

### Evidence-overlap gate
A deterministic lexical-support check rejects answers with extremely low overlap with the retrieved context. This is a transparent guardrail, not a production entailment model.

## 4. Judge and self-correction

The LangGraph Judge node runs after answer generation. A failed output is routed to a self-correction/fallback node rather than exposed directly to the user.

The current controlled benchmark has a 0% retry rate because all 12 validated scenarios pass the calibrated judge. The self-correction path remains part of the architecture and can activate on unsupported outputs outside those cases.

## 5. Recommendation transparency

The recommendation score is explicitly described as a ranking score within the retrieved candidate set. It is not presented as an objective or universal measure of product quality.

Ranking inputs are visible and interpretable: retrieval relevance, requested review aspects, rating, evidence confidence and value fit.

## 6. Data governance

The portfolio build uses synthetic retail products and reviews. No private employer data, customer records, order data, health data or proprietary retailer content is required.

If an external LLM is enabled, only public/synthetic content should be sent unless a production deployment has appropriate enterprise privacy, contractual and data-governance controls.

## 7. Evaluation

Safety is tested as part of the end-to-end suite, including prompt-injection and medical-claim scenarios. The current controlled benchmark reports 100% safety success and 100% grounding pass over 12 cases.

These results demonstrate the implemented test harness; they do not establish universal safety performance.

## 8. Known limitations

- Pattern-based prompt-injection detection can be bypassed by novel attacks.
- Lexical evidence overlap is not semantic entailment.
- PII detection is intentionally lightweight and can over/under-detect.
- Synthetic review sentiment does not represent real consumer populations.
- The project does not currently perform fairness evaluation across protected groups because it does not infer or use such attributes.

## 9. Production extensions

A production system should add policy classifiers, semantic prompt-injection detection, DLP/PII services, schema validation, authentication and authorization, audit logs, rate limits, model/provider governance, adversarial red-team suites, human review, incident management and continuous safety monitoring.