# Ask Victoria — Product Brief

## Problem
E-commerce shoppers often face information overload across product descriptions, specifications, ratings and large volumes of customer reviews. Important signals about fit, comfort, support, material and recurring complaints are difficult to discover quickly.

## Product vision
Ask Victoria is an agentic product-intelligence and conversational-commerce assistant that helps shoppers discover, compare and understand products using grounded product facts and customer-review evidence.

## Primary personas
- Shopper who wants fast product discovery under constraints such as price, color or use case.
- Shopper who wants to understand review themes such as comfort, fit or complaints.
- Shopper comparing multiple products and asking follow-up questions conversationally.
- AI/Product reviewer who wants transparency into routing, evidence, safety and evaluation.

## Core journeys
1. Natural-language product discovery.
2. Product-specific Q&A.
3. Review intelligence and aspect sentiment.
4. Product comparison.
5. Multi-turn follow-up using session context.
6. Safe handling of prompt injection, PII and unsupported medical/product claims.

## MVP functional requirements
- Route each user query to a specialized workflow.
- Retrieve product facts from the catalogue rather than model memory.
- Retrieve relevant customer-review evidence.
- Generate balanced review summaries.
- Preserve active-product context for follow-up questions.
- Judge generated answers and use a safe fallback when unsupported.
- Expose agent trace, evidence count, latency and a transparent groundedness proxy.
- Work without a paid model API; optionally use Gemini for natural-language synthesis.

## Success metrics
Portfolio metrics must be measured rather than invented. The included evaluation harness measures:
- intent-routing accuracy;
- safety behavior accuracy;
- groundedness proxy;
- evidence count;
- latency;
- retry/blocked behavior.

For a production system, add retrieval Precision@K/Recall@K, human factuality review, recommendation task success, conversion uplift, engagement, deflection, CSAT and cost/query.

## Responsible AI
The assistant does not infer hidden attributes, make health claims, expose internal prompts, or fabricate unavailable product facts. Public/synthetic data is used so no proprietary retail data is included.

## Portfolio positioning
This project is intended to demonstrate the intersection of AI Product thinking and hands-on engineering: business problem definition, agent architecture, tools, retrieval, memory, guardrails, evaluation, API design, UI and deployment.
