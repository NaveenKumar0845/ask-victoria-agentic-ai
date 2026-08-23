from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

import pandas as pd


@dataclass
class QualityMetrics:
    groundedness: float
    evidence_count: int
    answer_length: int
    latency_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9₹]+", text.lower()) if len(t) > 2}


def groundedness_score(answer: str, evidence: list[str]) -> float:
    if not answer or not evidence:
        return 0.0
    answer_tokens = _tokens(answer)
    evidence_tokens = _tokens(" ".join(evidence))
    if not answer_tokens:
        return 0.0
    supported = len(answer_tokens & evidence_tokens)
    return round(min(1.0, supported / max(1, len(answer_tokens))), 3)


def quality_metrics(answer: str, evidence: list[str], latency_ms: float) -> QualityMetrics:
    return QualityMetrics(
        groundedness_score(answer, evidence),
        len(evidence),
        len(answer),
        round(latency_ms, 1),
    )


ROUTER_TESTS = [
    ("Find me a black sports bra under ₹2000 for yoga", "recommendation"),
    ("Recommend something comfortable for the gym", "recommendation"),
    ("Compare the Everyday Cloud Sports Bra with the Sculpt Medium Support Bra", "comparison"),
    ("Everyday Cloud Sports Bra vs Sculpt Medium Support Bra", "comparison"),
    ("What do customers say about the fit?", "review"),
    ("Does the Everyday Cloud Sports Bra run small?", "review"),
    ("What material is the AirFlex Yoga Bra made from?", "product"),
    ("How much is the CloudSoft Lounge Tee?", "product"),
]

# A broader deterministic benchmark across categories, attributes and intents.
# Expected IDs are tied to the public-safe synthetic catalogue in src/data.py.
RETRIEVAL_TESTS = [
    ("black yoga bra soft under 2000", "AV1001"),
    ("soft navy leggings for yoga and travel", "AV2002"),
    ("black cushioned shoes for gym training", "AV4001"),
    ("white cotton modal relaxed tee", "AV3001"),
    ("medium support black bra for strength training", "AV1002"),
    ("mauve lightweight yoga bra removable padding", "AV1003"),
    ("black stable strength training shoes", "AV4103"),
    ("navy walking sneaker breathable comfort", "AV4102"),
    ("black compression tights for gym workouts", "AV2102"),
    ("pink relaxed cotton tee", "AV3103"),
    ("black grip socks for pilates studio", "AV9101"),
    ("pink cushioned recovery slides", "AV5001"),
    ("white breathable performance tee for gym", "AV3102"),
    ("mauve studio wrap jacket low impact", "AV7102"),
    ("black soft lounge joggers for travel", "AV8103"),
    ("black bottle sling for walks", "AV9104"),
]

# Recommendation tests intentionally emphasize review-sensitive language such as
# comfort/support/fit in addition to catalogue attributes.
RECOMMENDATION_TESTS = [
    ("Recommend a soft comfortable black yoga bra under ₹2000", "AV1001"),
    ("Best supportive black sports bra for strength training under ₹2000", "AV1002"),
    ("Recommend soft navy leggings for yoga and travel", "AV2002"),
    ("Best stable black training shoes for strength work", "AV4103"),
    ("Recommend a breathable white gym tee", "AV3102"),
    ("Best cushioned pink slides for recovery", "AV5001"),
    ("Recommend soft black lounge joggers for travel", "AV8103"),
    ("Best black grip socks for Pilates studio classes", "AV9101"),
]


def evaluate_router(route_fn: Callable[[str], str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"query": q, "expected": e, "predicted": route_fn(q), "correct": route_fn(q) == e}
            for q, e in ROUTER_TESTS
        ]
    )


def evaluate_retrieval(retriever, top_k: int = 3) -> pd.DataFrame:
    rows = []
    for query, expected_id in RETRIEVAL_TESTS:
        hits = retriever.search_products(query, top_k=top_k)
        ids = hits["product_id"].tolist()
        rank = ids.index(expected_id) + 1 if expected_id in ids else None
        rows.append(
            {
                "query": query,
                "expected_product": expected_id,
                "top_k": ", ".join(ids),
                f"recall@{top_k}": expected_id in ids,
                "top_1": ids[0] if ids else "",
                "top_1_correct": bool(ids and ids[0] == expected_id),
                "rank": rank if rank is not None else "miss",
                "reciprocal_rank": (1.0 / rank) if rank else 0.0,
            }
        )
    return pd.DataFrame(rows)


def retrieval_summary(retriever, top_k: int = 3) -> dict:
    frame = evaluate_retrieval(retriever, top_k=top_k)
    return {
        f"recall@{top_k}": float(frame[f"recall@{top_k}"].mean()) if len(frame) else 0.0,
        "top_1_accuracy": float(frame["top_1_correct"].mean()) if len(frame) else 0.0,
        "mrr": float(frame["reciprocal_rank"].mean()) if len(frame) else 0.0,
        "cases": len(frame),
    }


def evaluate_recommendations(recommend_fn: Callable[[str, int], tuple[list[dict], list[str]]], top_k: int = 3) -> pd.DataFrame:
    rows = []
    for query, expected_id in RECOMMENDATION_TESTS:
        products, _ = recommend_fn(query, top_k)
        ids = [p.get("product_id", "") for p in products]
        rank = ids.index(expected_id) + 1 if expected_id in ids else None
        top_score = products[0].get("recommendation_score") if products else None
        expected_score = None
        for product in products:
            if product.get("product_id") == expected_id:
                expected_score = product.get("recommendation_score")
                break
        rows.append(
            {
                "query": query,
                "expected_product": expected_id,
                "top_k": ", ".join(ids),
                f"recall@{top_k}": expected_id in ids,
                "top_1": ids[0] if ids else "",
                "top_1_correct": bool(ids and ids[0] == expected_id),
                "rank": rank if rank is not None else "miss",
                "reciprocal_rank": (1.0 / rank) if rank else 0.0,
                "top_score": top_score,
                "expected_score": expected_score,
            }
        )
    return pd.DataFrame(rows)


def recommendation_summary(recommend_fn: Callable[[str, int], tuple[list[dict], list[str]]], top_k: int = 3) -> dict:
    frame = evaluate_recommendations(recommend_fn, top_k=top_k)
    return {
        f"recall@{top_k}": float(frame[f"recall@{top_k}"].mean()) if len(frame) else 0.0,
        "top_1_accuracy": float(frame["top_1_correct"].mean()) if len(frame) else 0.0,
        "mrr": float(frame["reciprocal_rank"].mean()) if len(frame) else 0.0,
        "cases": len(frame),
    }


def evaluate_agent(agent_fn: Callable[[str], dict], queries: list[str]) -> pd.DataFrame:
    rows = []
    for query in queries:
        start = perf_counter()
        result = agent_fn(query)
        latency = (perf_counter() - start) * 1000
        metrics = quality_metrics(result.get("final_answer", ""), result.get("evidence", []), latency)
        rows.append(
            {
                "query": query,
                "intent": result.get("intent", ""),
                "groundedness": metrics.groundedness,
                "evidence_count": metrics.evidence_count,
                "latency_ms": metrics.latency_ms,
                "blocked": result.get("blocked", False),
            }
        )
    return pd.DataFrame(rows)
