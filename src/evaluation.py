from __future__ import annotations

import re
from dataclasses import dataclass, asdict
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
    return QualityMetrics(groundedness_score(answer, evidence), len(evidence), len(answer), round(latency_ms, 1))

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

def evaluate_router(route_fn: Callable[[str], str]) -> pd.DataFrame:
    return pd.DataFrame([{"query": q, "expected": e, "predicted": route_fn(q), "correct": route_fn(q) == e} for q, e in ROUTER_TESTS])

def evaluate_agent(agent_fn: Callable[[str], dict], queries: list[str]) -> pd.DataFrame:
    rows = []
    for query in queries:
        start = perf_counter()
        result = agent_fn(query)
        latency = (perf_counter() - start) * 1000
        metrics = quality_metrics(result.get("final_answer", ""), result.get("evidence", []), latency)
        rows.append({"query": query, "intent": result.get("intent", ""), "groundedness": metrics.groundedness, "evidence_count": metrics.evidence_count, "latency_ms": metrics.latency_ms, "blocked": result.get("blocked", False)})
    return pd.DataFrame(rows)
