from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean


@dataclass
class RunTelemetry:
    intent: str
    blocked: bool
    safety_category: str
    evidence_count: int
    selected_products: int
    groundedness: float
    retry_count: int
    latency_ms: float
    trace_steps: int
    tool_events: int
    judge_passed: bool

    def as_dict(self) -> dict:
        return asdict(self)


def telemetry_from_result(result: dict) -> RunTelemetry:
    trace = result.get("trace", []) or []
    tool_events = sum(
        1
        for step in trace
        if any(term in step.lower() for term in ["called", "retrieved", "ranked", "assembled"])
    )
    blocked = bool(result.get("blocked", False))
    grounded = bool(result.get("grounded", False))
    return RunTelemetry(
        intent=result.get("intent", "safety" if blocked else "unknown"),
        blocked=blocked,
        safety_category=result.get("safety_category", "ok"),
        evidence_count=len(result.get("evidence", []) or []),
        selected_products=len(result.get("selected_product_ids", []) or []),
        groundedness=float(result.get("groundedness_score", 1.0 if blocked else 0.0) or 0.0),
        retry_count=int(result.get("retry_count", 0) or 0),
        latency_ms=float(result.get("latency_ms", 0.0) or 0.0),
        trace_steps=len(trace),
        tool_events=tool_events,
        judge_passed=grounded or blocked,
    )


def aggregate_telemetry(results: list[dict]) -> dict:
    if not results:
        return {
            "runs": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_groundedness": 0.0,
            "block_rate": 0.0,
            "retry_rate": 0.0,
            "judge_pass_rate": 0.0,
            "avg_evidence_items": 0.0,
            "avg_tool_events": 0.0,
        }

    telemetry = [telemetry_from_result(r) for r in results]
    latencies = sorted(t.latency_ms for t in telemetry)
    p95_index = max(0, min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1)))))
    return {
        "runs": len(telemetry),
        "avg_latency_ms": round(mean(t.latency_ms for t in telemetry), 1),
        "p95_latency_ms": round(latencies[p95_index], 1),
        "avg_groundedness": round(mean(t.groundedness for t in telemetry), 3),
        "block_rate": round(mean(float(t.blocked) for t in telemetry), 3),
        "retry_rate": round(mean(float(t.retry_count > 0) for t in telemetry), 3),
        "judge_pass_rate": round(mean(float(t.judge_passed) for t in telemetry), 3),
        "avg_evidence_items": round(mean(t.evidence_count for t in telemetry), 2),
        "avg_tool_events": round(mean(t.tool_events for t in telemetry), 2),
    }


def trace_stage_counts(result: dict) -> dict[str, int]:
    stages = {
        "guardrail": 0,
        "router": 0,
        "product": 0,
        "review": 0,
        "recommendation": 0,
        "comparison": 0,
        "answer": 0,
        "judge": 0,
        "self_correction": 0,
        "finalize": 0,
    }
    for step in result.get("trace", []) or []:
        text = step.lower()
        if "guardrail" in text:
            stages["guardrail"] += 1
        if "supervisor" in text or "routed" in text:
            stages["router"] += 1
        if "product agent" in text:
            stages["product"] += 1
        if "review intelligence agent" in text:
            stages["review"] += 1
        if "recommendation agent" in text:
            stages["recommendation"] += 1
        if "comparison agent" in text:
            stages["comparison"] += 1
        if "answer agent" in text:
            stages["answer"] += 1
        if "judge agent" in text:
            stages["judge"] += 1
        if "self-correction" in text:
            stages["self_correction"] += 1
        if "finalized" in text:
            stages["finalize"] += 1
    return stages
