from __future__ import annotations

import streamlit as st

from src.e2e_evaluation import E2E_CASES, evaluate_end_to_end
from src.graph import ask_victoria


BUILD_ID = "2026-08-23-e2e-r3"

st.set_page_config(page_title="Ask Victoria · End-to-End Evaluation", page_icon="🧪", layout="wide")
st.title("🧪 End-to-End Agent Evaluation")
st.caption(
    "Controlled evaluation across routing, product selection, safety behavior, answer completion, grounding, retries and latency."
)
st.caption(f"Build: `{BUILD_ID}` · benchmark is recomputed fresh on every page run")

if st.button("↻ Run benchmark again", use_container_width=False):
    st.rerun()

with st.spinner("Running the controlled agent benchmark…"):
    frame, summary, _ = evaluate_end_to_end(ask_victoria)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Task success", f"{summary['task_success_rate']:.0%}")
m2.metric("Routing success", f"{summary['routing_success_rate']:.0%}")
m3.metric("Product selection", f"{summary['product_selection_rate']:.0%}")
m4.metric("Safety success", f"{summary['safety_success_rate']:.0%}")
m5.metric("Grounding pass", f"{summary['grounding_pass_rate']:.0%}")

m6, m7, m8, m9 = st.columns(4)
m6.metric("Cases", summary["cases"])
m7.metric("Average latency", f"{summary['avg_latency_ms']:.0f} ms")
m8.metric("P95 latency", f"{summary['p95_latency_ms']:.0f} ms")
m9.metric("Retry rate", f"{summary['retry_rate']:.0%}")

st.markdown("### Case-level results")
st.dataframe(frame, hide_index=True, use_container_width=True)

failures = frame[~frame["success"]]
st.markdown("### Failure analysis")
if failures.empty:
    st.success("All cases in the current controlled benchmark passed.")
else:
    st.warning(f"{len(failures)} of {len(frame)} benchmark cases need attention.")
    failure_columns = [
        "case",
        "routing",
        "product_selection",
        "safety",
        "answer",
        "grounded",
        "intent",
        "diagnostic_category",
        "selected_products",
    ]
    available_columns = [column for column in failure_columns if column in failures.columns]
    st.dataframe(failures[available_columns], hide_index=True, use_container_width=True)

retry_cases = frame[frame["retry_count"] > 0]
st.markdown("### Retry / self-correction analysis")
if retry_cases.empty:
    st.success("No benchmark case required the self-correction path in this run.")
else:
    retry_columns = [
        "case",
        "intent",
        "diagnostic_category",
        "retry_count",
        "groundedness",
        "latency_ms",
        "selected_products",
    ]
    available_retry_columns = [column for column in retry_columns if column in retry_cases.columns]
    st.dataframe(retry_cases[available_retry_columns], hide_index=True, use_container_width=True)
    st.caption(
        "A retry is not automatically a task failure. It means the Judge rejected the first draft and the LangGraph self-correction path produced the final safe response."
    )

st.markdown("### Benchmark coverage")
coverage_rows = []
for case in E2E_CASES:
    coverage_rows.append(
        {
            "case": case["name"],
            "intent / safety target": case.get("expected_intent", case.get("expected_safety_category", "")),
            "context-aware": bool(case.get("context")),
            "product validation": bool(case.get("expected_products")),
            "answer-term validation": bool(case.get("expected_answer_terms")),
        }
    )
st.dataframe(coverage_rows, hide_index=True, use_container_width=True)

st.info(
    "Interpretation: this is a deterministic portfolio benchmark over synthetic data, not a production SLA or external user-study result. "
    "Its purpose is to make system behavior measurable, reproducible and debuggable."
)
