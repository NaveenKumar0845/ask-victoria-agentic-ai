from __future__ import annotations

import pandas as pd
import streamlit as st

from src.e2e_evaluation import evaluate_end_to_end
from src.graph import ask_victoria
from src.observability import telemetry_from_result, trace_stage_counts


BUILD_ID = "2026-08-23-observability-r3"

st.set_page_config(page_title="Ask Victoria · Observability", page_icon="📡", layout="wide")
st.title("📡 Agent Observability")
st.caption(
    "Transparent runtime telemetry for the controlled portfolio environment. "
    "No external paid observability service is required."
)
st.caption(f"Build: `{BUILD_ID}` · telemetry is recomputed fresh on every page run")

if st.button("↻ Refresh telemetry", use_container_width=False):
    st.rerun()

with st.spinner("Running fresh benchmark telemetry…"):
    frame, summary, results = evaluate_end_to_end(ask_victoria)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Benchmark runs", summary["runs"])
m2.metric("Avg latency", f"{summary['avg_latency_ms']:.0f} ms")
m3.metric("P95 latency", f"{summary['p95_latency_ms']:.0f} ms")
m4.metric("Judge pass rate", f"{summary['judge_pass_rate']:.0%}")
m5.metric("Retry rate", f"{summary['retry_rate']:.0%}")

m6, m7, m8, m9 = st.columns(4)
m6.metric("Avg groundedness proxy", f"{summary['avg_groundedness']:.0%}")
m7.metric("Avg evidence items", f"{summary['avg_evidence_items']:.1f}")
m8.metric("Avg tool events", f"{summary['avg_tool_events']:.1f}")
m9.metric("Safety block rate", f"{summary['block_rate']:.0%}")

st.markdown("### Run-level telemetry")
telemetry_rows = []
for case_result, (_, case_row) in zip(results, frame.iterrows()):
    telemetry = telemetry_from_result(case_result).as_dict()
    telemetry_rows.append({"case": case_row["case"], **telemetry})
st.dataframe(pd.DataFrame(telemetry_rows), hide_index=True, use_container_width=True)

st.markdown("### Latency by benchmark case")
latency_frame = frame[["case", "latency_ms"]].set_index("case")
st.bar_chart(latency_frame)

st.markdown("### Agent-stage activity")
st.caption("Counts are inferred from the explicit LangGraph execution trace emitted by each run.")
stage_rows = []
for case_result, (_, case_row) in zip(results, frame.iterrows()):
    stage_rows.append({"case": case_row["case"], **trace_stage_counts(case_result)})
st.dataframe(pd.DataFrame(stage_rows), hide_index=True, use_container_width=True)

st.markdown("### Inspect a trace")
case_names = frame["case"].tolist()
selected_case = st.selectbox("Benchmark case", case_names)
case_index = case_names.index(selected_case)
selected_result = results[case_index]
selected_telemetry = telemetry_from_result(selected_result)

a, b, c, d = st.columns(4)
a.metric("Intent", selected_telemetry.intent)
b.metric("Evidence", selected_telemetry.evidence_count)
c.metric("Groundedness", f"{selected_telemetry.groundedness:.0%}")
d.metric("Latency", f"{selected_telemetry.latency_ms:.0f} ms")

if selected_result.get("safety_category"):
    st.caption(f"Diagnostic category: `{selected_result.get('safety_category')}`")

for step_number, step in enumerate(selected_result.get("trace", []), start=1):
    st.write(f"**{step_number}.** {step}")

st.info(
    "Portfolio note: these metrics are generated from the controlled synthetic benchmark. "
    "They demonstrate observability design and should not be described as production traffic statistics."
)
