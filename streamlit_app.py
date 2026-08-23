from __future__ import annotations

import streamlit as st

from src.data import load_products, load_reviews
from src.intelligence import summarize_reviews
from src.graph import ask_victoria

st.set_page_config(page_title="Ask Victoria", page_icon="✨", layout="wide")

st.markdown("""
<style>
.hero {padding: 1.2rem 1.4rem; border-radius: 18px; background: linear-gradient(135deg,#fff0f6,#f8f2ff); margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2.3rem;}
.small {opacity:.75; font-size:.92rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>✨ Ask Victoria</h1><p>Agentic Product Intelligence & Conversational Commerce Platform</p><p class="small">Independent portfolio demo using synthetic retail data. Not affiliated with Victoria\'s Secret.</p></div>', unsafe_allow_html=True)

products = load_products()
reviews = load_reviews()

with st.sidebar:
    st.header("Product Explorer")
    selected_name = st.selectbox("Choose a product", products["name"].tolist())
    p = products[products["name"] == selected_name].iloc[0]
    summary = summarize_reviews(reviews[reviews["product_id"] == p["product_id"]])
    st.metric("Price", f"₹{int(p['price'])}")
    st.metric("Rating", f"{summary['average_rating']} / 5")
    st.caption(f"{summary['review_count']} demo reviews")
    st.write(f"**Category:** {p['category']}")
    st.write(f"**Color:** {p['color']}")
    st.write(f"**Material:** {p['material']}")
    st.write(f"**Support:** {p['support']}")
    st.write(f"**Padding:** {p['padding']}")

left, right = st.columns([1.05, 1.45])
with left:
    st.subheader(selected_name)
    st.write(p["description"])
    st.markdown("#### What customers love")
    st.success(summary["love"])
    st.markdown("#### What customers mention")
    st.warning(summary["mention"])
    st.markdown("#### Aspect intelligence")
    if summary["aspects"]:
        st.json(summary["aspects"], expanded=False)

with right:
    st.subheader("Ask the shopping assistant")
    prompts = [
        "Find me a black sports bra under ₹2000 for yoga",
        "Which sports bra is most comfortable according to customers?",
        "Does the Everyday Cloud Sports Bra run small?",
        "Compare the Everyday Cloud Sports Bra with the Sculpt Medium Support Bra",
    ]
    cols = st.columns(2)
    for i, q in enumerate(prompts):
        if cols[i % 2].button(q, use_container_width=True):
            st.session_state["prefill"] = q

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prefill = st.session_state.pop("prefill", "")
    query = st.chat_input("Ask about products, reviews, fit, support, price or comparisons…")
    query = query or prefill
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Routing across product and review agents…"):
                result = ask_victoria(query)
            answer = result.get("final_answer", "I couldn't produce a grounded answer.")
            st.markdown(answer)
            with st.expander("Agent trace"):
                st.write({"intent": result.get("intent"), "selected_product_ids": result.get("selected_product_ids", []), "grounded": result.get("grounded"), "retry_count": result.get("retry_count", 0)})
        st.session_state.messages.append({"role":"assistant","content":answer})

st.divider()
st.caption("Zero-cost architecture: Streamlit + LangGraph + scikit-learn retrieval. Gemini is optional through GEMINI_API_KEY; the app has a deterministic fallback so it remains functional without paid APIs.")
