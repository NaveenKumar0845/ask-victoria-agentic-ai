# Ask Victoria — Agentic Product Intelligence & Conversational Commerce Platform

A zero-cost, portfolio-ready agentic AI retail assistant built with **LangGraph, Streamlit and retrieval over product + customer-review evidence**.

> Independent educational portfolio project using synthetic retail data. Not affiliated with or endorsed by Victoria's Secret or any retail brand.

## What this project demonstrates

- **Supervisor/router agent** for intent-based orchestration
- **Product agent** for product discovery and structured specifications
- **Review intelligence agent** for review retrieval, aspect extraction and sentiment signals
- **Recommendation workflow** that combines product constraints with customer evidence
- **Comparison workflow** across product specs and review intelligence
- **RAG-style evidence grounding** over product and customer-review corpora
- **Judge node + fallback/self-correction path** to avoid unsupported answers
- **Streamlit product-detail + conversational UI**
- **Optional Gemini generation**, with a deterministic zero-cost fallback
- **Automated tests** for routing, retrieval and end-to-end execution

## Architecture

```text
Customer Query
     │
     ▼
Supervisor / Router (LangGraph)
     │
 ┌───┼───────────────┐
 ▼   ▼               ▼
Product Agent   Review Agent   Comparison / Recommendation
 │        │               │
 └────────┴───────┬───────┘
                  ▼
            Evidence Layer
        Products + Reviews +
         Aspect Intelligence
                  │
                  ▼
             Answer Node
                  │
                  ▼
              Judge Node
              /       \
            PASS      FAIL
             │          │
             ▼          ▼
          Response   Safe Retry
```

## Review intelligence

The demo extracts recurring aspects including **comfort, fit, support, material, activity, padding and durability**, then derives balanced positive/negative signals and product-level summaries.

## Zero-cost design

The app runs without any paid API:

- Streamlit — UI
- LangGraph — orchestration
- scikit-learn TF-IDF + cosine similarity — retrieval
- synthetic public-safe demo data — product/review corpus
- deterministic grounded answer fallback — no API needed

If you add a free Gemini API key, the same evidence is sent to Gemini for natural-language synthesis.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Optional Gemini

Create `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY="your-key"
```

For Streamlit Community Cloud, add `GEMINI_API_KEY` under **App settings → Secrets**.

## Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app from this repository.
3. Set the main file path to `streamlit_app.py`.
4. Optionally add `GEMINI_API_KEY` in Secrets.
5. Deploy.

The application remains functional if no Gemini key is configured.

## Example questions

- `Find me a black sports bra under ₹2000 for yoga`
- `Which sports bra is most comfortable according to customers?`
- `Does the Everyday Cloud Sports Bra run small?`
- `Compare the Everyday Cloud Sports Bra with the Sculpt Medium Support Bra`

## Repository structure

```text
.
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── .streamlit/config.toml
├── src/
│   ├── data.py
│   ├── intelligence.py
│   ├── retrieval.py
│   ├── llm.py
│   └── graph.py
└── tests/
    └── test_core.py
```

## Responsible AI design

The assistant is instructed and gated to avoid inventing product claims, health/medical claims, inventory, or unsupported specifications. When available evidence is insufficient, the expected behavior is to say so rather than guess.

## Next production upgrades

- Sentence-transformer embeddings + vector database
- Persistent MongoDB product/review intelligence collections
- Long-term shopper preference memory
- Offline topic discovery pipeline
- LLM-as-Judge evaluation dashboard
- Prompt-injection and PII guardrails
- FastAPI service layer
- Observability and latency/cost telemetry
- Richer catalogue UX
