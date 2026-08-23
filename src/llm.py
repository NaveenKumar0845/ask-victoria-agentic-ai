from __future__ import annotations

import os

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


def gemini_answer(prompt: str) -> str | None:
    """Optional Gemini synthesis.

    The project remains fully functional when no key is present. The default
    model is configurable via GEMINI_MODEL so the repository is not coupled to
    one model version.
    """
    key = _secret("GEMINI_API_KEY")
    if not key:
        return None
    model = _secret("GEMINI_MODEL", DEFAULT_MODEL)
    try:
        from google import genai

        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text
    except Exception:
        return None
