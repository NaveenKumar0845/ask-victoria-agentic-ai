from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.data import load_products
from src.graph import ask_victoria

app = FastAPI(
    title="Ask Victoria API",
    description="Agentic Product Intelligence & Conversational Commerce API",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    context: dict[str, Any] = Field(default_factory=dict)
    conversation: list[dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    intent: str | None = None
    selected_product_ids: list[str] = Field(default_factory=list)
    grounded: bool = False
    groundedness_score: float = 0.0
    blocked: bool = False
    retry_count: int = 0
    latency_ms: float = 0.0
    trace: list[str] = Field(default_factory=list)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ask-victoria"}


@app.get("/products")
def products() -> list[dict]:
    return load_products().to_dict("records")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = ask_victoria(
        request.query,
        context=request.context,
        conversation=request.conversation,
    )
    return ChatResponse(
        answer=result.get("final_answer", ""),
        intent=result.get("intent"),
        selected_product_ids=result.get("selected_product_ids", []),
        grounded=result.get("grounded", False),
        groundedness_score=float(result.get("groundedness_score", 0.0) or 0.0),
        blocked=result.get("blocked", False),
        retry_count=result.get("retry_count", 0),
        latency_ms=float(result.get("latency_ms", 0.0) or 0.0),
        trace=result.get("trace", []),
    )
