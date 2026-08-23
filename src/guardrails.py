from __future__ import annotations

import re
from dataclasses import dataclass

PROMPT_INJECTION_PATTERNS = (
    "ignore previous",
    "ignore all previous",
    "ignore your instructions",
    "reveal your system prompt",
    "show me your system prompt",
    "developer message",
    "jailbreak",
    "act as unrestricted",
)
MEDICAL_PATTERNS = (
    "cure",
    "treat my",
    "treats pain",
    "diagnose",
    "heal",
    "medical advice",
    "prevent injury",
)
UNSUPPORTED_CLAIM_PATTERNS = (
    "guaranteed",
    "clinically proven",
    "doctor approved",
    "will cure",
    "will treat",
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s-]{7,}\d)(?!\d)")
CLAIM_NUMBER_RE = re.compile(r"₹\s*\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:/5|/100|%)")
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z-]+")
STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "you", "your",
    "can", "could", "would", "should", "based", "available", "product", "products", "evidence", "customer",
    "customers", "review", "reviews", "according", "within", "using", "into", "has", "have", "its", "their",
}


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    category: str = "ok"
    message: str = ""


def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL REDACTED]", text)
    text = PHONE_RE.sub("[PHONE REDACTED]", text)
    return text


def check_input(query: str) -> GuardrailDecision:
    q = query.lower().strip()
    if any(pattern in q for pattern in PROMPT_INJECTION_PATTERNS):
        return GuardrailDecision(
            False,
            "prompt_injection",
            "I can help with products, reviews and shopping decisions, but I can't reveal or override internal instructions.",
        )
    if any(pattern in q for pattern in MEDICAL_PATTERNS):
        return GuardrailDecision(
            False,
            "medical_claim",
            "I can summarize product features and customer feedback, but I can't provide medical advice or claim that a retail product can diagnose, cure, treat, or prevent a health condition.",
        )
    return GuardrailDecision(True)


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in WORD_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    }


def _normalized_claim_numbers(text: str) -> set[str]:
    return {re.sub(r"\s+", "", match).lower() for match in CLAIM_NUMBER_RE.findall(text)}


def _support_ratio(answer: str, evidence: list[str]) -> float:
    answer_tokens = _meaningful_tokens(answer)
    if not answer_tokens:
        return 1.0
    evidence_tokens = _meaningful_tokens(" ".join(evidence))
    return len(answer_tokens & evidence_tokens) / max(1, len(answer_tokens))


def check_output(answer: str, evidence: list[str]) -> GuardrailDecision:
    a = answer.lower()
    if any(pattern in a for pattern in UNSUPPORTED_CLAIM_PATTERNS):
        return GuardrailDecision(False, "unsupported_claim", "The generated answer contains an unsupported claim.")
    if not evidence:
        return GuardrailDecision(False, "no_evidence", "No supporting product or review evidence was retrieved.")

    # Numeric claims are especially risky in commerce because a hallucinated price,
    # rating or score can directly change a purchase decision. Require every explicit
    # numeric claim in the answer to appear in the retrieved evidence.
    answer_numbers = _normalized_claim_numbers(answer)
    evidence_numbers = _normalized_claim_numbers(" ".join(evidence))
    unsupported_numbers = answer_numbers - evidence_numbers
    if unsupported_numbers:
        return GuardrailDecision(
            False,
            "unsupported_numeric_claim",
            "The generated answer contains a price, rating, percentage or score that is not present in the retrieved evidence.",
        )

    # Lightweight deterministic grounding gate. This is intentionally conservative
    # and transparent; it is not presented as a production factuality model.
    support = _support_ratio(answer, evidence)
    if support < 0.12:
        return GuardrailDecision(
            False,
            "low_evidence_overlap",
            "The generated answer has too little lexical support from the retrieved evidence.",
        )

    return GuardrailDecision(True)
