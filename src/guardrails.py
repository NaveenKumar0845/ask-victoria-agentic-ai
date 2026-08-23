from __future__ import annotations

import re
from dataclasses import dataclass

PROMPT_INJECTION_PATTERNS = ("ignore previous", "ignore all previous", "ignore your instructions", "reveal your system prompt", "show me your system prompt", "developer message", "jailbreak", "act as unrestricted")
MEDICAL_PATTERNS = ("cure", "treat my", "treats pain", "diagnose", "heal", "medical advice", "prevent injury")
UNSUPPORTED_CLAIM_PATTERNS = ("guaranteed", "clinically proven", "doctor approved", "will cure", "will treat")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s-]{7,}\d)(?!\d)")

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
        return GuardrailDecision(False, "prompt_injection", "I can help with products, reviews and shopping decisions, but I can't reveal or override internal instructions.")
    if any(pattern in q for pattern in MEDICAL_PATTERNS):
        return GuardrailDecision(False, "medical_claim", "I can summarize product features and customer feedback, but I can't provide medical advice or claim that a retail product can diagnose, cure, treat, or prevent a health condition.")
    return GuardrailDecision(True)

def check_output(answer: str, evidence: list[str]) -> GuardrailDecision:
    a = answer.lower()
    if any(pattern in a for pattern in UNSUPPORTED_CLAIM_PATTERNS):
        return GuardrailDecision(False, "unsupported_claim", "The generated answer contains an unsupported claim.")
    if not evidence:
        return GuardrailDecision(False, "no_evidence", "No supporting product or review evidence was retrieved.")
    return GuardrailDecision(True)
