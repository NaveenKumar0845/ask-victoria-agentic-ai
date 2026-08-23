from __future__ import annotations

from collections import Counter
import re

import pandas as pd

from .data import ASPECT_KEYWORDS

POSITIVE_WORDS = {
    "love", "great", "excellent", "soft", "comfortable", "comfort", "supportive",
    "secure", "breathable", "favorite", "flattering", "durable", "lightweight",
    "stable", "perfect", "smooth", "flexible", "easy", "good", "cushioned",
}
NEGATIVE_WORDS = {
    "small", "tight", "narrow", "wide", "firm", "oversized", "uncomfortable",
    "poor", "stiff", "scratchy", "loose", "slips", "digging", "issue", "concern",
}
NEGATION_WORDS = {"not", "no", "never", "isn't", "wasn't", "don't", "doesn't"}

QUERY_ASPECT_TERMS = {
    "comfort": ["comfortable", "comfort", "soft", "cushioned", "relaxed", "easy wear"],
    "fit": ["fit", "size", "sizing", "runs small", "run small", "true to size", "tight", "loose"],
    "support": ["support", "supportive", "secure", "stable", "compression"],
    "material": ["material", "fabric", "breathable", "mesh", "modal", "cotton", "knit"],
    "activity": ["yoga", "pilates", "gym", "training", "running", "walking", "studio", "workout"],
    "padding": ["padding", "padded", "removable pad", "fixed pad"],
    "durability": ["durable", "durability", "quality", "wash", "long lasting"],
    "style": ["style", "flattering", "cute", "looks", "color"],
}


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _sentiment_for_text(text: str) -> str:
    tokens = _word_tokens(text)
    pos = 0
    neg = 0
    for idx, token in enumerate(tokens):
        window = set(tokens[max(0, idx - 2):idx])
        negated = bool(window & NEGATION_WORDS)
        if token in POSITIVE_WORDS:
            if negated:
                neg += 1
            else:
                pos += 1
        elif token in NEGATIVE_WORDS:
            if negated:
                pos += 1
            else:
                neg += 1
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "mixed"


def aspect_sentiment(review_text: str) -> dict[str, str]:
    text = review_text.lower()
    sentiment = _sentiment_for_text(text)
    results: dict[str, str] = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            results[aspect] = sentiment
    return results


def infer_query_aspects(query: str) -> list[str]:
    q = query.lower()
    return [aspect for aspect, terms in QUERY_ASPECT_TERMS.items() if any(term in q for term in terms)]


def _fit_signal(review_texts: list[str]) -> dict:
    combined = " ".join(review_texts).lower()
    small_terms = ["runs small", "run small", "size up", "tight", "snug"]
    large_terms = ["runs large", "run large", "size down", "oversized", "loose"]
    true_terms = ["true to size", "fits well", "great fit", "good fit"]
    small = sum(combined.count(t) for t in small_terms)
    large = sum(combined.count(t) for t in large_terms)
    true = sum(combined.count(t) for t in true_terms)
    total = small + large + true
    if total == 0:
        return {"label": "insufficient fit evidence", "small": 0, "true_to_size": 0, "large": 0}
    if small > max(large, true):
        label = "tends to run small"
    elif large > max(small, true):
        label = "tends to run large"
    elif true > 0:
        label = "generally true to size"
    else:
        label = "mixed fit feedback"
    return {"label": label, "small": small, "true_to_size": true, "large": large}


def _representative_quotes(reviews: pd.DataFrame, aspect: str, limit: int = 2) -> list[str]:
    keywords = ASPECT_KEYWORDS.get(aspect, [])
    scored: list[tuple[int, int, str]] = []
    for _, row in reviews.iterrows():
        text = str(row["review_text"])
        lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword in lower)
        if matches:
            scored.append((matches, int(row.get("rating", 0)), text))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    seen: set[str] = set()
    quotes: list[str] = []
    for _, _, text in scored:
        if text not in seen:
            quotes.append(text)
            seen.add(text)
        if len(quotes) >= limit:
            break
    return quotes


def summarize_reviews(reviews: pd.DataFrame) -> dict:
    if reviews.empty:
        return {
            "love": "No customer review evidence yet.",
            "mention": "No review concerns available.",
            "aspects": {},
            "aspect_scores": {},
            "average_rating": 0.0,
            "review_count": 0,
            "fit_signal": {"label": "insufficient fit evidence", "small": 0, "true_to_size": 0, "large": 0},
            "confidence": 0.0,
            "representative_quotes": {},
        }

    aspect_counts: dict[str, Counter] = {}
    review_texts = reviews["review_text"].astype(str).tolist()
    for text in review_texts:
        for aspect, sentiment in aspect_sentiment(text).items():
            aspect_counts.setdefault(aspect, Counter())[sentiment] += 1

    aspect_scores: dict[str, dict] = {}
    for aspect, counts in aspect_counts.items():
        positive = counts.get("positive", 0)
        negative = counts.get("negative", 0)
        mixed = counts.get("mixed", 0)
        total = positive + negative + mixed
        # Mixed mentions contribute half credit to avoid forcing neutral statements positive/negative.
        positive_rate = ((positive + 0.5 * mixed) / total) if total else 0.0
        aspect_scores[aspect] = {
            "positive": positive,
            "negative": negative,
            "mixed": mixed,
            "mentions": total,
            "positive_rate": round(positive_rate, 3),
            "confidence": round(min(1.0, total / 12.0), 3),
        }

    ranked_positive = sorted(
        ((a, v["positive_rate"], v["mentions"]) for a, v in aspect_scores.items()),
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )
    ranked_negative = sorted(
        (
            (a, (v["negative"] / v["mentions"]) if v["mentions"] else 0.0, v["mentions"])
            for a, v in aspect_scores.items()
        ),
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )

    love_aspects = [a for a, score, mentions in ranked_positive if score >= 0.60 and mentions > 0][:3]
    concern_aspects = [a for a, rate, mentions in ranked_negative if rate >= 0.20 and mentions > 0][:2]
    if not love_aspects:
        love_aspects = [a for a, _, _ in ranked_positive[:2]] or ["overall experience"]

    representative_quotes = {
        aspect: _representative_quotes(reviews, aspect)
        for aspect in list(dict.fromkeys(love_aspects + concern_aspects))
    }

    return {
        "love": "Customers most often praise " + ", ".join(love_aspects) + ".",
        "mention": (
            "Some reviewers mention " + " and ".join(concern_aspects) + " concerns."
            if concern_aspects
            else "No recurring negative theme dominates the available reviews."
        ),
        "aspects": {a: dict(c) for a, c in aspect_counts.items()},
        "aspect_scores": aspect_scores,
        "average_rating": round(float(reviews["rating"].mean()), 2),
        "review_count": int(len(reviews)),
        "fit_signal": _fit_signal(review_texts),
        "confidence": round(min(1.0, len(reviews) / 20.0), 3),
        "representative_quotes": representative_quotes,
    }
