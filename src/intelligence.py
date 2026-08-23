from __future__ import annotations

from collections import Counter
import pandas as pd
from .data import ASPECT_KEYWORDS

POSITIVE_WORDS = {"love","great","excellent","soft","comfortable","comfort","supportive","secure","breathable","favorite","flattering","durable","lightweight","stable","perfect"}
NEGATIVE_WORDS = {"small","tight","narrow","wide","not","firm","oversized","little"}


def aspect_sentiment(review_text: str) -> dict[str, str]:
    text = review_text.lower()
    tokens = set(text.replace(".", "").replace(",", "").split())
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    sentiment = "positive" if pos > neg else "negative" if neg > pos else "mixed"
    results = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(k in text for k in keywords):
            results[aspect] = sentiment
    return results


def summarize_reviews(reviews: pd.DataFrame) -> dict:
    if reviews.empty:
        return {"love": "No customer review evidence yet.", "mention": "No review concerns available.", "aspects": {}, "average_rating": 0.0, "review_count": 0}
    aspect_counts: dict[str, Counter] = {}
    for text in reviews["review_text"].astype(str):
        for aspect, sentiment in aspect_sentiment(text).items():
            aspect_counts.setdefault(aspect, Counter())[sentiment] += 1
    ranked_positive = sorted(((a, c.get("positive", 0), sum(c.values())) for a, c in aspect_counts.items()), key=lambda x: (x[1], x[2]), reverse=True)
    ranked_negative = sorted(((a, c.get("negative", 0), sum(c.values())) for a, c in aspect_counts.items()), key=lambda x: (x[1], x[2]), reverse=True)
    love_aspects = [a for a, p, _ in ranked_positive if p > 0][:3] or ["overall experience"]
    concern_aspects = [a for a, n, _ in ranked_negative if n > 0][:2]
    return {
        "love": "Customers most often praise " + ", ".join(love_aspects) + ".",
        "mention": ("Some reviewers mention " + " and ".join(concern_aspects) + " concerns.") if concern_aspects else "No recurring negative theme dominates the available reviews.",
        "aspects": {a: dict(c) for a, c in aspect_counts.items()},
        "average_rating": round(float(reviews["rating"].mean()), 2),
        "review_count": int(len(reviews)),
    }
