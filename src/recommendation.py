from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .intelligence import infer_query_aspects, summarize_reviews
from .retrieval import extract_constraints


@dataclass
class RecommendationBreakdown:
    product_id: str
    final_score: float
    retrieval_score: float
    aspect_score: float
    rating_score: float
    confidence_score: float
    value_score: float
    reasons: list[str]


def _normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    low = float(series.min())
    high = float(series.max())
    if abs(high - low) < 1e-9:
        return pd.Series(np.ones(len(series)), index=series.index, dtype=float)
    return (series - low) / (high - low)


def _aspect_score(summary: dict, requested_aspects: list[str]) -> tuple[float, list[str]]:
    scores = summary.get("aspect_scores", {})
    reasons: list[str] = []
    if requested_aspects:
        selected = []
        for aspect in requested_aspects:
            details = scores.get(aspect)
            if details:
                selected.append(float(details["positive_rate"]))
                reasons.append(
                    f"{aspect}: {details['positive_rate']:.0%} positive across {details['mentions']} mentions"
                )
        if selected:
            return float(np.mean(selected)), reasons
        return 0.45, ["limited review evidence for the requested aspect"]

    prominent = sorted(
        scores.items(),
        key=lambda item: (item[1].get("mentions", 0), item[1].get("positive_rate", 0)),
        reverse=True,
    )[:3]
    if not prominent:
        return 0.5, ["limited aspect-level review evidence"]
    values = [float(details["positive_rate"]) for _, details in prominent]
    reasons.extend(
        f"{aspect}: {details['positive_rate']:.0%} positive"
        for aspect, details in prominent[:2]
    )
    return float(np.mean(values)), reasons


def rank_recommendations(
    query: str,
    candidates: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Explainable recommendation ranking over retrieved candidates.

    Final score weights:
      42% retrieval relevance
      25% requested review-aspect sentiment
      15% average rating
      10% review evidence confidence
       8% value/price fit

    This layer ranks only already-retrieved candidates; it does not bypass hard
    filters such as category, color or maximum price.
    """
    if candidates.empty:
        return candidates.copy()

    ranked = candidates.copy().reset_index(drop=True)
    if "score" not in ranked.columns:
        ranked["score"] = 0.0
    ranked["retrieval_component"] = _normalize(ranked["score"].astype(float))

    requested_aspects = infer_query_aspects(query)
    constraints = extract_constraints(query)
    max_price = constraints.get("max_price")

    aspect_components: list[float] = []
    rating_components: list[float] = []
    confidence_components: list[float] = []
    value_components: list[float] = []
    reasons_col: list[list[str]] = []
    fit_signals: list[str] = []
    avg_ratings: list[float] = []
    review_counts: list[int] = []

    min_price = float(ranked["price"].min())
    max_candidate_price = float(ranked["price"].max())
    price_span = max(max_candidate_price - min_price, 1.0)

    for _, product in ranked.iterrows():
        subset = reviews[reviews["product_id"] == product["product_id"]]
        summary = summarize_reviews(subset)

        aspect_component, aspect_reasons = _aspect_score(summary, requested_aspects)
        rating = float(summary.get("average_rating", 0.0))
        rating_component = max(0.0, min(1.0, (rating - 1.0) / 4.0))
        confidence_component = float(summary.get("confidence", 0.0))

        price = float(product["price"])
        if max_price is not None and max_price > 0:
            # Among valid products, reward headroom below the user's ceiling.
            value_component = max(0.0, min(1.0, (max_price - price) / max_price + 0.5))
        else:
            value_component = 1.0 - ((price - min_price) / price_span)

        reasons = list(aspect_reasons)
        reasons.append(f"rating: {rating:.2f}/5 from {summary.get('review_count', 0)} reviews")
        if max_price is not None:
            reasons.append(f"price ₹{int(price)} within ₹{int(max_price)} budget")
        reasons.append(f"fit signal: {summary.get('fit_signal', {}).get('label', 'insufficient fit evidence')}")

        aspect_components.append(aspect_component)
        rating_components.append(rating_component)
        confidence_components.append(confidence_component)
        value_components.append(value_component)
        reasons_col.append(reasons)
        fit_signals.append(summary.get("fit_signal", {}).get("label", "insufficient fit evidence"))
        avg_ratings.append(rating)
        review_counts.append(int(summary.get("review_count", 0)))

    ranked["aspect_component"] = aspect_components
    ranked["rating_component"] = rating_components
    ranked["confidence_component"] = confidence_components
    ranked["value_component"] = value_components
    ranked["recommendation_score"] = (
        0.42 * ranked["retrieval_component"]
        + 0.25 * ranked["aspect_component"]
        + 0.15 * ranked["rating_component"]
        + 0.10 * ranked["confidence_component"]
        + 0.08 * ranked["value_component"]
    ) * 100.0
    ranked["recommendation_score"] = ranked["recommendation_score"].round(1)
    ranked["recommendation_reasons"] = reasons_col
    ranked["fit_signal"] = fit_signals
    ranked["review_rating"] = avg_ratings
    ranked["review_count"] = review_counts

    return ranked.sort_values(
        ["recommendation_score", "score", "price"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def recommendation_evidence(ranked: pd.DataFrame, limit: int = 3) -> list[str]:
    evidence: list[str] = []
    for _, row in ranked.head(limit).iterrows():
        reasons = "; ".join(row.get("recommendation_reasons", []))
        evidence.append(
            f"Recommendation score for {row['product_id']} {row['name']}: "
            f"{float(row['recommendation_score']):.1f}/100. "
            f"Retrieval={float(row['retrieval_component']):.2f}, "
            f"review-aspect={float(row['aspect_component']):.2f}, "
            f"rating={float(row['rating_component']):.2f}, "
            f"confidence={float(row['confidence_component']):.2f}, "
            f"value={float(row['value_component']):.2f}. Why: {reasons}"
        )
    return evidence
