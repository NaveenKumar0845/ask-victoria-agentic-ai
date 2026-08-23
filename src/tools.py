from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .intelligence import infer_query_aspects, summarize_reviews
from .recommendation import rank_recommendations, recommendation_evidence
from .retrieval import RetailRetriever, extract_constraints


@dataclass
class RetailTools:
    products: pd.DataFrame
    reviews: pd.DataFrame
    retriever: RetailRetriever

    def search_products(self, query: str, top_k: int = 5) -> tuple[list[dict], list[str]]:
        chosen = self.retriever.search_products(query, top_k=top_k, **extract_constraints(query))
        products = chosen.to_dict("records")
        evidence = [
            f"{r['product_id']} | {r['name']} | ₹{int(r['price'])} | {r['color']} | {r['material']} | "
            f"{r['support']} | padding={r['padding']} | {r['description']}"
            for _, r in chosen.iterrows()
        ]
        return products, evidence

    def recommend_products(self, query: str, top_k: int = 5) -> tuple[list[dict], list[str]]:
        candidates = self.retriever.search_products(query, top_k=max(top_k, 8), **extract_constraints(query))
        ranked = rank_recommendations(query, candidates, self.reviews).head(top_k)
        products = ranked.to_dict("records")
        evidence = [
            f"{r['product_id']} | {r['name']} | ₹{int(r['price'])} | {r['color']} | {r['material']} | "
            f"{r['support']} | {r['description']}"
            for _, r in ranked.iterrows()
        ]
        evidence.extend(recommendation_evidence(ranked, limit=min(3, top_k)))
        return products, evidence

    def review_intelligence(self, query: str, product_ids: list[str], top_k: int = 8) -> list[str]:
        evidence: list[str] = []
        review_hits = self.retriever.search_reviews(query, product_ids=product_ids, top_k=top_k)
        requested_aspects = infer_query_aspects(query)

        for pid in product_ids[:3]:
            subset = self.reviews[self.reviews["product_id"] == pid]
            if subset.empty:
                continue
            summary = summarize_reviews(subset)
            name = self.products.loc[self.products["product_id"] == pid, "name"].iloc[0]
            evidence.append(
                f"Review intelligence for {pid} {name}: rating {summary['average_rating']}/5 from "
                f"{summary['review_count']} synthetic reviews; review confidence {summary['confidence']:.0%}. "
                f"{summary['love']} {summary['mention']} Fit signal: {summary['fit_signal']['label']}."
            )

            aspects_to_report = requested_aspects or list(summary.get("aspect_scores", {}).keys())[:3]
            for aspect in aspects_to_report[:4]:
                details = summary.get("aspect_scores", {}).get(aspect)
                if not details:
                    continue
                evidence.append(
                    f"Aspect evidence for {pid} {aspect}: {details['positive_rate']:.0%} positive-equivalent "
                    f"across {details['mentions']} mentions; confidence {details['confidence']:.0%}."
                )
                for quote in summary.get("representative_quotes", {}).get(aspect, [])[:1]:
                    evidence.append(f"Representative customer quote for {pid} {aspect}: {quote}")

        for _, row in review_hits.head(5).iterrows():
            evidence.append(
                f"Customer review for {row['product_id']} ({int(row['rating'])}/5): {row['review_text']}"
            )
        return evidence

    def compare_products(self, query: str, top_k: int = 3) -> tuple[list[dict], list[str]]:
        chosen = self.retriever.search_products(query, top_k=top_k)
        products = chosen.to_dict("records")
        evidence: list[str] = []
        for product in products:
            subset = self.reviews[self.reviews["product_id"] == product["product_id"]]
            summary = summarize_reviews(subset)
            top_aspects = sorted(
                summary.get("aspect_scores", {}).items(),
                key=lambda item: (item[1].get("mentions", 0), item[1].get("positive_rate", 0)),
                reverse=True,
            )[:3]
            aspect_text = ", ".join(
                f"{aspect} {details['positive_rate']:.0%} positive"
                for aspect, details in top_aspects
            )
            evidence.append(
                f"{product['name']} ({product['product_id']}): ₹{int(product['price'])}, {product['support']}, "
                f"{product['material']}, rating {summary['average_rating']}/5 from {summary['review_count']} reviews, "
                f"fit={summary['fit_signal']['label']}. Review aspects: {aspect_text}. "
                f"{summary['love']} {summary['mention']}"
            )
        return products, evidence
