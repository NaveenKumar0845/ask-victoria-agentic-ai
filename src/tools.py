from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .intelligence import summarize_reviews
from .retrieval import RetailRetriever, extract_constraints

@dataclass
class RetailTools:
    products: pd.DataFrame
    reviews: pd.DataFrame
    retriever: RetailRetriever

    def search_products(self, query: str, top_k: int = 5) -> tuple[list[dict], list[str]]:
        chosen = self.retriever.search_products(query, top_k=top_k, **extract_constraints(query))
        products = chosen.to_dict("records")
        evidence = [f"{r['product_id']} | {r['name']} | ₹{int(r['price'])} | {r['color']} | {r['material']} | {r['support']} | padding={r['padding']} | {r['description']}" for _, r in chosen.iterrows()]
        return products, evidence

    def review_intelligence(self, query: str, product_ids: list[str], top_k: int = 8) -> list[str]:
        evidence: list[str] = []
        review_hits = self.retriever.search_reviews(query, product_ids=product_ids, top_k=top_k)
        for pid in product_ids[:3]:
            subset = self.reviews[self.reviews["product_id"] == pid]
            if subset.empty:
                continue
            summary = summarize_reviews(subset)
            name = self.products.loc[self.products["product_id"] == pid, "name"].iloc[0]
            evidence.append(f"Review intelligence for {pid} {name}: rating {summary['average_rating']}/5 from {summary['review_count']} demo reviews. {summary['love']} {summary['mention']}")
        for _, row in review_hits.head(5).iterrows():
            evidence.append(f"Customer review for {row['product_id']} ({int(row['rating'])}/5): {row['review_text']}")
        return evidence

    def compare_products(self, query: str, top_k: int = 3) -> tuple[list[dict], list[str]]:
        chosen = self.retriever.search_products(query, top_k=top_k)
        products = chosen.to_dict("records")
        evidence = []
        for product in products:
            subset = self.reviews[self.reviews["product_id"] == product["product_id"]]
            summary = summarize_reviews(subset)
            evidence.append(f"{product['name']} ({product['product_id']}): ₹{int(product['price'])}, {product['support']}, {product['material']}, rating {summary['average_rating']}/5. {summary['love']} {summary['mention']}")
        return products, evidence
