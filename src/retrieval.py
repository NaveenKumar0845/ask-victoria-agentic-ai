from __future__ import annotations

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RetailRetriever:
    def __init__(self, products: pd.DataFrame, reviews: pd.DataFrame):
        self.products = products.copy()
        self.reviews = reviews.copy()
        self.product_docs = self.products.apply(lambda r: " ".join(str(r.get(c, "")) for c in ["name","category","color","material","support","padding","description"]), axis=1).tolist()
        self.review_docs = self.reviews["review_text"].astype(str).tolist()
        self.product_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.review_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.product_matrix = self.product_vectorizer.fit_transform(self.product_docs)
        self.review_matrix = self.review_vectorizer.fit_transform(self.review_docs)

    def search_products(self, query: str, top_k: int = 5, max_price: float | None = None, color: str | None = None, category: str | None = None) -> pd.DataFrame:
        q = self.product_vectorizer.transform([query])
        result = self.products.copy()
        result["score"] = cosine_similarity(q, self.product_matrix).ravel()
        if max_price is not None:
            result = result[result["price"] <= max_price]
        if color:
            result = result[result["color"].str.lower() == color.lower()]
        if category:
            result = result[result["category"].str.lower().str.contains(category.lower(), regex=False)]
        return result.sort_values(["score","price"], ascending=[False, True]).head(top_k)

    def search_reviews(self, query: str, product_ids: list[str] | None = None, top_k: int = 8) -> pd.DataFrame:
        q = self.review_vectorizer.transform([query])
        result = self.reviews.copy()
        result["score"] = cosine_similarity(q, self.review_matrix).ravel()
        if product_ids:
            result = result[result["product_id"].isin(product_ids)]
        return result.sort_values("score", ascending=False).head(top_k)


def extract_constraints(query: str) -> dict:
    text = query.lower()
    price_match = re.search(r"(?:under|below|less than|<)\s*₹?\s*([0-9]+)", text)
    color = next((c for c in ["black","white","navy","mauve","pink"] if c in text), None)
    category = None
    if "bra" in text: category = "Sports Bra"
    elif "legging" in text: category = "Leggings"
    elif "shoe" in text or "trainer" in text or "slide" in text: category = "Shoes"
    elif "tee" in text or "t-shirt" in text: category = "T-Shirt"
    return {"max_price": float(price_match.group(1)) if price_match else None, "color": color, "category": category}
