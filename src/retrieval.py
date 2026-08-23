from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


SYNONYM_EXPANSIONS = {
    "comfortable": "soft comfort relaxed easy wear",
    "comfort": "soft relaxed easy wear",
    "yoga": "studio stretching pilates low impact",
    "pilates": "studio stretching yoga low impact",
    "gym": "training workout strength fitness",
    "strength": "training gym stable secure support",
    "running": "run cardio lightweight breathable",
    "run small": "small tight snug sizing size up",
    "runs small": "small tight snug sizing size up",
    "supportive": "support secure stable compression",
    "breathable": "airflow mesh lightweight fabric",
    "lounge": "relaxed soft everyday travel recovery",
    "travel": "everyday lounge relaxed walking",
    "recovery": "cushioned soft relaxed post workout",
}


def expand_query(query: str) -> str:
    expanded = query.lower()
    for phrase, synonyms in SYNONYM_EXPANSIONS.items():
        if phrase in expanded:
            expanded += " " + synonyms
    return expanded


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 1}


def _field_overlap(query_tokens: set[str], value: str) -> float:
    value_tokens = _tokens(value)
    if not value_tokens:
        return 0.0
    return len(query_tokens & value_tokens) / len(value_tokens)


class HybridVectorIndex:
    """Zero-cost hybrid retrieval using lexical TF-IDF + dense LSA vectors.

    Truncated SVD turns sparse TF-IDF documents into dense latent semantic vectors.
    This is intentionally dependency-light for Streamlit Community Cloud while still
    providing vector retrieval beyond exact keyword matching.
    """

    def __init__(self, documents: list[str], max_components: int = 96):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.lexical_matrix = self.vectorizer.fit_transform(documents)

        feature_count = self.lexical_matrix.shape[1]
        doc_count = self.lexical_matrix.shape[0]
        max_valid = max(1, min(feature_count - 1, doc_count - 1, max_components))
        self.svd = TruncatedSVD(n_components=max_valid, random_state=42)
        self.dense_matrix = normalize(self.svd.fit_transform(self.lexical_matrix))

    def scores(
        self,
        query: str,
        lexical_weight: float = 0.45,
        semantic_weight: float = 0.55,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = self.vectorizer.transform([expand_query(query)])
        lexical = cosine_similarity(q, self.lexical_matrix).ravel()
        q_dense = normalize(self.svd.transform(q))
        semantic = cosine_similarity(q_dense, self.dense_matrix).ravel()
        hybrid = lexical_weight * lexical + semantic_weight * semantic
        return hybrid, lexical, semantic


class RetailRetriever:
    def __init__(self, products: pd.DataFrame, reviews: pd.DataFrame):
        self.products = products.copy().reset_index(drop=True)
        self.reviews = reviews.copy().reset_index(drop=True)
        self.product_docs = self.products.apply(
            lambda r: " ".join(
                str(r.get(c, ""))
                for c in ["name", "category", "color", "material", "support", "padding", "description"]
            ),
            axis=1,
        ).tolist()
        self.review_docs = self.reviews["review_text"].astype(str).tolist()
        self.product_index = HybridVectorIndex(self.product_docs)
        self.review_index = HybridVectorIndex(self.review_docs)

    @property
    def retrieval_mode(self) -> str:
        return "Constraint-aware hybrid retrieval: TF-IDF + dense LSA vectors + structured reranking"

    def _structured_product_scores(self, query: str) -> np.ndarray:
        """Lightweight field-aware reranker on top of vector similarity.

        It rewards explicit matches in high-value commerce fields (name, category,
        color, material, support, padding) without replacing semantic retrieval.
        """
        query_tokens = _tokens(expand_query(query))
        scores: list[float] = []
        for _, row in self.products.iterrows():
            name_score = _field_overlap(query_tokens, row.get("name", ""))
            attribute_score = np.mean(
                [
                    _field_overlap(query_tokens, row.get("category", "")),
                    _field_overlap(query_tokens, row.get("color", "")),
                    _field_overlap(query_tokens, row.get("material", "")),
                    _field_overlap(query_tokens, row.get("support", "")),
                    _field_overlap(query_tokens, row.get("padding", "")),
                ]
            )
            description_score = _field_overlap(query_tokens, row.get("description", ""))
            scores.append(0.30 * name_score + 0.45 * attribute_score + 0.25 * description_score)
        return np.asarray(scores, dtype=float)

    def search_products(
        self,
        query: str,
        top_k: int = 5,
        max_price: float | None = None,
        color: str | None = None,
        category: str | None = None,
    ) -> pd.DataFrame:
        # Infer structured shopping constraints here so the retriever behaves the same
        # whether called directly by evaluation or through an agent tool.
        inferred = extract_constraints(query)
        max_price = max_price if max_price is not None else inferred["max_price"]
        color = color if color is not None else inferred["color"]
        category = category if category is not None else inferred["category"]

        hybrid, lexical, semantic = self.product_index.scores(query)
        structured = self._structured_product_scores(query)

        result = self.products.copy()
        result["vector_score"] = hybrid
        result["lexical_score"] = lexical
        result["semantic_score"] = semantic
        result["structured_score"] = structured
        result["score"] = 0.78 * hybrid + 0.22 * structured

        if max_price is not None:
            result = result[result["price"] <= max_price]
        if color:
            result = result[result["color"].str.lower() == color.lower()]
        if category:
            result = result[result["category"].str.lower().str.contains(category.lower(), regex=False)]

        return result.sort_values(["score", "price"], ascending=[False, True]).head(top_k)

    def search_reviews(self, query: str, product_ids: list[str] | None = None, top_k: int = 8) -> pd.DataFrame:
        hybrid, lexical, semantic = self.review_index.scores(query)
        result = self.reviews.copy()
        result["score"] = hybrid
        result["lexical_score"] = lexical
        result["semantic_score"] = semantic
        if product_ids:
            result = result[result["product_id"].isin(product_ids)]
        return result.sort_values("score", ascending=False).head(top_k)


def extract_constraints(query: str) -> dict:
    text = query.lower()
    price_match = re.search(r"(?:under|below|less than|<)\s*₹?\s*([0-9][0-9,]*)", text)
    color = next((c for c in ["black", "white", "navy", "mauve", "pink"] if c in text), None)
    category = None
    if "bra" in text:
        category = "Sports Bra"
    elif "legging" in text or "tights" in text:
        category = "Leggings"
    elif "shoe" in text or "trainer" in text or "sneaker" in text or "slide" in text:
        category = "Shoes"
    elif "tee" in text or "t-shirt" in text or "shirt" in text:
        category = "T-Shirt"
    elif "short" in text:
        category = "Shorts"
    elif "jacket" in text or "wrap" in text:
        category = "Jacket"
    elif "pajama" in text or "sleep" in text or "lounge jogger" in text:
        category = "Sleepwear"
    elif "sock" in text or "tote" in text or "headband" in text or "sling" in text:
        category = "Accessories"

    max_price = None
    if price_match:
        max_price = float(price_match.group(1).replace(",", ""))

    return {
        "max_price": max_price,
        "color": color,
        "category": category,
    }
