# Retrieval Architecture

Ask Victoria uses a **zero-cost hybrid vector retrieval layer** designed to remain reliable on Streamlit Community Cloud without requiring a paid embedding API or a large hosted vector database.

## Retrieval pipeline

```text
User query
   |
   v
Constraint extraction
(price / color / category)
   |
   v
Query expansion
(comfort -> soft / relaxed, yoga -> studio / stretching, etc.)
   |
   +-----------------------------+
   |                             |
   v                             v
TF-IDF lexical similarity    Dense LSA vector similarity
   |                             |
   +-------------+---------------+
                 |
                 v
        Weighted hybrid score
     45% lexical + 55% semantic
                 |
                 v
          Top-K evidence
```

## Why dense LSA embeddings?

The project deliberately uses **TF-IDF + Truncated SVD (Latent Semantic Analysis)** to create dense vectors. Truncated SVD projects sparse term-document vectors into a lower-dimensional latent space, allowing retrieval to capture relationships that are not limited to exact token overlap.

This is not presented as a transformer embedding model. The design choice is intentional:

- zero API cost;
- no GPU requirement;
- no model-download dependency for the live Streamlit demo;
- deterministic and reproducible evaluation;
- substantially lighter than running a transformer encoder on Community Cloud;
- easy to replace with SentenceTransformers, Gemini embeddings, OpenAI embeddings or a hosted vector database in a production architecture.

## Hybrid scoring

For each query, Ask Victoria computes:

```text
hybrid_score = 0.45 * lexical_similarity + 0.55 * dense_semantic_similarity
```

The weights are visible in code and can be tuned using an offline evaluation set.

## Two vector indexes

Two independent indexes are created:

1. **Product index** — product name, category, color, material, support, padding and product description.
2. **Review index** — individual customer-review text.

The Product Agent and Recommendation Agent primarily use the product index, while the Review Intelligence Agent retrieves evidence from the review index and combines it with aspect-level aggregate summaries.

## Metadata filtering

Semantic relevance is not sufficient for structured commerce constraints. After scoring, the retriever can apply deterministic metadata filters for:

- maximum price;
- color;
- category.

This is why a query such as `black sports bra under ₹2000 for yoga` is handled as a combination of structured filtering and semantic ranking rather than vector similarity alone.

## Evaluation

The repository includes a retrieval test set and reports:

- Recall@3;
- Top-1 retrieval accuracy;
- retrieved product IDs for every evaluation query.

These metrics appear in the Streamlit **Evaluation Lab**. They are calculated from the running retrieval implementation rather than hard-coded into the UI.

## Production upgrade path

A production implementation could replace the dense LSA layer while preserving the same retrieval interface:

```text
Current portfolio implementation
TF-IDF + LSA dense vectors
          |
          v
Production alternatives
SentenceTransformers / Gemini embeddings / other embedding model
          |
          v
FAISS / MongoDB Atlas Vector Search / Qdrant / Pinecone / pgvector
          |
          v
Hybrid retrieval + metadata filters + reranker
```

For this portfolio project, the current architecture provides genuine dense-vector retrieval while keeping the public demo free and operationally simple.
