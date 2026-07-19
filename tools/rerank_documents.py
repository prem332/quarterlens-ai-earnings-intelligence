"""
tools/rerank_documents.py

Cross-encoder reranking for QuarterLens AI retrieval pipeline.
Slots after MMR in the retrieval chain:
    AI Search (20) → MMR (10) → Cross-encoder (5) → Agents

Design decisions (mirrors run_finbert.py):
  - Lazy singleton: model loads once per process on first call (~80MB, CPU-only)
  - Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — purpose-built for passage reranking,
    trained on MS MARCO (query/passage relevance), strong on financial text
  - Batched scoring: all query+chunk pairs scored in one model call
  - Adds 'rerank_score' field to each chunk dict — preserves original 'score' (BM25+vector)
    so both scores are available for debugging/ablation
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Lazy singleton — model loads once, on first call (~80MB, CPU-only)
# ---------------------------------------------------------------------------
_cross_encoder: Optional[object] = None
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder  # type: ignore
        _cross_encoder = CrossEncoder(
            _MODEL_NAME,
            device="cpu",
        )
    return _cross_encoder


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def rerank_documents(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Cross-encoder reranking over retrieved chunks.

    Scores each (query, chunk_content) pair jointly — more accurate than
    embedding cosine similarity because the model sees both texts together.
    Runs after MMR so it operates on an already-diversified candidate set.

    Args:
        query:  Original query string.
        chunks: Candidate chunk dicts from MMR output. Each must have 'content'.
        top_k:  Number of chunks to return after reranking.

    Returns:
        Top-k chunks sorted by rerank_score descending.
        Each chunk dict gains a 'rerank_score' field (float).
        Original 'score' (BM25+vector RRF score) is preserved for ablation.
    """
    if not chunks:
        return []

    top_k = min(top_k, len(chunks))
    model = _get_cross_encoder()

    # Build (query, passage) pairs for batch scoring
    pairs = [(query, c.get("content", "")) for c in chunks]

    # Single batched inference call — returns list of float scores
    scores: list[float] = model.predict(pairs).tolist()

    # Attach rerank_score to each chunk
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        enriched = dict(chunk)
        enriched["rerank_score"] = round(float(score), 6)
        scored_chunks.append(enriched)

    # Sort by rerank_score descending, return top_k
    scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored_chunks[:top_k]

# Warm-up: load model at import time so first rerank_documents() call has no cold-start penalty
_get_cross_encoder()