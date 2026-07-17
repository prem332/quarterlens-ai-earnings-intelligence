from __future__ import annotations

import math
from typing import Optional

from azure_clients.ai_search_client import ai_search
from azure_clients.openai_client import openai_client

# OData-filterable field names as they exist in the index
_FILTER_MAP = {
    "doc_type": "form",
    "company": "ticker",
    "quarter": "fiscal_label",   # index field is fiscal_label, not quarter
}

# MMR defaults
_MMR_FETCH_K = 20
_MMR_LAMBDA  = 0.5


def _build_odata_filter(doc_type: Optional[str], company: Optional[str], quarter: Optional[str]) -> Optional[str]:
    """Build an OData $filter string from the non-None arguments."""
    clauses: list[str] = []
    params = {"doc_type": doc_type, "company": company, "quarter": quarter}
    for field, value in params.items():
        if value is not None:
            index_field = _FILTER_MAP[field]
            safe_value = value.replace("'", "''")
            clauses.append(f"{index_field} eq '{safe_value}'")
    return " and ".join(clauses) if clauses else None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors — pure Python, no numpy dependency."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_rerank(
    chunks: list[dict],
    query_embedding: list[float],
    top_k: int,
    lambda_param: float = _MMR_LAMBDA,
) -> list[dict]:
    """
    Maximal Marginal Relevance reranking.

    Iteratively selects chunks that balance relevance to the query against
    redundancy with already-selected chunks. Reduces the echo-chamber effect
    where top-k by score returns near-duplicate windows from the same passage.

    Args:
        chunks:          Candidate chunks from AI Search (already filtered/scored).
        query_embedding: 1536-dim embedding of the original query.
        top_k:           Number of chunks to return.
        lambda_param:    Trade-off weight. 1.0 = pure relevance (no MMR),
                         0.0 = pure diversity. Default 0.5 = balanced.

    Returns:
        Reranked list of up to top_k chunks.
    """
    if not chunks:
        return []

    top_k = min(top_k, len(chunks))

    # Embed all candidate chunk contents in one batched API call
    contents = [c.get("content", "") for c in chunks]
    chunk_embeddings: list[list[float]] = openai_client.embed_batch(contents)

    # Relevance scores: cosine similarity between each chunk and the query
    relevance: list[float] = [
        _cosine_similarity(emb, query_embedding) for emb in chunk_embeddings
    ]

    selected_indices: list[int] = []
    remaining_indices: list[int] = list(range(len(chunks)))

    for _ in range(top_k):
        if not remaining_indices:
            break

        best_idx: int | None = None
        best_score = float("-inf")

        for i in remaining_indices:
            # Relevance term
            rel = relevance[i]

            # Redundancy term: max cosine similarity to any already-selected chunk
            if selected_indices:
                redundancy = max(
                    _cosine_similarity(chunk_embeddings[i], chunk_embeddings[j])
                    for j in selected_indices
                )
            else:
                redundancy = 0.0

            mmr_score = lambda_param * rel - (1.0 - lambda_param) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [chunks[i] for i in selected_indices]


def search_documents(
    query: str,
    doc_type: Optional[str] = None,
    company: Optional[str] = None,
    quarter: Optional[str] = None,
    top: int = 5,
    mmr: bool = False,
    mmr_fetch_k: int = _MMR_FETCH_K,
    mmr_lambda: float = _MMR_LAMBDA,
    rerank: bool = False,
    rerank_top_k: int = 5,
    use_cache: bool = True,
) -> dict:
    """
    Hybrid search (vector + BM25) over quarterlens-filings.

    Retrieval chain when all options enabled:
        L2 cache check → AI Search (mmr_fetch_k=20) → MMR (top=10) → Cross-encoder (rerank_top_k=5)

    Args:
        query:          Natural-language search query.
        doc_type:       Optional filter — '10-Q', '10-K', or 'transcript'.
        company:        Optional filter — ticker symbol e.g. 'AAPL'.
        quarter:        Optional filter — fiscal label e.g. 'FY2025-Q3'.
        top:            Number of results after MMR (default 5).
        mmr:            Enable MMR diversity reranking (default False).
        mmr_fetch_k:    Candidate pool fetched from AI Search before MMR (default 20).
        mmr_lambda:     MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity (default 0.5).
        rerank:         Enable cross-encoder reranking after MMR (default False).
        rerank_top_k:   Final number of chunks after cross-encoder reranking (default 5).
        use_cache:      Enable L2 retrieval cache (default True).

    Returns:
        dict with 'results' list and 'count'.
    """
    from azure_clients.redis_client import get_retrieval_cached, set_retrieval_cached

    # ── L2 cache check — skip AI Search + MMR + reranker if cached ────────
    if use_cache and company and quarter:
        cached_chunks = get_retrieval_cached(query, company or "", quarter or "")
        if cached_chunks is not None:
            return {"results": cached_chunks, "count": len(cached_chunks)}

    # 1. Embed the query (L1 cache inside embed())
    embedding: list[float] = openai_client.embed(query)

    # 2. Build OData filter
    odata_filter = _build_odata_filter(doc_type, company, quarter)

    # 3. Fetch candidate pool — larger when MMR is enabled
    fetch_k = max(mmr_fetch_k, top) if mmr else top
    raw_results = ai_search.search(
        query_text=query,
        query_vector=embedding,
        top_k=fetch_k,
        filters=odata_filter,
    )

    # 4. Normalise results
    results: list[dict] = []
    for hit in raw_results:
        results.append(
            {
                "chunk_id":     hit.get("chunk_id", ""),
                "content":      hit.get("text", hit.get("content", "")),
                "company":      hit.get("ticker", hit.get("company", "")),
                "quarter":      hit.get("fiscal_label", hit.get("quarter", "")),
                "doc_type":     hit.get("form", hit.get("doc_type", "")),
                "fiscal_label": hit.get("fiscal_label", ""),
                "accession":    hit.get("accession", ""),
                "section":      hit.get("section", ""),
                "score":        hit.get("@search.score", 0.0),
            }
        )

    # 5. MMR reranking — diversity-aware chunk selection
    if mmr and len(results) > top:
        results = _mmr_rerank(
            chunks=results,
            query_embedding=embedding,
            top_k=top,
            lambda_param=mmr_lambda,
        )
    else:
        results = results[:top]

    # 6. Cross-encoder reranking — accuracy reranking on MMR output
    if rerank and results:
        from tools.rerank_documents import rerank_documents
        results = rerank_documents(
            query=query,
            chunks=results,
            top_k=rerank_top_k,
        )

    # ── L2 cache set — store results for future identical queries ─────────
    if use_cache and company and quarter and results:
        set_retrieval_cached(query, company, quarter, results)

    return {"results": results, "count": len(results)}