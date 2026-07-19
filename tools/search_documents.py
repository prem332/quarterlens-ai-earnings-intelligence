"""
tools/search_documents.py

Raw hybrid search (BM25 + vector) over quarterlens-filings.

Phase 3 refactor: MMR and cross-encoder reranking moved to retrieval_agent.py
so that global reranking operates across merged filing + transcript candidates
instead of reranking each source independently.

This function now does:
    L2 cache check → embed query → AI Search hybrid → normalize → L2 cache set

The mmr/rerank parameters are retained for backward compatibility but are no-ops.
Any caller passing mmr=True or rerank=True should be migrated to use the
retrieval_agent orchestration layer directly.
"""

from __future__ import annotations

import math
from typing import Optional

from azure_clients.ai_search_client import ai_search
from azure_clients.openai_client import openai_client

# OData-filterable field names as they exist in the index
_FILTER_MAP = {
    "doc_type": "form",
    "company":  "ticker",
    "quarter":  "fiscal_label",
}

# MMR defaults — kept here so _mmr_rerank remains importable from retrieval_agent
_MMR_FETCH_K = 20
_MMR_LAMBDA  = 0.5


def _build_odata_filter(
    doc_type: Optional[str],
    company:  Optional[str],
    quarter:  Optional[str],
) -> Optional[str]:
    clauses: list[str] = []
    for field, value in {"doc_type": doc_type, "company": company, "quarter": quarter}.items():
        if value is not None:
            index_field = _FILTER_MAP[field]
            safe_value = value.replace("'", "''")
            clauses.append(f"{index_field} eq '{safe_value}'")
    return " and ".join(clauses) if clauses else None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_rerank(
    chunks: list[dict],
    query_embedding: list[float],
    top_k: int,
    lambda_param: float = _MMR_LAMBDA,
) -> list[dict]:
    """
    Maximal Marginal Relevance reranking.

    Public so retrieval_agent can import and call it on the merged candidate pool.
    Embedding field is not returned by AI Search (select="*" does not include
    vector fields), so chunk embeddings are computed via embed_batch on first call.

    Args:
        chunks:          Candidate chunk dicts. Each must have 'content'.
        query_embedding: 1536-dim embedding of the original query.
        top_k:           Number of chunks to return.
        lambda_param:    1.0 = pure relevance, 0.0 = pure diversity.

    Returns:
        Reranked list of up to top_k chunks.
    """
    if not chunks:
        return []

    top_k = min(top_k, len(chunks))

    contents = [c.get("content", "") for c in chunks]
    chunk_embeddings: list[list[float]] = openai_client.embed_batch(contents)

    relevance = [_cosine_similarity(emb, query_embedding) for emb in chunk_embeddings]

    selected: list[int] = []
    remaining = list(range(len(chunks)))

    for _ in range(top_k):
        if not remaining:
            break
        best_idx, best_score = None, float("-inf")
        for i in remaining:
            redundancy = (
                max(_cosine_similarity(chunk_embeddings[i], chunk_embeddings[j]) for j in selected)
                if selected else 0.0
            )
            score = lambda_param * relevance[i] - (1.0 - lambda_param) * redundancy
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [chunks[i] for i in selected]


def search_documents(
    query: str,
    doc_type: Optional[str] = None,
    company:  Optional[str] = None,
    quarter:  Optional[str] = None,
    top: int = 5,
    mmr: bool = False,             # no-op — reranking is now in retrieval_agent
    mmr_fetch_k: int = _MMR_FETCH_K,
    mmr_lambda: float = _MMR_LAMBDA,
    rerank: bool = False,          # no-op — reranking is now in retrieval_agent
    rerank_top_k: int = 5,
    use_cache: bool = True,
) -> dict:
    """
    Hybrid search (BM25 + vector RRF) over quarterlens-filings.

    Returns raw candidates — no MMR, no cross-encoder.
    Reranking is orchestrated globally in retrieval_agent after merging
    filing and transcript candidates.

    Args:
        query:      Natural-language search query.
        doc_type:   OData filter on 'form' field ('10-Q', '10-K', 'transcript').
        company:    OData filter on 'ticker' field.
        quarter:    OData filter on 'fiscal_label' field.
        top:        Number of raw candidates to return from AI Search.
        mmr:        Deprecated no-op. Pass mmr=False.
        rerank:     Deprecated no-op. Pass rerank=False.
        use_cache:  Enable L2 retrieval cache (keyed on query+company+quarter).

    Returns:
        {'results': list[dict], 'count': int}
    """
    from azure_clients.redis_client import get_retrieval_cached, set_retrieval_cached

    # L2 cache — raw candidates cached so re-runs within TTL skip AI Search
    if use_cache and company and quarter:
        cached = get_retrieval_cached(query, company, quarter)
        if cached is not None:
            return {"results": cached, "count": len(cached)}

    # Embed query (L1 embedding cache inside openai_client.embed)
    embedding: list[float] = openai_client.embed(query)

    odata_filter = _build_odata_filter(doc_type, company, quarter)

    raw_results = ai_search.search(
        query_text=query,
        query_vector=embedding,
        top_k=top,
        filters=odata_filter,
    )

    results: list[dict] = []
    for hit in raw_results:
        results.append({
            "chunk_id":     hit.get("chunk_id", ""),
            "content":      hit.get("text", hit.get("content", "")),
            "company":      hit.get("ticker", hit.get("company", "")),
            "quarter":      hit.get("fiscal_label", hit.get("quarter", "")),
            "doc_type":     hit.get("form", hit.get("doc_type", "")),
            "fiscal_label": hit.get("fiscal_label", ""),
            "accession":    hit.get("accession", ""),
            "section":      hit.get("section", ""),
            "score":        hit.get("@search.score", 0.0),
        })

    if use_cache and company and quarter and results:
        set_retrieval_cached(query, company, quarter, results)

    return {"results": results, "count": len(results)}