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

import numpy as np

from azure_clients.ai_search_client import ai_search
from azure_clients.openai_client import openai_client

# OData-filterable field names as they exist in the index
_FILTER_MAP = {
    "doc_type": "form",
    "company":  "ticker",
    "quarter":  "fiscal_label",
}

# Default relevance/diversity balance for mmr_rerank. retrieval_agent always
# passes lambda_param explicitly (env-ablatable there); this is the fallback.
_MMR_LAMBDA = 0.5


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

    Chunk vectors are resolved in this order:
      1. The 'embedding' field on the chunk, if the index returned it. The
         live index has that field non-retrievable, so this is normally
         absent — but a rebuilt index with retrievable=True makes it present,
         and then MMR costs nothing extra.
      2. openai_client.embed_batch(), which is itself L1-cached, so repeat
         retrievals of the same chunks don't re-embed.

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
    chunk_embeddings: list[list[float]] = [None] * len(chunks)  # type: ignore[list-item]
    missing_idx: list[int] = []
    for i, c in enumerate(chunks):
        emb = c.get("embedding")
        if isinstance(emb, list) and emb:
            chunk_embeddings[i] = emb
        else:
            missing_idx.append(i)

    if missing_idx:
        fetched = openai_client.embed_batch([contents[i] for i in missing_idx])
        for i, emb in zip(missing_idx, fetched):
            chunk_embeddings[i] = emb

    # Vectorised. The previous implementation called _cosine_similarity in a
    # nested Python loop: measured 819 calls for a 24-chunk pool at top_k=10,
    # each one iterating 1536 dimensions AND recomputing both vectors' norms
    # from scratch every time -- roughly 3.8M interpreter-level operations,
    # 322ms of pure CPU inside retrieval.
    #
    # Normalising once turns every cosine into a dot product, so relevance is
    # a single matrix-vector product and all pairwise similarities are one
    # matmul, both in BLAS rather than the interpreter. Selection logic and
    # tie-breaking order are unchanged -- verified to produce an identical
    # ordering against the previous implementation.
    matrix = np.asarray(chunk_embeddings, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    query_vec = np.asarray(query_embedding, dtype=np.float32)
    query_vec /= np.linalg.norm(query_vec) + 1e-12

    relevance = matrix @ query_vec        # (n,)
    pairwise = matrix @ matrix.T          # (n, n)

    selected: list[int] = []
    remaining = list(range(len(chunks)))

    for _ in range(top_k):
        if not remaining:
            break
        redundancy = (
            pairwise[np.ix_(remaining, selected)].max(axis=1)
            if selected
            else np.zeros(len(remaining), dtype=np.float32)
        )
        scores = lambda_param * relevance[remaining] - (1.0 - lambda_param) * redundancy
        # argmax keeps the first maximum, matching the strict ">" comparison
        # the original loop used, so ties resolve to the same candidate.
        best_idx = remaining[int(np.argmax(scores))]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [chunks[i] for i in selected]


def search_documents(
    query: str,
    doc_type: Optional[str] = None,
    company:  Optional[str] = None,
    quarter:  Optional[str] = None,
    top: int = 5,
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
        use_cache:  Enable L2 retrieval cache (keyed on
                    query+company+quarter+doc_type).

    Returns:
        {'results': list[dict], 'count': int}
    """
    from azure_clients.redis_client import get_retrieval_cached, set_retrieval_cached

    # L2 cache — raw candidates cached so re-runs within TTL skip AI Search
    if use_cache and company and quarter:
        cached = get_retrieval_cached(query, company, quarter, doc_type=doc_type or "all")
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
        row = {
            "chunk_id":     hit.get("chunk_id", ""),
            "content":      hit.get("text", hit.get("content", "")),
            "company":      hit.get("ticker", hit.get("company", "")),
            "quarter":      hit.get("fiscal_label", hit.get("quarter", "")),
            "doc_type":     hit.get("form", hit.get("doc_type", "")),
            "fiscal_label": hit.get("fiscal_label", ""),
            "accession":    hit.get("accession", ""),
            "section":      hit.get("section", ""),
            "chunk_index":  hit.get("chunk_index", -1),
            "chunk_total":  hit.get("chunk_total", -1),
            "parent_id":    hit.get("parent_id", ""),
            "parent_index": hit.get("parent_index", 0),
            "parent_total": hit.get("parent_total", 1),
            "score":        hit.get("@search.score", 0.0),
        }
        # Only present if the index marks 'embedding' retrievable. Absent on
        # the current live index; carried through when available so mmr_rerank
        # can skip re-embedding entirely. Deliberately NOT written to the L2
        # cache below — 1536 floats x 24 chunks per entry would bloat Redis for
        # no gain, since embeddings have their own longer-lived L1 cache.
        emb = hit.get("embedding")
        if isinstance(emb, list) and emb:
            row["embedding"] = emb
        results.append(row)

    if use_cache and company and quarter and results:
        cacheable = [{k: v for k, v in r.items() if k != "embedding"} for r in results]
        set_retrieval_cached(query, company, quarter, cacheable, doc_type=doc_type or "all")

    return {"results": results, "count": len(results)}


def fetch_parent_siblings(
    parent_id: str,
    company: Optional[str] = None,
    quarter: Optional[str] = None,
    top: int = 50,
) -> list[dict]:
    """
    All child chunks of one L2 parent, ordered by parent_index — for small-to-big
    parent reconstruction. Filter is defensively scoped by ticker+fiscal_label;
    parent_id is a UUID so company/quarter are belt-and-suspenders against mixing.
    Concatenating the returned 'content' values reconstructs the parent block.
    """
    if not parent_id:
        return []

    def _esc(v: str) -> str:
        return v.replace("'", "''")

    clauses = [f"parent_id eq '{_esc(parent_id)}'"]
    if company:
        clauses.append(f"ticker eq '{_esc(company)}'")
    if quarter:
        clauses.append(f"fiscal_label eq '{_esc(quarter)}'")

    try:
        hits = ai_search.filter_search(" and ".join(clauses), top=top)
    except Exception as exc:  # noqa: BLE001 — expansion is best-effort
        print(f"[search_documents] fetch_parent_siblings failed: {exc}")
        return []

    rows = [
        {"parent_index": h.get("parent_index", 0),
         "content": h.get("text", h.get("content", ""))}
        for h in hits
    ]
    rows.sort(key=lambda r: r["parent_index"])
    return rows