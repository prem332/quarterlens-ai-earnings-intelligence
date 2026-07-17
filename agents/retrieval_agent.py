"""
agents/retrieval_agent.py

Hybrid retrieval (BM25 + vector) from Azure AI Search.
Fetches filing chunks and transcript chunks separately, then merges.
Degrades gracefully — search failures return [] so the pipeline continues.
"""

from __future__ import annotations

import time
from typing import Optional

from graph.state import GraphState, DecisionLogEntry, RetrievalResult
from tools.search_documents import search_documents

# Retrieval config
_TOP_K = 10
_FILING_WEIGHT = 0.6   # 60% filing, 40% transcript


def retrieval_agent(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    t0 = time.time()
    company = state["company"]
    quarter = state["quarter"]
    query = state["query"]

    results: list[RetrievalResult] = []

    # Retrieve from 10-Q/10-K filing
    filing_k = max(1, round(_TOP_K * _FILING_WEIGHT))
    filing_hits = _safe_search(
        query=query,
        company=company,
        quarter=quarter,
        top_k=filing_k,
        doc_type_filter="filing",
    )
    results.extend(filing_hits)

    # Retrieve from earnings call transcript
    transcript_k = _TOP_K - filing_k
    transcript_hits = _safe_search(
        query=query,
        company=company,
        quarter=quarter,
        top_k=transcript_k,
        doc_type_filter="transcript",
    )
    results.extend(transcript_hits)

    entry: DecisionLogEntry = {
        "agent": "retrieval_agent",
        "tool_called": "search_documents",
        "input_summary": f"company={company} quarter={quarter} top_k={_TOP_K}",
        "output_summary": (
            f"{len(results)} chunks retrieved "
            f"({len(filing_hits)} filing, {len(transcript_hits)} transcript)"
        ),
        "confidence": None,
        "tokens_used": None,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }

    return {
        "retrieval_results": results,
        "decision_log_entries": [entry],
    }


def _safe_search(
    query: str,
    company: str,
    quarter: str,
    top_k: int,
    doc_type_filter: str,
) -> list[RetrievalResult]:
    """
    Wraps search_documents(); maps raw hits to RetrievalResult.
    doc_type_filter is kept for logging but not passed as a filter —
    the index uses 'form' field (10-Q/10-K) not 'filing'/'transcript'.
    Returns [] on any exception so the pipeline degrades gracefully.
    """
    try:
        hits = search_documents(
            query=query,
            doc_type=None,   # index uses 'form' field with 10-Q/10-K values, not filing/transcript
            company=company,
            quarter=quarter,
            top=top_k,
        )
        return [
            RetrievalResult(
                chunk_id=h.get("chunk_id", ""),
                content=h.get("content", ""),
                company=h.get("company", company),
                quarter=h.get("quarter", quarter),
                doc_type=h.get("doc_type", doc_type_filter),
                fiscal_label=h.get("fiscal_label", quarter),
                score=float(h.get("score", 0.0)),
                accession=h.get("accession", ""),
                section=h.get("section", ""),
            )
            for h in (hits.get("results", []) if isinstance(hits, dict) else hits or [])
        ]
    except Exception as exc:
        print(f"[retrieval_agent] search_documents failed ({doc_type_filter}): {exc}")
        return []