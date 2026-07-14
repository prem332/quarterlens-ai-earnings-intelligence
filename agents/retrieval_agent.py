"""
Calls search_documents() (hybrid vector+BM25 over AI Search) to fetch relevant
chunks for the company/quarter/query. The LLM is NOT called here — tool output
is the retrieval result. No LLM synthesis needed; downstream agents consume the
raw chunks directly.

Tool: search_documents(query, doc_type, company, quarter) → list[dict]
"""

import time
from graph.state import GraphState, DecisionLogEntry, RetrievalResult
from tools.search_documents import search_documents


_TOP_K = 10           # chunks retrieved per agent invocation
_FILING_WEIGHT = 0.6  # split: 60% filing chunks, 40% transcript chunks


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
    filing_hits = _safe_search(query, doc_type=None, company=company,
                               quarter=quarter, top_k=filing_k,
                               doc_type_filter="filing")
    results.extend(filing_hits)

    # Retrieve from earnings call transcript
    transcript_k = _TOP_K - filing_k
    transcript_hits = _safe_search(query, doc_type=None, company=company,
                                   quarter=quarter, top_k=transcript_k,
                                   doc_type_filter="transcript")
    results.extend(transcript_hits)

    entry: DecisionLogEntry = {
        "agent": "retrieval_agent",
        "tool_called": "search_documents",
        "input_summary": f"company={company} quarter={quarter} top_k={_TOP_K}",
        "output_summary": f"{len(results)} chunks retrieved "
                          f"({len(filing_hits)} filing, {len(transcript_hits)} transcript)",
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
    doc_type,
    company: str,
    quarter: str,
    top_k: int,
    doc_type_filter: str,
) -> list[RetrievalResult]:
    """
    Wraps search_documents(); maps raw hits to RetrievalResult.
    doc_type_filter is passed as a filter string to narrow results.
    Returns [] on any exception so the pipeline degrades gracefully.
    """
    try:
        hits = search_documents(
            query=query,
            doc_type=doc_type_filter,
            company=company,
            quarter=quarter,
            top_k=top_k,
        )
        return [
            RetrievalResult(
                chunk_id=h.get("chunk_id", ""),
                content=h.get("content", ""),
                company=h.get("company", company),
                quarter=h.get("quarter", quarter),
                doc_type=h.get("doc_type", doc_type_filter),
                fiscal_label=h.get("fiscal_label", quarter),
                score=float(h.get("@search.score", h.get("score", 0.0))),
            )
            for h in (hits or [])
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"[retrieval_agent] search_documents failed ({doc_type_filter}): {exc}")
        return []