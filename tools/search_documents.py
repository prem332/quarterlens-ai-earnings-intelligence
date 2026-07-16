from __future__ import annotations

from typing import Optional

from azure_clients.ai_search_client import ai_search
from azure_clients.openai_client import openai_client

# OData-filterable field names as they exist in the index
_FILTER_MAP = {
    "doc_type": "form",
    "company": "ticker",
    "quarter": "fiscal_label",   # index field is fiscal_label, not quarter
}


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


def search_documents(
    query: str,
    doc_type: Optional[str] = None,
    company: Optional[str] = None,
    quarter: Optional[str] = None,
    top: int = 5,
) -> dict:
    """
    Hybrid search (vector + BM25) over quarterlens-filings.

    Args:
        query:    Natural-language search query.
        doc_type: Optional filter — '10-Q', '10-K', or 'transcript'.
        company:  Optional filter — ticker symbol e.g. 'AAPL'.
        quarter:  Optional filter — fiscal label e.g. 'FY2025-Q3'.
        top:      Number of results to return (default 5).

    Returns:
        dict with 'results' list and 'count'.
    """
    # 1. Embed the query
    embedding: list[float] = openai_client.embed(query)

    # 2. Build OData filter
    odata_filter = _build_odata_filter(doc_type, company, quarter)

    # 3. Execute hybrid search via ai_search wrapper (BM25 + vector, RRF fusion)
    raw_results = ai_search.search(
        query_text=query,
        query_vector=embedding,
        top_k=top,
        filters=odata_filter,
    )

    # 4. Normalise results — wrapper returns full index field dicts
    results: list[dict] = []
    for hit in raw_results:
        results.append(
            {
                "chunk_id":    hit.get("chunk_id", ""),
                "content":     hit.get("text", hit.get("content", "")),
                "company":     hit.get("ticker", hit.get("company", "")),
                "quarter":     hit.get("fiscal_label", hit.get("quarter", "")),
                "doc_type":    hit.get("form", hit.get("doc_type", "")),
                "fiscal_label": hit.get("fiscal_label", ""),
                "accession":   hit.get("accession", ""),
                "section":     hit.get("section", ""),
                "score":       hit.get("@search.score", 0.0),
            }
        )

    return {"results": results, "count": len(results)}