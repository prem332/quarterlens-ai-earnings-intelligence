from __future__ import annotations

from typing import Optional

from azure.search.documents.models import VectorizedQuery

from azure_clients.ai_search_client import ai_search
from azure_clients.openai_client import openai_client

# Index name — matches what indexer.py created
_INDEX = "quarterlens-filings"

# Fields returned in every search hit (subset of index schema)
_SELECT_FIELDS = ["content", "company", "quarter", "doc_type", "fiscal_label", "chunk_id"]

# OData-filterable field names as they exist in the index
_FILTER_MAP = {
    "doc_type": "doc_type",
    "company": "company",
    "quarter": "quarter",
}


def _build_odata_filter(doc_type: Optional[str], company: Optional[str], quarter: Optional[str]) -> Optional[str]:
    """Build an OData $filter string from the non-None arguments."""
    clauses: list[str] = []
    params = {"doc_type": doc_type, "company": company, "quarter": quarter}
    for field, value in params.items():
        if value is not None:
            index_field = _FILTER_MAP[field]
            # OData string equality: field eq 'value'
            safe_value = value.replace("'", "''")  # escape single quotes
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
        quarter:  Optional filter — fiscal label e.g. 'Q2_FY2025'.
        top:      Number of results to return (default 5).

    Returns:
        dict with 'results' list and 'count'.
    """
    # 1. Embed the query
    embedding: list[float] = openai_client.embed(query)

    # 2. Build vector query (HNSW cosine, targets the 'embedding' field)
    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=top,
        fields="embedding",
    )

    # 3. Optional OData filter
    odata_filter = _build_odata_filter(doc_type, company, quarter)

    # 4. Execute hybrid search (BM25 text + vector)
    client = ai_search.get_client(_INDEX)
    raw_results = client.search(
        search_text=query,          # BM25 leg
        vector_queries=[vector_query],  # vector leg
        filter=odata_filter,
        select=_SELECT_FIELDS,
        top=top,
    )

    # 5. Normalise results
    results: list[dict] = []
    for hit in raw_results:
        known_fields = {"content", "company", "quarter", "doc_type", "fiscal_label", "chunk_id"}
        metadata = {k: v for k, v in hit.items() if k not in known_fields and not k.startswith("@")}
        results.append(
            {
                "content": hit.get("content", ""),
                "company": hit.get("company", ""),
                "quarter": hit.get("quarter") or hit.get("fiscal_label", ""),
                "doc_type": hit.get("doc_type", ""),
                "score": hit.get("@search.score", 0.0),
                "metadata": metadata,
            }
        )

    return {"results": results, "count": len(results)}