import logging
from typing import Optional

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

from azure_clients.key_vault_client import kv

logger = logging.getLogger(__name__)

# Fields must match the index schema created by data_pipeline/indexer.py
VECTOR_FIELD = "embedding"
TEXT_FIELD = "content"
DEFAULT_TOP_K = 5


class AISearchClient:
    """
    Wraps Azure AI Search for hybrid retrieval over the quarterlens-filings index.
    """

    def __init__(self):
        endpoint = kv.get_secret("AZURE-SEARCH-ENDPOINT")
        admin_key = kv.get_secret("AZURE-SEARCH-ADMIN-KEY")
        index = kv.get_secret_cached("AZURE-SEARCH-INDEX") if self._has_index_secret() else "quarterlens-filings"

        self._client = SearchClient(
            endpoint=endpoint,
            index_name=index,
            credential=AzureKeyCredential(admin_key),
        )
        logger.info("AISearchClient: connected to index '%s'", index)

    @staticmethod
    def _has_index_secret() -> bool:
        """AZURE-SEARCH-INDEX may not be in KV (it's not sensitive); fall back gracefully."""
        try:
            from azure_clients.key_vault_client import kv as _kv
            _kv.get_secret("AZURE-SEARCH-INDEX")
            return True
        except ValueError:
            return False

    def search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search: BM25 keyword + vector, fused by RRF.

        Args:
            query_text:   Raw text query for BM25.
            query_vector: Embedding of the query (1536-dim, text-embedding-3-small).
            top_k:        Number of results to return.
            filters:      OData filter string, e.g. "company eq 'AAPL' and quarter eq 'Q1-2025'".

        Returns:
            List of result dicts with all index fields + @search.score.
        """
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields=VECTOR_FIELD,
        )

        results = self._client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            filter=filters,
            top=top_k,
            select="*",
        )

        hits = []
        for r in results:
            hit = dict(r)
            hit["score"] = r["@search.score"]
            hits.append(hit)

        logger.debug(
            "AISearchClient: query='%s' filters='%s' returned %d results",
            query_text, filters, len(hits),
        )
        return hits

    def vector_only_search(
        self,
        query_vector: list[float],
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[str] = None,
    ) -> list[dict]:
        """
        Pure vector search — used when no keyword query is available.
        """
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields=VECTOR_FIELD,
        )

        results = self._client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filters,
            top=top_k,
            select="*",
        )

        hits = [dict(r) | {"score": r["@search.score"]} for r in results]
        logger.debug("AISearchClient: vector-only search returned %d results", len(hits))
        return hits


# Module-level singleton
ai_search = AISearchClient()