from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
    VectorSearchAlgorithmMetric,
)
from azure_clients.key_vault_client import kv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("indexer")

INDEX_NAME = kv.get_secret("AZURE-SEARCH-INDEX")
EMBED_DIM = 1536
UPLOAD_BATCH = 100   # smaller batches for Free tier quota headroom
HNSW_ALGO = "hnsw-cosine"
VECTOR_PROFILE = "vector-profile"


def make_index_client() -> SearchIndexClient:
    endpoint = kv.get_secret("AZURE-SEARCH-ENDPOINT")
    key = kv.get_secret("AZURE-SEARCH-ADMIN-KEY")
    return SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def build_index() -> SearchIndex:
    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBED_DIM,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
        # Filterable provenance / facets
        SimpleField(name="ticker", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="fiscal_label", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="form", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="section", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        # Phase 4: subsection field for structure-aware filtering
        SimpleField(name="subsection", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="report_date", type=SearchFieldDataType.String,
                    filterable=True, sortable=True),
        SimpleField(name="cik", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="accession", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="chunk_total", type=SearchFieldDataType.Int32),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=HNSW_ALGO,
                parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE,
                algorithm_configuration_name=HNSW_ALGO,
            )
        ],
    )

    return SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)


def recreate_index(client: SearchIndexClient) -> None:
    import time
    from azure.search.documents import SearchClient
    from azure_clients.key_vault_client import kv as _kv

    existing = [i for i in client.list_index_names()]
    if INDEX_NAME in existing:
        log.info("Deleting existing index '%s'", INDEX_NAME)
        client.delete_index(INDEX_NAME)
        # Free F0 tier needs time to reclaim storage after deletion
        # Poll until index is gone or timeout after 60s
        for attempt in range(12):
            time.sleep(5)
            current = [i for i in client.list_index_names()]
            if INDEX_NAME not in current:
                log.info("Index confirmed deleted after %ds", (attempt + 1) * 5)
                break
            log.info("Waiting for index deletion... (%ds)", (attempt + 1) * 5)
        time.sleep(5)  # extra buffer for storage reclaim

    log.info("Creating index '%s'", INDEX_NAME)
    client.create_index(build_index())


def _load_all_docs(embedding_manifest: list[dict]) -> list[dict]:
    """Load every embedded chunk across all filings into upload-ready docs."""
    docs: list[dict] = []
    for entry in embedding_manifest:
        emb_path = Path(entry["embeddings_path"])
        if not emb_path.exists():
            log.warning("Missing embeddings file, skipping: %s", emb_path)
            continue
        for chunk in json.loads(emb_path.read_text(encoding="utf-8")):
            docs.append({
                "chunk_id":     chunk["chunk_id"],
                "text":         chunk["text"],
                "embedding":    chunk["embedding"],
                "ticker":       chunk["ticker"],
                "fiscal_label": chunk["fiscal_label"],
                "form":         chunk["form"],
                "section":      chunk["section"],
                "subsection":   chunk.get("subsection", ""),  # Phase 4: new field
                "report_date":  chunk["report_date"],
                "cik":          chunk["cik"],
                "accession":    chunk["accession"],
                "chunk_index":  chunk["chunk_index"],
                "chunk_total":  chunk["chunk_total"],
            })
    return docs


def upload_docs(endpoint: str, key: str, docs: list[dict]) -> int:
    import time
    client = SearchClient(
        endpoint=endpoint, index_name=INDEX_NAME, credential=AzureKeyCredential(key)
    )
    uploaded = 0
    for start in range(0, len(docs), UPLOAD_BATCH):
        batch = docs[start:start + UPLOAD_BATCH]
        # Retry on 429 (quota) with exponential backoff
        for attempt in range(5):
            try:
                results = client.upload_documents(documents=batch)
                succeeded = sum(1 for r in results if r.succeeded)
                uploaded += succeeded
                if succeeded != len(batch):
                    for r in results:
                        if not r.succeeded:
                            log.error("  upload failed: key=%s status=%s", r.key, r.status_code)
                log.info("  uploaded %d/%d (batch %d-%d)",
                         succeeded, len(batch), start, start + len(batch))
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait = 30 * (2 ** attempt)
                    log.warning("  429/quota on batch %d, waiting %ds (attempt %d/5)...",
                                start, wait, attempt + 1)
                    time.sleep(wait)
                else:
                    raise
    return uploaded


def run(embedding_manifest_path: str) -> None:
    manifest_p = Path(embedding_manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Embedding manifest not found: {embedding_manifest_path}")

    endpoint = kv.get_secret("AZURE-SEARCH-ENDPOINT")
    key = kv.get_secret("AZURE-SEARCH-ADMIN-KEY")

    embedding_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    index_client = make_index_client()
    recreate_index(index_client)

    docs = _load_all_docs(embedding_manifest)
    log.info("Loaded %d documents for upload", len(docs))

    uploaded = upload_docs(endpoint, key, docs)
    log.info("Done. %d/%d documents indexed into '%s'.", uploaded, len(docs), INDEX_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index embedded filings into Azure AI Search.")
    parser.add_argument("--manifest", default="data/embeddings/embedding_manifest.json")
    args = parser.parse_args()
    run(args.manifest)


if __name__ == "__main__":
    main()