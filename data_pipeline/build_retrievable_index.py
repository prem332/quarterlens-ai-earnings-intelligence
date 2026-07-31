"""
One-off experimental script: build a second AI Search index
(quarterlens-filings-v2) identical to the live quarterlens-filings index,
except the `embedding` field is retrievable=True.

Does NOT touch the live index, the AZURE-SEARCH-INDEX Key Vault secret, or
anything the deployed app reads. Uses the already-computed local
data/embeddings/*.json vectors -- zero new Azure OpenAI embedding calls.

Purpose: let mmr_rerank() (tools/search_documents.py) read chunk vectors
directly from the search response instead of re-embedding every candidate
chunk live on every retrieval. This script only builds the test index;
retrieval-side code changes and evaluation happen separately.

2026-07-31 attempt notes (see project memory for full context):
  - First two runs both got hit by a duplicate-process bug (two identical
    python processes launched at the same instant, both hammering the same
    index) plus what looked like broader F0 quota exhaustion from the day's
    testing volume -- batch 0 alone took ~17 minutes across all 5 retries.
  - Reduced UPLOAD_BATCH 100->20 and added a fixed pacing delay between
    every batch (not just on 429) to stay further under whatever burst
    threshold was being hit. Untested with this config yet -- try again
    when quota has had time to reset (next morning is fine), and make
    ABSOLUTELY sure only one process is launched (run_in_background: true
    from the very first call; never let this hit a foreground timeout and
    get "promoted" -- that's what spawned the duplicate both times).
"""
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from data_pipeline.manifest_io import exists_or_warn, read_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_retrievable_index")

NEW_INDEX_NAME = "quarterlens-filings-v2"
EMBED_DIM = 1536
UPLOAD_BATCH = 20          # reduced from 100 -- smaller payload per request
BATCH_PACING_SECONDS = 5   # fixed delay between every batch, success or not
HNSW_ALGO = "hnsw-cosine"
VECTOR_PROFILE = "vector-profile"


def build_index_schema() -> SearchIndex:
    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=True,  # <-- the one schema change vs. the live index
            vector_search_dimensions=EMBED_DIM,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
        SimpleField(name="ticker", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="fiscal_label", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="form", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="section", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="subsection", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="report_date", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="cik", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="accession", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="chunk_total", type=SearchFieldDataType.Int32),
        SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="parent_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="parent_total", type=SearchFieldDataType.Int32),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=HNSW_ALGO,
                parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            )
        ],
        profiles=[
            VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name=HNSW_ALGO)
        ],
    )

    return SearchIndex(name=NEW_INDEX_NAME, fields=fields, vector_search=vector_search)


def load_all_docs(embedding_manifest: list[dict]) -> list[dict]:
    docs: list[dict] = []
    for entry in embedding_manifest:
        emb_path = Path(entry["embeddings_path"])
        if not exists_or_warn(emb_path, "embeddings file", log):
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
                "subsection":   chunk.get("subsection", ""),
                "report_date":  chunk["report_date"],
                "cik":          chunk["cik"],
                "accession":    chunk["accession"],
                "chunk_index":  chunk["chunk_index"],
                "chunk_total":  chunk["chunk_total"],
                "parent_id":    chunk.get("parent_id", chunk["chunk_id"]),
                "parent_index": chunk.get("parent_index", 0),
                "parent_total": chunk.get("parent_total", 1),
            })
    return docs


def upload_docs(endpoint: str, key: str, docs: list[dict]) -> int:
    client = SearchClient(endpoint=endpoint, index_name=NEW_INDEX_NAME, credential=AzureKeyCredential(key))
    uploaded = 0
    for start in range(0, len(docs), UPLOAD_BATCH):
        batch = docs[start:start + UPLOAD_BATCH]
        for attempt in range(5):
            try:
                results = client.upload_documents(documents=batch)
                succeeded = sum(1 for r in results if r.succeeded)
                uploaded += succeeded
                if succeeded != len(batch):
                    for r in results:
                        if not r.succeeded:
                            log.error("  upload failed: key=%s status=%s msg=%s", r.key, r.status_code, r.error_message)
                log.info("  uploaded %d/%d (batch %d-%d)", succeeded, len(batch), start, start + len(batch))
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait = 30 * (2 ** attempt)
                    log.warning("  429/quota on batch %d, waiting %ds (attempt %d/5)...", start, wait, attempt + 1)
                    time.sleep(wait)
                else:
                    raise
        time.sleep(BATCH_PACING_SECONDS)
    return uploaded


def main():
    endpoint = kv.get_secret("AZURE-SEARCH-ENDPOINT")
    key = kv.get_secret("AZURE-SEARCH-ADMIN-KEY")

    index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    existing = list(index_client.list_index_names())
    if NEW_INDEX_NAME in existing:
        log.info("Index '%s' already exists -- deleting to rebuild clean", NEW_INDEX_NAME)
        index_client.delete_index(NEW_INDEX_NAME)
        time.sleep(10)

    log.info("Creating index '%s' (embedding retrievable=True)", NEW_INDEX_NAME)
    index_client.create_index(build_index_schema())

    manifest = read_manifest("data/embeddings/embedding_manifest.json", "Embedding manifest")
    docs = load_all_docs(manifest)
    log.info("Loaded %d documents for upload", len(docs))

    uploaded = upload_docs(endpoint, key, docs)
    log.info("Done. %d/%d documents indexed into '%s'.", uploaded, len(docs), NEW_INDEX_NAME)


if __name__ == "__main__":
    main()
