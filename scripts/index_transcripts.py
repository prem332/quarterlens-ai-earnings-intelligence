"""
scripts/index_transcripts.py

Parses, chunks, embeds, and indexes earnings call transcripts into the
existing Azure AI Search index — ADDITIVE only, does not recreate the index.

Pipeline:
  data/raw/transcripts/{ticker}/{fiscal_label}.json
      → parse (split into sections)
      → chunk (512 tokens, 50 overlap)
      → embed (text-embedding-3-small, 1536-dim)
      → upload to AI Search (additive, preserves filing chunks)

Transcript doc_type in index: "transcript"
Section key: "full" (transcripts are not split by section)

Usage:
    python scripts/index_transcripts.py
    python scripts/index_transcripts.py --dry-run   # count files, no API calls
    python scripts/index_transcripts.py --ticker AAPL  # single company
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("index_transcripts")

# ── Paths ─────────────────────────────────────────────────────────────────────
TRANSCRIPT_DIR = Path("data/raw/transcripts")
COMPANIES = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]

# CIK map — matches manifest.json from edgar_downloader
CIK_MAP = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "NVDA":  "0001045810",
    "GOOGL": "0001652044",
    "META":  "0001326801",
}


# ── Step 1: Parse transcript JSON → section dicts ────────────────────────────

def parse_transcript(path: Path) -> list[dict] | None:
    """
    Load a transcript JSON file and produce a list of section dicts
    compatible with chunk_filing() in chunking.py.

    Transcript JSON structure (from transcript_fetcher.py):
        ticker, fiscal_label, fiscal_year, fiscal_quarter,
        provider, call_date, char_count, metadata, text, fetched_at
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
        return None

    text = (record.get("text") or "").strip()
    if not text:
        log.warning("Empty transcript: %s", path)
        return None

    ticker = record.get("ticker", path.parent.name)
    fiscal_label = record.get("fiscal_label", path.stem)
    call_date = record.get("call_date") or ""
    cik = CIK_MAP.get(ticker, "0000000000")

    # Split transcript into ~3000-char sections to keep chunks meaningful
    # Each section becomes one entry for chunk_filing()
    section_size = 3000
    sections = []
    for i in range(0, len(text), section_size):
        chunk_text = text[i:i + section_size].strip()
        if not chunk_text:
            continue
        sections.append({
            "ticker":       ticker,
            "cik":          cik,
            "fiscal_label": fiscal_label,
            "report_date":  call_date,
            "form":         "transcript",
            "accession":    f"transcript_{ticker}_{fiscal_label}",
            "section":      f"transcript_part_{i // section_size}",
            "text":         chunk_text,
        })

    log.info("  Parsed %s %s — %d chars → %d sections",
             ticker, fiscal_label, len(text), len(sections))
    return sections


# ── Step 2: Chunk ─────────────────────────────────────────────────────────────

def chunk_sections(sections: list[dict]) -> list[dict]:
    """Chunk sections using same logic as data_pipeline/chunking.py."""
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")

    chunks = []
    for section in sections:
        tokens = encoder.encode(section["text"])
        chunk_size, overlap = 512, 50
        start = 0
        section_chunks = []
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            section_chunks.append(encoder.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += chunk_size - overlap

        total = len(section_chunks)
        for i, chunk_text in enumerate(section_chunks):
            chunks.append({
                "ticker":       section["ticker"],
                "cik":          section["cik"],
                "fiscal_label": section["fiscal_label"],
                "report_date":  section["report_date"],
                "form":         section["form"],
                "accession":    section["accession"],
                "section":      section["section"],
                "chunk_id":     str(uuid.uuid4()),
                "chunk_index":  i,
                "chunk_total":  total,
                "text":         chunk_text,
            })

    return chunks


# ── Step 3: Embed ─────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed chunks using text-embedding-3-small via openai_client."""
    from azure_clients.openai_client import openai_client

    batch_size = 100
    embedded = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c["text"] for c in batch]
        vectors = openai_client.embed_batch(texts)
        for chunk, vec in zip(batch, vectors):
            embedded.append({**chunk, "embedding": vec})
        log.info("  Embedded batch %d-%d", start, start + len(batch))

    return embedded


# ── Step 4: Upload to AI Search (additive) ────────────────────────────────────

def upload_to_search(embedded_chunks: list[dict]) -> int:
    """Upload chunks to AI Search index — additive, no index recreation."""
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure_clients.key_vault_client import kv

    endpoint = kv.get_secret("AZURE-SEARCH-ENDPOINT")
    key = kv.get_secret("AZURE-SEARCH-ADMIN-KEY")
    index_name = kv.get_secret("AZURE-SEARCH-INDEX")

    client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(key),
    )

    # Build upload docs — same field mapping as indexer.py
    docs = []
    for c in embedded_chunks:
        docs.append({
            "chunk_id":     c["chunk_id"],
            "text":         c["text"],
            "embedding":    c["embedding"],
            "ticker":       c["ticker"],
            "fiscal_label": c["fiscal_label"],
            "form":         c["form"],           # "transcript"
            "section":      c["section"],
            "report_date":  c["report_date"],
            "cik":          c["cik"],
            "accession":    c["accession"],
            "chunk_index":  c["chunk_index"],
            "chunk_total":  c["chunk_total"],
        })

    uploaded = 0
    batch_size = 500
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        results = client.upload_documents(documents=batch)
        succeeded = sum(1 for r in results if r.succeeded)
        uploaded += succeeded
        failed = len(batch) - succeeded
        if failed:
            for r in results:
                if not r.succeeded:
                    log.error("  Upload failed: key=%s status=%s", r.key, r.status_code)
        log.info("  Uploaded %d/%d docs (batch %d-%d)",
                 succeeded, len(batch), start, start + len(batch))

    return uploaded


# ── Main ──────────────────────────────────────────────────────────────────────

def run(ticker_filter: str | None = None, dry_run: bool = False) -> None:
    companies = [ticker_filter] if ticker_filter else COMPANIES
    total_chunks = 0
    total_uploaded = 0
    skipped = []

    for ticker in companies:
        ticker_dir = TRANSCRIPT_DIR / ticker
        if not ticker_dir.exists():
            log.warning("No transcript directory for %s", ticker)
            skipped.append(ticker)
            continue

        transcript_files = sorted(ticker_dir.glob("*.json"))
        log.info("== %s: %d transcript files ==", ticker, len(transcript_files))

        for path in transcript_files:
            log.info("Processing %s", path.name)

            # Parse
            sections = parse_transcript(path)
            if not sections:
                skipped.append(str(path))
                continue

            if dry_run:
                log.info("  [DRY RUN] Would chunk/embed/upload %d sections", len(sections))
                continue

            # Chunk
            chunks = chunk_sections(sections)
            log.info("  Chunked → %d chunks", len(chunks))

            # Embed
            embedded = embed_chunks(chunks)
            log.info("  Embedded → %d vectors", len(embedded))

            # Upload
            uploaded = upload_to_search(embedded)
            log.info("  Uploaded → %d docs to AI Search", uploaded)

            total_chunks += len(chunks)
            total_uploaded += uploaded

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TRANSCRIPT INDEXING COMPLETE")
    print("=" * 55)
    print(f"  Companies processed : {len(companies)}")
    print(f"  Total chunks        : {total_chunks}")
    print(f"  Total uploaded      : {total_uploaded}")
    print(f"  Skipped             : {len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"    - {s}")
    print("=" * 55)

    if not dry_run:
        print("\nVerify with:")
        print("  python -c \"")
        print("  import sys; sys.path.insert(0, '.')")
        print("  from tools.search_documents import search_documents")
        print("  r = search_documents('revenue earnings call', doc_type='transcript', top=5)")
        print("  print(r['count'], 'transcript chunks found')\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index transcripts into Azure AI Search additively.")
    parser.add_argument("--ticker", help="Process single ticker only (e.g. AAPL)")
    parser.add_argument("--dry-run", action="store_true", help="Count files only, no API calls")
    args = parser.parse_args()
    run(ticker_filter=args.ticker, dry_run=args.dry_run)