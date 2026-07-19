"""
data_pipeline/chunking.py

Recursive chunking for QuarterLens AI — Phase 2 (deviation #28).

Strategy: RecursiveCharacterTextSplitter with token-based length (cl100k_base).
  Separators: [\n\n, \n, ". ", " ", ""] — respects paragraph/sentence/line boundaries.
  Sizing: 512 tokens / 50 overlap — identical to fixed-size baseline for clean ablation.
  Fixed-size baseline metrics locked in MLflow finetuned-eval-v5.

Covers both filings (from parsed_manifest.json) and transcripts
(from transcripts_manifest.json) so that embedding.py and indexer.py
work completely unchanged — they just process more entries from chunk_manifest.json.

Pipeline:
    python -m data_pipeline.chunking    # chunk filings + transcripts
    python -m data_pipeline.embedding   # embed all chunks
    python -m data_pipeline.indexer     # recreate index + upload all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("chunking")

CHUNK_SIZE    = 512    # tokens
CHUNK_OVERLAP = 50     # tokens
ENCODING      = "cl100k_base"  # matches text-embedding-3-small / GPT-4 family

# Recursive separators — priority order:
# \n\n (paragraph) → \n (line/table-row) → ". " (sentence) → " " (word) → "" (char)
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# CIK map — used when chunking transcripts (not stored in transcripts_manifest.json)
_CIK_MAP = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "NVDA":  "0001045810",
    "GOOGL": "0001652044",
    "META":  "0001326801",
}

# Max chars per transcript section before recursive chunking
_TRANSCRIPT_SECTION_SIZE = 3000


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def _make_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Token-aware RecursiveCharacterTextSplitter using tiktoken cl100k_base.
    Replaces fixed-size token windows — same sizing, single-variable ablation.
    """
    encoder = tiktoken.get_encoding(ENCODING)

    def _token_len(text: str) -> int:
        return len(encoder.encode(text))

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=_token_len,
        is_separator_regex=False,
    )


def chunk_text(
    text: str,
    encoder: tiktoken.Encoding,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text using RecursiveCharacterTextSplitter.
    Signature kept identical to fixed-size version for backward compatibility.
    encoder parameter retained for API compatibility; tiktoken used internally.
    """
    if not text.strip():
        return []
    splitter = _make_splitter(chunk_size, overlap)
    return splitter.split_text(text)


def chunk_filing(parsed_sections: list[dict], encoder: tiktoken.Encoding) -> list[dict]:
    """
    Chunk all sections from one parsed filing.
    Each input dict has: ticker, cik, fiscal_label, report_date, form,
                         accession, section, text.
    Each output chunk adds: chunk_id, chunk_index, chunk_total (per section).
    Signature and output structure unchanged from fixed-size version.
    """
    chunks: list[dict] = []

    for section in parsed_sections:
        text_chunks = chunk_text(section["text"], encoder)
        total = len(text_chunks)
        for i, chunk_text_str in enumerate(text_chunks):
            chunks.append({
                # provenance — carried from parser output
                "ticker":       section["ticker"],
                "cik":          section["cik"],
                "fiscal_label": section["fiscal_label"],
                "report_date":  section["report_date"],
                "form":         section["form"],
                "accession":    section["accession"],
                "section":      section["section"],
                # chunk identity
                "chunk_id":     str(uuid.uuid4()),
                "chunk_index":  i,
                "chunk_total":  total,
                # content
                "text":         chunk_text_str,
            })

    return chunks


def chunk_transcript(record: dict, encoder: tiktoken.Encoding) -> list[dict]:
    """
    Chunk one transcript record into chunk dicts compatible with embedding.py.

    Transcript JSON structure (from transcript_fetcher.py):
        ticker, fiscal_label, call_date, text, ...

    Splits transcript text into 3000-char sections first, then recursively
    chunks each section. This avoids very long single chunks from dense
    earnings call transcripts.

    Output chunk dict structure is identical to chunk_filing() output so
    embedding.py and indexer.py handle transcripts without any changes.
    """
    ticker = record.get("ticker", "")
    fiscal_label = record.get("fiscal_label", "")
    call_date = record.get("call_date") or ""
    text = (record.get("text") or "").strip()
    cik = _CIK_MAP.get(ticker, "0000000000")

    if not text:
        return []

    chunks: list[dict] = []
    for i in range(0, len(text), _TRANSCRIPT_SECTION_SIZE):
        section_text = text[i:i + _TRANSCRIPT_SECTION_SIZE].strip()
        if not section_text:
            continue

        split_texts = chunk_text(section_text, encoder)
        total = len(split_texts)

        for j, chunk_text_str in enumerate(split_texts):
            chunks.append({
                "ticker":       ticker,
                "cik":          cik,
                "fiscal_label": fiscal_label,
                "report_date":  call_date,
                "form":         "transcript",
                "accession":    f"transcript_{ticker}_{fiscal_label}",
                "section":      f"transcript_part_{i // _TRANSCRIPT_SECTION_SIZE}",
                "chunk_id":     str(uuid.uuid4()),
                "chunk_index":  j,
                "chunk_total":  total,
                "text":         chunk_text_str,
            })

    return chunks


def run(
    parsed_manifest_path: str,
    out_root: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    transcripts_manifest_path: str | None = None,
) -> None:
    """
    Chunk all filings from parsed_manifest + all transcripts from
    transcripts_manifest into chunk_manifest.json.

    embedding.py reads chunk_manifest.json and processes all entries
    (filings + transcripts) without any changes.
    """
    manifest_p = Path(parsed_manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Parsed manifest not found: {parsed_manifest_path}")

    parsed_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    out_root_p = Path(out_root)
    encoder = _get_encoder()
    chunk_manifest: list[dict] = []

    # ── Chunk filings ─────────────────────────────────────────────────────────
    log.info("=== Chunking %d filings (recursive) ===", len(parsed_manifest))

    for entry in parsed_manifest:
        parsed_path = Path(entry["parsed_path"])
        if not parsed_path.exists():
            log.warning("Missing parsed file, skipping: %s", parsed_path)
            continue

        sections = json.loads(parsed_path.read_text(encoding="utf-8"))
        log.info(
            "Chunking %s %s (%s) — %d sections",
            entry["ticker"], entry["fiscal_label"], entry["form"], len(sections)
        )

        chunks = chunk_filing(sections, encoder)

        out_dir = out_root_p / entry["ticker"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{entry['fiscal_label']}_{entry['form'].replace('-', '')}_chunks.json"
        out_file.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        log.info(
            "  %s %s: %d chunks -> %s",
            entry["ticker"], entry["fiscal_label"], len(chunks), out_file.name
        )

        chunk_manifest.append({
            **{k: entry[k] for k in
               ("ticker", "cik", "fiscal_label", "form", "report_date", "accession")},
            "section_count": entry["section_count"],
            "chunk_count":   len(chunks),
            "chunks_path":   str(out_file),
        })

    # ── Chunk transcripts ─────────────────────────────────────────────────────
    transcripts_manifest_p = Path(
        transcripts_manifest_path
        or str(Path(parsed_manifest_path).parent.parent / "raw" / "transcripts" / "transcripts_manifest.json")
    )

    if transcripts_manifest_p.exists():
        transcripts_manifest = json.loads(transcripts_manifest_p.read_text(encoding="utf-8"))
        log.info("=== Chunking %d transcripts (recursive) ===", len(transcripts_manifest))

        for entry in transcripts_manifest:
            local_path = Path(entry.get("local_path", ""))
            if not local_path.exists():
                log.warning("Missing transcript file, skipping: %s", local_path)
                continue

            record = json.loads(local_path.read_text(encoding="utf-8"))
            ticker = record.get("ticker", "")
            fiscal_label = record.get("fiscal_label", "")

            log.info("Chunking transcript %s %s", ticker, fiscal_label)

            chunks = chunk_transcript(record, encoder)
            if not chunks:
                log.warning("  No chunks produced for %s %s", ticker, fiscal_label)
                continue

            out_dir = out_root_p / ticker
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{fiscal_label}_transcript_chunks.json"
            out_file.write_text(
                json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            log.info(
                "  %s %s: %d chunks -> %s", ticker, fiscal_label, len(chunks), out_file.name
            )

            cik = _CIK_MAP.get(ticker, "0000000000")
            chunk_manifest.append({
                "ticker":        ticker,
                "cik":           cik,
                "fiscal_label":  fiscal_label,
                "form":          "transcript",
                "report_date":   record.get("call_date") or "",
                "accession":     f"transcript_{ticker}_{fiscal_label}",
                "section_count": len(chunks),
                "chunk_count":   len(chunks),
                "chunks_path":   str(out_file),
            })
    else:
        log.warning(
            "Transcripts manifest not found at %s — skipping transcripts",
            transcripts_manifest_p
        )

    # ── Write chunk manifest ──────────────────────────────────────────────────
    chunk_manifest_path = out_root_p / "chunk_manifest.json"
    chunk_manifest_path.write_text(
        json.dumps(chunk_manifest, indent=2), encoding="utf-8"
    )

    total_chunks = sum(e["chunk_count"] for e in chunk_manifest)
    filing_entries = sum(1 for e in chunk_manifest if e["form"] != "transcript")
    transcript_entries = sum(1 for e in chunk_manifest if e["form"] == "transcript")

    log.info(
        "Done. %d filing entries + %d transcript entries = %d total entries, "
        "%d total chunks. Manifest: %s",
        filing_entries, transcript_entries,
        len(chunk_manifest), total_chunks, chunk_manifest_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk parsed filings + transcripts for QuarterLens AI."
    )
    parser.add_argument(
        "--manifest",
        default="data/parsed/parsed_manifest.json",
    )
    parser.add_argument("--out", default="data/chunks")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument(
        "--transcripts-manifest",
        default=None,
        help="Path to transcripts_manifest.json. "
             "Defaults to data/raw/transcripts/transcripts_manifest.json",
    )
    args = parser.parse_args()
    run(
        args.manifest,
        args.out,
        args.chunk_size,
        args.overlap,
        args.transcripts_manifest,
    )


if __name__ == "__main__":
    main()