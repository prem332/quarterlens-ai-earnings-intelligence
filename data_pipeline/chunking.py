from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from pathlib import Path

import tiktoken

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("chunking")

CHUNK_SIZE = 512      # tokens
CHUNK_OVERLAP = 50    # tokens
ENCODING = "cl100k_base"  # matches text-embedding-3-small / GPT-4 family


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def chunk_text(
    text: str,
    encoder: tiktoken.Encoding,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping token windows.
    Returns list of decoded string chunks.
    """
    tokens = encoder.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(encoder.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks


def chunk_filing(parsed_sections: list[dict], encoder: tiktoken.Encoding) -> list[dict]:
    """
    Chunk all sections from one parsed filing.
    Each input dict has: ticker, cik, fiscal_label, report_date, form,
                         accession, section, text.
    Each output chunk adds: chunk_id, chunk_index, chunk_total (per section).
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


def run(
    parsed_manifest_path: str,
    out_root: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> None:
    manifest_p = Path(parsed_manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Parsed manifest not found: {parsed_manifest_path}")

    parsed_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    out_root_p = Path(out_root)
    encoder = _get_encoder()
    chunk_manifest: list[dict] = []

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

    chunk_manifest_path = out_root_p / "chunk_manifest.json"
    chunk_manifest_path.write_text(
        json.dumps(chunk_manifest, indent=2), encoding="utf-8"
    )

    total_chunks = sum(e["chunk_count"] for e in chunk_manifest)
    log.info(
        "Done. %d/%d filings chunked, %d total chunks. Manifest: %s",
        len(chunk_manifest), len(parsed_manifest), total_chunks, chunk_manifest_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk parsed filings for QuarterLens AI.")
    parser.add_argument(
        "--manifest",
        default="data/parsed/parsed_manifest.json",
    )
    parser.add_argument("--out", default="data/chunks")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()
    run(args.manifest, args.out, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()