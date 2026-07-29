"""
data_pipeline/manifest_io.py

Shared manifest read/project/write helpers for the ingestion chain.

The pipeline is a 5-stage manifest chain, each stage consuming the previous
stage's manifest and emitting its own:

    manifest.json -> parsed_manifest.json -> chunk_manifest.json
                  -> embedding_manifest.json -> AI Search index

Every stage repeated the same three blocks: the read+exists guard, the
per-item "missing file, skip" warning, and the provenance-key projection.
They live here instead so the six provenance keys can't drift between stages.

Two deliberate non-goals:
  - No JSON *content* writer. Dump flags differ per payload on purpose:
    manifests use indent=2; chunk/section files add ensure_ascii=False so
    filing text isn't mangled; embedding vector files use neither, because
    indenting a 1536-float array per chunk bloats the file several-fold.
  - financials_fetcher.load_manifest is intentionally NOT migrated. It has a
    different contract (raises ValueError on empty, accepts a dict-shaped
    manifest with "filings"/"documents" keys) that the four stages here do not.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

# The provenance keys carried unchanged through every stage of the chain.
PROVENANCE_KEYS = ("ticker", "cik", "fiscal_label", "form", "report_date", "accession")


def read_manifest(path: str | Path, label: str) -> list[dict]:
    """
    Load a manifest, or raise FileNotFoundError naming which one is missing.

    `label` is the human name used in the error ("Manifest", "Chunk manifest").
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def exists_or_warn(path: Path, what: str, log: logging.Logger) -> bool:
    """
    True if `path` exists; otherwise log a skip warning and return False.

    The caller's own logger is passed in so the record keeps that module's
    logger name rather than this one's.
    """
    if path.exists():
        return True
    log.warning("Missing %s, skipping: %s", what, path)
    return False


def provenance(entry: dict) -> dict:
    """Project the six provenance keys forward from an upstream manifest entry."""
    return {k: entry[k] for k in PROVENANCE_KEYS}


def write_manifest(path: str | Path, entries: list[dict]) -> Path:
    """Write a manifest (indent=2, ASCII-escaped — these hold tickers/labels only)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return p
