"""
Chunking configuration — sizes, regexes, and the shared token encoder.

Kept in one small module so every other chunking submodule imports constants
from a single source of truth.
"""
from __future__ import annotations

import re

import tiktoken

ENCODING = "cl100k_base"

# ── Sizing ────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 400          # default target for non-hierarchical section grouping
CHUNK_MIN = 80            # minimum tokens — parents below this are skipped as boilerplate

# Hierarchical targets (Deviation #32 — semantic + small-to-big)
PARENT_TARGET_TOKENS = 1000   # L2 parent block target (non-MDA sections grouped to ~this)
CHILD_TARGET_TOKENS = 250     # L3 semantic child target
CHILD_MIN_TOKENS = 80         # no child smaller than this (tiny tail merged back)
CHILD_MAX_TOKENS = 375        # hard cap per child (target x1.5)
SEMANTIC_PERCENTILE = 25      # consecutive-sentence similarity valley threshold

# Transcript: group this many speaker turns per chunk
TRANSCRIPT_TURNS_PER_CHUNK = 4

# ── CIK map ───────────────────────────────────────────────────────────────────
CIK_MAP = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "NVDA":  "0001045810",
    "GOOGL": "0001652044",
    "META":  "0001326801",
}

# ── Patterns ──────────────────────────────────────────────────────────────────
# ALL-CAPS subsection header at start of a chunk (metadata tagging).
SUBSECTION_HEADER_RE = re.compile(r"(?:^|(?<=[.!?] ))([A-Z][A-Z &/\-]{2,49})(?=[ ][A-Za-z])")

# Macro-topic boundary scanner for MDA structural splitting (Fix 6) — scans
# anywhere in the text; 6+ char run rejects short false positives (MD&A, USG, SEC).
MDA_MACRO_HEADER_RE = re.compile(r"(?:^|(?<=[.!?] ))([A-Z][A-Z &/\-]{5,49})(?=[ ][A-Za-z])")

BULLET_CHAR = "•"  # •

SPEAKER_TURN_RE = re.compile(r"^([A-Z][^:]{2,50}):\s", re.MULTILINE)

MDA_SUBSECTION_KEYWORDS = {
    "revenue", "gross margin", "operating income", "operating expenses",
    "net income", "earnings per share", "segment", "cloud", "productivity",
    "intelligent cloud", "more personal computing", "liquidity", "capital",
    "cash flows", "overview", "critical accounting", "recent accounting",
    "three months", "nine months", "twelve months",
}


def get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def token_count(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))