"""
data_pipeline/chunking.py

Structure-aware chunking for QuarterLens AI — Phase 4 (deviation #29).

Strategy: Sentence-boundary chunking with zero overlap for filings,
speaker-turn chunking for transcripts.

Key changes from Phase 2 (RecursiveCharacterTextSplitter):
  - ZERO overlap: eliminates sliding-window duplicate density (was 0.50-0.60)
  - Sentence-boundary grouping: groups sentences into ~400-token chunks
    without splitting mid-sentence
  - Subsection detection: detects ALL-CAPS headers within sections to add
    'subsection' metadata field for future section-aware filtering
  - Speaker-turn chunking for transcripts: groups 4 speaker turns per chunk
    instead of fixed character windows
  - Minimum chunk size: 80 tokens — skips boilerplate headers that pollute index
  - Filterable 'subsection' field added to chunks (requires indexer.py update)

Pipeline:
    python -m data_pipeline.chunking    # chunk filings + transcripts
    python -m data_pipeline.embedding   # embed all chunks
    python -m data_pipeline.indexer     # recreate index + upload all

Baseline locked in MLflow before this change:
    baseline-evidence-consistency-25:
    faithfulness=0.9274, answer_relevancy=0.8264, context_precision=0.2960,
    precision@5=0.6500, recall@5=1.0000, llm_judge=3.0560
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from pathlib import Path

import tiktoken

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("chunking")

# Chunk sizing
CHUNK_SIZE    = 400    # target tokens per chunk (reduced from 512 for tighter evidence)
CHUNK_OVERLAP = 0      # zero overlap — eliminates sliding-window duplicate density
CHUNK_MIN     = 80     # minimum tokens — skip boilerplate headers
ENCODING      = "cl100k_base"

# Transcript: group this many speaker turns per chunk
_TRANSCRIPT_TURNS_PER_CHUNK = 4

# CIK map
_CIK_MAP = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "NVDA":  "0001045810",
    "GOOGL": "0001652044",
    "META":  "0001326801",
}

# ALL-CAPS subsection header pattern (3-50 chars, appears after sentence end or at start)
# Matches: OVERVIEW, SEGMENT RESULTS, CRITICAL ACCOUNTING ESTIMATES, etc.
_SUBSECTION_HEADER_RE = re.compile(
    r'(?:^|(?<=[.!?] ))([A-Z][A-Z &/\-]{2,49})(?=[ ][A-Za-z])'
)

# Known financial subsection keywords for MDA
_MDA_SUBSECTION_KEYWORDS = {
    "revenue", "gross margin", "operating income", "operating expenses",
    "net income", "earnings per share", "segment", "cloud", "productivity",
    "intelligent cloud", "more personal computing", "liquidity", "capital",
    "cash flows", "overview", "critical accounting", "recent accounting",
    "three months", "nine months", "twelve months",
}

# Speaker turn pattern for transcripts
_SPEAKER_TURN_RE = re.compile(r'^([A-Z][^:]{2,50}):\s', re.MULTILINE)


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def _token_count(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))


# ── Subsection detection ──────────────────────────────────────────────────────

def _detect_subsection(text: str, section: str) -> str:
    """
    Detect the subsection header this text belongs to.
    Returns empty string if no subsection detected.
    Only meaningful for mda and risk_factors sections.
    """
    if section not in ("mda", "risk_factors", "business"):
        return ""

    # Check for ALL-CAPS header at start of text
    m = _SUBSECTION_HEADER_RE.match(text.strip())
    if m:
        candidate = m.group(1).strip()
        if len(candidate) >= 3:
            return candidate.lower().replace(" ", "_")

    # Check for known MDA keywords at start
    text_lower = text.lower().strip()
    for keyword in _MDA_SUBSECTION_KEYWORDS:
        if text_lower.startswith(keyword):
            return keyword.replace(" ", "_")

    return ""


# ── Sentence splitter ─────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences on '. ', '! ', '? ' boundaries.
    Handles common abbreviations (Mr., Dr., Inc., etc.) to avoid false splits.
    """
    # Protect common abbreviations
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Corp|Ltd|Co|vs|etc|approx|est|avg)\.',
                  r'\1<DOT>', text)
    # Split on sentence-ending punctuation followed by space+capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"(])', text)
    # Restore protected dots
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


# ── Sentence grouping into chunks ─────────────────────────────────────────────

def _group_sentences_into_chunks(
    sentences: list[str],
    encoder: tiktoken.Encoding,
    target_tokens: int = CHUNK_SIZE,
    min_tokens: int = CHUNK_MIN,
) -> list[str]:
    """
    Group sentences into chunks targeting target_tokens with zero overlap.
    Never splits mid-sentence. Skips groups under min_tokens (boilerplate).
    """
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = _token_count(sentence, encoder)

        # Single sentence exceeds target — emit current group first, then
        # split the long sentence recursively
        if s_tokens > target_tokens:
            if current_sentences:
                text = " ".join(current_sentences)
                if _token_count(text, encoder) >= min_tokens:
                    chunks.append(text)
                current_sentences = []
                current_tokens = 0
            # Split long sentence at word boundaries
            words = sentence.split()
            word_group: list[str] = []
            word_tokens = 0
            for word in words:
                w_tok = _token_count(word, encoder)
                if word_tokens + w_tok > target_tokens and word_group:
                    text = " ".join(word_group)
                    if _token_count(text, encoder) >= min_tokens:
                        chunks.append(text)
                    word_group = [word]
                    word_tokens = w_tok
                else:
                    word_group.append(word)
                    word_tokens += w_tok
            if word_group:
                text = " ".join(word_group)
                if _token_count(text, encoder) >= min_tokens:
                    chunks.append(text)
            continue

        if current_tokens + s_tokens > target_tokens and current_sentences:
            text = " ".join(current_sentences)
            if _token_count(text, encoder) >= min_tokens:
                chunks.append(text)
            current_sentences = [sentence]
            current_tokens = s_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += s_tokens

    if current_sentences:
        text = " ".join(current_sentences)
        if _token_count(text, encoder) >= min_tokens:
            chunks.append(text)

    return chunks


# ── Filing chunker ────────────────────────────────────────────────────────────

def chunk_filing(parsed_sections: list[dict], encoder: tiktoken.Encoding) -> list[dict]:
    """
    Chunk all sections from one parsed filing using sentence-boundary grouping.

    Zero overlap — eliminates sliding-window duplicate density.
    Adds 'subsection' metadata field for future section-aware filtering.

    Each input dict has: ticker, cik, fiscal_label, report_date, form,
                         accession, section, text.
    Each output chunk adds: chunk_id, chunk_index, chunk_total, subsection.
    """
    chunks: list[dict] = []

    for section in parsed_sections:
        text = section.get("text", "").strip()
        if not text:
            continue

        sentences = _split_sentences(text)
        if not sentences:
            continue

        chunk_texts = _group_sentences_into_chunks(sentences, encoder)
        total = len(chunk_texts)

        for i, chunk_text in enumerate(chunk_texts):
            subsection = _detect_subsection(chunk_text, section["section"])
            chunks.append({
                "ticker":       section["ticker"],
                "cik":          section["cik"],
                "fiscal_label": section["fiscal_label"],
                "report_date":  section["report_date"],
                "form":         section["form"],
                "accession":    section["accession"],
                "section":      section["section"],
                "subsection":   subsection,
                "chunk_id":     str(uuid.uuid4()),
                "chunk_index":  i,
                "chunk_total":  total,
                "text":         chunk_text,
            })

    return chunks


# ── Transcript chunker ────────────────────────────────────────────────────────

def _split_speaker_turns(text: str) -> list[tuple[str, str]]:
    """
    Split transcript text into (speaker, content) tuples.
    Falls back to sentence splitting if no speaker pattern found.
    """
    matches = list(_SPEAKER_TURN_RE.finditer(text))
    if not matches:
        return [("", text)]

    turns = []
    for i, m in enumerate(matches):
        speaker = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            turns.append((speaker, content))
    return turns


def chunk_transcript(record: dict, encoder: tiktoken.Encoding) -> list[dict]:
    """
    Chunk one transcript using speaker-turn-aware grouping.

    Groups _TRANSCRIPT_TURNS_PER_CHUNK speaker turns per chunk.
    Falls back to sentence-boundary chunking if no speaker pattern detected.
    Adds 'subsection' field (speaker name or empty) for metadata consistency.
    """
    ticker = record.get("ticker", "")
    fiscal_label = record.get("fiscal_label", "")
    call_date = record.get("call_date") or ""
    text = (record.get("text") or "").strip()
    cik = _CIK_MAP.get(ticker, "0000000000")

    if not text:
        return []

    turns = _split_speaker_turns(text)

    # If no speaker pattern detected, fall back to sentence chunking
    if len(turns) == 1 and not turns[0][0]:
        sentences = _split_sentences(text)
        chunk_texts = _group_sentences_into_chunks(sentences, encoder)
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunks.append({
                "ticker":       ticker,
                "cik":          cik,
                "fiscal_label": fiscal_label,
                "report_date":  call_date,
                "form":         "transcript",
                "accession":    f"transcript_{ticker}_{fiscal_label}",
                "section":      "transcript_part_0",
                "subsection":   "",
                "chunk_id":     str(uuid.uuid4()),
                "chunk_index":  i,
                "chunk_total":  len(chunk_texts),
                "text":         chunk_text,
            })
        return chunks

    # Group turns into chunks of _TRANSCRIPT_TURNS_PER_CHUNK
    chunks: list[dict] = []
    turn_groups: list[list[tuple[str, str]]] = []
    current_group: list[tuple[str, str]] = []
    current_tokens = 0

    for speaker, content in turns:
        turn_text = f"{speaker}: {content}" if speaker else content
        t_tokens = _token_count(turn_text, encoder)

        if (len(current_group) >= _TRANSCRIPT_TURNS_PER_CHUNK or
                current_tokens + t_tokens > CHUNK_SIZE * 1.5) and current_group:
            turn_groups.append(current_group)
            current_group = [(speaker, content)]
            current_tokens = t_tokens
        else:
            current_group.append((speaker, content))
            current_tokens += t_tokens

    if current_group:
        turn_groups.append(current_group)

    # Assign section names based on group position in transcript
    total_groups = len(turn_groups)
    for group_idx, group in enumerate(turn_groups):
        chunk_text = " ".join(
            f"{spk}: {cnt}" if spk else cnt
            for spk, cnt in group
        ).strip()

        if not chunk_text or _token_count(chunk_text, encoder) < CHUNK_MIN:
            continue

        # Section name encodes position in transcript for backward compat
        section_name = f"transcript_part_{group_idx}"
        # Subsection = first speaker in group (useful for FinBERT routing)
        first_speaker = group[0][0] if group[0][0] else ""

        chunks.append({
            "ticker":       ticker,
            "cik":          cik,
            "fiscal_label": fiscal_label,
            "report_date":  call_date,
            "form":         "transcript",
            "accession":    f"transcript_{ticker}_{fiscal_label}",
            "section":      section_name,
            "subsection":   first_speaker.lower().replace(" ", "_") if first_speaker else "",
            "chunk_id":     str(uuid.uuid4()),
            "chunk_index":  group_idx,
            "chunk_total":  total_groups,
            "text":         chunk_text,
        })

    return chunks


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run(
    parsed_manifest_path: str,
    out_root: str,
    chunk_size: int = CHUNK_SIZE,
    transcripts_manifest_path: str | None = None,
) -> None:
    """
    Chunk all filings from parsed_manifest + all transcripts from
    transcripts_manifest into chunk_manifest.json.

    embedding.py and indexer.py process all entries unchanged,
    except indexer.py needs the new 'subsection' field added to the schema.
    """
    manifest_p = Path(parsed_manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Parsed manifest not found: {parsed_manifest_path}")

    parsed_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    out_root_p = Path(out_root)
    encoder = _get_encoder()
    chunk_manifest: list[dict] = []

    # ── Chunk filings ─────────────────────────────────────────────────────────
    log.info("=== Chunking %d filings (structure-aware, zero overlap) ===", len(parsed_manifest))

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
        log.info("=== Chunking %d transcripts (speaker-turn aware) ===", len(transcripts_manifest))

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
    parser.add_argument("--manifest", default="data/parsed/parsed_manifest.json")
    parser.add_argument("--out", default="data/chunks")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--transcripts-manifest",
        default=None,
        help="Path to transcripts_manifest.json.",
    )
    args = parser.parse_args()
    run(
        args.manifest,
        args.out,
        args.chunk_size,
        args.transcripts_manifest,
    )


if __name__ == "__main__":
    main()