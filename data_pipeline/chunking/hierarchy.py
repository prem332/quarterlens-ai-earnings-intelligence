"""
Parent/child assembly + pipeline runner.

Filings: section → L2 parent blocks (structural) → L3 semantic children. Each
child is an exact offset-slice of its parent and carries parent_id / parent_index
/ parent_total so retrieval can reconstruct the parent ("small-to-big").
Transcripts: speaker-turn chunking with a degenerate hierarchy (parent = self).
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import tiktoken

from .config import (
    CHUNK_MIN, CHUNK_SIZE, CIK_MAP, PARENT_TARGET_TOKENS,
    SPEAKER_TURN_RE, TRANSCRIPT_TURNS_PER_CHUNK, get_encoder, token_count,
)
from .semantic import semantic_split
from .structural import (
    detect_subsection, group_sentences_into_chunks, section_parent_blocks, split_sentences,
)

log = logging.getLogger("chunking")


def _record(section: dict, text: str, chunk_index: int, chunk_total: int,
            parent_id: str, parent_index: int, parent_total: int) -> dict:
    return {
        "ticker":       section["ticker"],
        "cik":          section["cik"],
        "fiscal_label": section["fiscal_label"],
        "report_date":  section["report_date"],
        "form":         section["form"],
        "accession":    section["accession"],
        "section":      section["section"],
        "subsection":   detect_subsection(text, section["section"]),
        "chunk_id":     str(uuid.uuid4()),
        "chunk_index":  chunk_index,
        "chunk_total":  chunk_total,
        "parent_id":    parent_id,
        "parent_index": parent_index,
        "parent_total": parent_total,
        "text":         text,
    }


# ── Filing chunker ──────────────────────────────────────────────────────────────

def chunk_filing(parsed_sections: list[dict], encoder: tiktoken.Encoding) -> list[dict]:
    """Hierarchical + semantic chunking of one parsed filing (all its sections)."""
    chunks: list[dict] = []
    for section in parsed_sections:
        text = section.get("text", "").strip()
        if not text:
            continue

        # L2 parents → L3 semantic children (exact offset slices of each parent).
        children: list[tuple[str, str, int, int]] = []  # (text, parent_id, p_index, p_total)
        for parent_text in section_parent_blocks(
            section["section"], text, encoder, PARENT_TARGET_TOKENS,
        ):
            spans = semantic_split(parent_text, encoder)
            parent_id = str(uuid.uuid4())
            for p_index, (s, e) in enumerate(spans):
                children.append((parent_text[s:e], parent_id, p_index, len(spans)))

        total = len(children)
        for chunk_index, (child_text, parent_id, p_index, p_total) in enumerate(children):
            chunks.append(_record(section, child_text, chunk_index, total,
                                   parent_id, p_index, p_total))
    return chunks


# ── Transcript chunker (degenerate hierarchy: parent = self) ────────────────────

def _split_speaker_turns(text: str) -> list[tuple[str, str]]:
    matches = list(SPEAKER_TURN_RE.finditer(text))
    if not matches:
        return [("", text)]
    turns: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            turns.append((m.group(1).strip(), content))
    return turns


def _transcript_record(record: dict, cik: str, section_name: str, subsection: str,
                       chunk_index: int, chunk_total: int, text: str) -> dict:
    chunk_id = str(uuid.uuid4())
    return {
        "ticker":       record.get("ticker", ""),
        "cik":          cik,
        "fiscal_label": record.get("fiscal_label", ""),
        "report_date":  record.get("call_date") or "",
        "form":         "transcript",
        "accession":    f"transcript_{record.get('ticker', '')}_{record.get('fiscal_label', '')}",
        "section":      section_name,
        "subsection":   subsection,
        "chunk_id":     chunk_id,
        "chunk_index":  chunk_index,
        "chunk_total":  chunk_total,
        "parent_id":    chunk_id,   # degenerate hierarchy — parent is the chunk itself
        "parent_index": 0,
        "parent_total": 1,
        "text":         text,
    }


def chunk_transcript(record: dict, encoder: tiktoken.Encoding) -> list[dict]:
    ticker = record.get("ticker", "")
    text = (record.get("text") or "").strip()
    cik = CIK_MAP.get(ticker, "0000000000")
    if not text:
        return []

    turns = _split_speaker_turns(text)

    if len(turns) == 1 and not turns[0][0]:   # no speaker pattern — sentence fallback
        parts = group_sentences_into_chunks(split_sentences(text), encoder)
        return [
            _transcript_record(record, cik, "transcript_part_0", "", i, len(parts), t)
            for i, t in enumerate(parts)
        ]

    groups: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_tokens = 0
    for speaker, content in turns:
        t_tokens = token_count(f"{speaker}: {content}" if speaker else content, encoder)
        if (len(current) >= TRANSCRIPT_TURNS_PER_CHUNK
                or current_tokens + t_tokens > CHUNK_SIZE * 1.5) and current:
            groups.append(current)
            current, current_tokens = [(speaker, content)], t_tokens
        else:
            current.append((speaker, content))
            current_tokens += t_tokens
    if current:
        groups.append(current)

    chunks: list[dict] = []
    for idx, group in enumerate(groups):
        body = " ".join(f"{spk}: {cnt}" if spk else cnt for spk, cnt in group).strip()
        if not body or token_count(body, encoder) < CHUNK_MIN:
            continue
        subsection = group[0][0].lower().replace(" ", "_") if group[0][0] else ""
        chunks.append(_transcript_record(
            record, cik, f"transcript_part_{idx}", subsection, idx, len(groups), body))
    return chunks


# ── Pipeline runner ──────────────────────────────────────────────────────────────

def run(parsed_manifest_path: str, out_root: str,
        transcripts_manifest_path: str | None = None) -> None:
    manifest_p = Path(parsed_manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Parsed manifest not found: {parsed_manifest_path}")

    parsed_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    out_root_p = Path(out_root)
    encoder = get_encoder()
    chunk_manifest: list[dict] = []

    log.info("=== Chunking %d filings (hierarchical + semantic) ===", len(parsed_manifest))
    for entry in parsed_manifest:
        parsed_path = Path(entry["parsed_path"])
        if not parsed_path.exists():
            log.warning("Missing parsed file, skipping: %s", parsed_path)
            continue
        sections = json.loads(parsed_path.read_text(encoding="utf-8"))
        chunks = chunk_filing(sections, encoder)
        out_file = out_root_p / entry["ticker"] / (
            f"{entry['fiscal_label']}_{entry['form'].replace('-', '')}_chunks.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("  %s %s: %d chunks", entry["ticker"], entry["fiscal_label"], len(chunks))
        chunk_manifest.append({
            **{k: entry[k] for k in
               ("ticker", "cik", "fiscal_label", "form", "report_date", "accession")},
            "section_count": entry["section_count"],
            "chunk_count":   len(chunks),
            "chunks_path":   str(out_file),
        })

    transcripts_p = Path(
        transcripts_manifest_path
        or Path(parsed_manifest_path).parent.parent / "raw" / "transcripts" / "transcripts_manifest.json")
    if transcripts_p.exists():
        log.info("=== Chunking transcripts (speaker-turn aware) ===")
        for entry in json.loads(transcripts_p.read_text(encoding="utf-8")):
            local_path = Path(entry.get("local_path", ""))
            if not local_path.exists():
                log.warning("Missing transcript, skipping: %s", local_path)
                continue
            record = json.loads(local_path.read_text(encoding="utf-8"))
            chunks = chunk_transcript(record, encoder)
            if not chunks:
                continue
            ticker, label = record.get("ticker", ""), record.get("fiscal_label", "")
            out_file = out_root_p / ticker / f"{label}_transcript_chunks.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
            log.info("  %s %s: %d chunks", ticker, label, len(chunks))
            chunk_manifest.append({
                "ticker": ticker, "cik": CIK_MAP.get(ticker, "0000000000"),
                "fiscal_label": label, "form": "transcript",
                "report_date": record.get("call_date") or "",
                "accession": f"transcript_{ticker}_{label}",
                "section_count": len(chunks), "chunk_count": len(chunks),
                "chunks_path": str(out_file),
            })
    else:
        log.warning("Transcripts manifest not found at %s — skipping", transcripts_p)

    manifest_out = out_root_p / "chunk_manifest.json"
    manifest_out.write_text(json.dumps(chunk_manifest, indent=2), encoding="utf-8")
    total = sum(e["chunk_count"] for e in chunk_manifest)
    log.info("Done. %d entries, %d total chunks. Manifest: %s",
             len(chunk_manifest), total, manifest_out)
