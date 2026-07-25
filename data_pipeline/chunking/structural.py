"""
Structural splitting — the deterministic, non-semantic layer.

  - MDA macro-header / bullet boundaries (Fix 6)
  - Table detection + row-safe splitting (table-aware parsing companion)
  - Token-greedy sentence grouping (used to build L2 parent blocks)
  - Subsection metadata detection

Table rows are treated as atomic units everywhere: a cell value split across a
boundary silently corrupts the figure.
"""
from __future__ import annotations

import tiktoken

from .config import (
    BULLET_CHAR, CHUNK_MIN, CHUNK_SIZE, MDA_MACRO_HEADER_RE,
    MDA_SUBSECTION_KEYWORDS, SUBSECTION_HEADER_RE, token_count,
)
from .sentences import split_sentences


def is_row_shaped(text: str) -> bool:
    """Table-extracted text: newline-delimited rows with '|' cell separators."""
    return "\n" in text and text.count("|") >= 2


# ── Subsection metadata ─────────────────────────────────────────────────────────

def detect_subsection(text: str, section: str) -> str:
    if section not in ("mda", "risk_factors", "business"):
        return ""
    m = SUBSECTION_HEADER_RE.match(text.strip())
    if m and len(m.group(1).strip()) >= 3:
        return m.group(1).strip().lower().replace(" ", "_")
    lowered = text.lower().strip()
    for keyword in MDA_SUBSECTION_KEYWORDS:
        if lowered.startswith(keyword):
            return keyword.replace(" ", "_")
    return ""


# ── MDA structural boundaries (Fix 6) ───────────────────────────────────────────

def split_mda_boundaries(text: str) -> list[str]:
    """Split MDA text at ALL-CAPS macro headers and bullet markers. [text] if none."""
    positions = [m.start() for m in MDA_MACRO_HEADER_RE.finditer(text)]
    bounds = sorted(set([0] + positions + [len(text)]))
    segments = [text[bounds[i]:bounds[i + 1]].strip() for i in range(len(bounds) - 1)]
    segments = [s for s in segments if s]

    units: list[str] = []
    for seg in segments:
        if BULLET_CHAR not in seg:
            units.append(seg)
            continue
        parts = [p.strip() for p in seg.split(BULLET_CHAR) if p.strip()]
        units.extend(parts)
    return units


def coalesce_mda_units(
    units: list[str],
    encoder: tiktoken.Encoding,
    min_tokens: int = CHUNK_MIN,
    max_tokens: int = CHUNK_SIZE,
) -> list[str]:
    """Merge sub-min units forward just enough to clear the floor — keeps topic separation."""
    if not units:
        return []
    merged: list[str] = []
    current, current_tok = units[0], token_count(units[0], encoder)
    for nxt in units[1:]:
        nxt_tok = token_count(nxt, encoder)
        if current_tok < min_tokens and current_tok + nxt_tok <= max_tokens:
            current, current_tok = current + " " + nxt, current_tok + nxt_tok
        else:
            merged.append(current)
            current, current_tok = nxt, nxt_tok
    merged.append(current)
    return merged


# ── Row-safe oversized split (strings — used when building parents) ─────────────

def split_oversized_unit(
    text: str,
    encoder: tiktoken.Encoding,
    target_tokens: int,
    min_tokens: int,
) -> list[str]:
    """Split an oversized unit; table rows at line boundaries, prose at word boundaries."""
    chunks: list[str] = []
    if is_row_shaped(text):
        group, group_tokens = [], 0
        for line in (ln for ln in text.split("\n") if ln.strip()):
            line_tokens = token_count(line, encoder)
            if line_tokens > target_tokens:
                if group:
                    _emit("\n".join(group), encoder, min_tokens, chunks)
                    group, group_tokens = [], 0
                chunks.extend(split_oversized_unit(line, encoder, target_tokens, min_tokens))
                continue
            if group_tokens + line_tokens > target_tokens and group:
                _emit("\n".join(group), encoder, min_tokens, chunks)
                group, group_tokens = [line], line_tokens
            else:
                group.append(line)
                group_tokens += line_tokens
        if group:
            _emit("\n".join(group), encoder, min_tokens, chunks)
        return chunks

    group, group_tokens = [], 0
    for word in text.split():
        w_tok = token_count(word, encoder)
        if group_tokens + w_tok > target_tokens and group:
            _emit(" ".join(group), encoder, min_tokens, chunks)
            group, group_tokens = [word], w_tok
        else:
            group.append(word)
            group_tokens += w_tok
    if group:
        _emit(" ".join(group), encoder, min_tokens, chunks)
    return chunks


def _emit(text: str, encoder: tiktoken.Encoding, min_tokens: int, out: list[str]) -> None:
    if token_count(text, encoder) >= min_tokens:
        out.append(text)


# ── Token-greedy sentence grouping (builds L2 parent blocks) ────────────────────

def group_sentences_into_chunks(
    sentences: list[str],
    encoder: tiktoken.Encoding,
    target_tokens: int = CHUNK_SIZE,
    min_tokens: int = CHUNK_MIN,
) -> list[str]:
    """Group sentences up to target_tokens, never mid-sentence; skip sub-min groups."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        s_tokens = token_count(sentence, encoder)
        if s_tokens > target_tokens:
            if current:
                _emit(" ".join(current), encoder, min_tokens, chunks)
                current, current_tokens = [], 0
            chunks.extend(split_oversized_unit(sentence, encoder, target_tokens, min_tokens))
            continue
        if current_tokens + s_tokens > target_tokens and current:
            _emit(" ".join(current), encoder, min_tokens, chunks)
            current, current_tokens = [sentence], s_tokens
        else:
            current.append(sentence)
            current_tokens += s_tokens
    if current:
        _emit(" ".join(current), encoder, min_tokens, chunks)
    return chunks


def section_parent_blocks(section: str, text: str, encoder: tiktoken.Encoding,
                          parent_target: int) -> list[str]:
    """
    L2 parent blocks for one section.
      MDA  → macro-header/bullet units (coalesced) — the natural topic blocks.
      else → token-greedy ~parent_target blocks over sentences.
    Sub-min blocks are dropped (boilerplate) — dropping a whole parent keeps the
    child→parent reconstruction invariant intact for retained parents.
    """
    if section == "mda":
        units = split_mda_boundaries(text)
        if len(units) > 1:
            blocks = coalesce_mda_units(units, encoder, max_tokens=parent_target)
            return [b for b in blocks if token_count(b, encoder) >= CHUNK_MIN]
    sentences = split_sentences(text)
    return group_sentences_into_chunks(sentences, encoder, target_tokens=parent_target)
