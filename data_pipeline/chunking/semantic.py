"""
Semantic child splitting (Deviation #32).

Splits an L2 parent block into small, single-topic children and returns (start,
end) offset spans that PARTITION the parent exactly — children slice the original
text and reconstruct it byte-for-byte.

A parent is partitioned into contiguous runs of prose vs. table rows:
  - prose runs → topic-shift breakpoints from local-MiniLM sentence similarity
  - table runs → atomic row-boundary grouping (never split a row mid-cell)
Then a floor pass merges any sub-min child forward and a ceiling pass word-splits
anything still over max, so every child lands in [min, max] where the text allows.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import tiktoken

from .config import (
    CHILD_MAX_TOKENS, CHILD_MIN_TOKENS, CHILD_TARGET_TOKENS, SEMANTIC_PERCENTILE, token_count,
)
from .sentences import sentence_spans

Span = tuple[int, int]

_model: Optional[object] = None
_MODEL_NAME = "all-MiniLM-L6-v2"
_WORD_RE = re.compile(r"\S+")


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model = SentenceTransformer(_MODEL_NAME, device="cpu")
    return _model


# ── Run classification (prose vs table) ─────────────────────────────────────────

def _line_spans(text: str) -> list[Span]:
    """Newline-delimited spans covering [0, len(text)]; each keeps its trailing '\\n'."""
    spans, pos, n = [], 0, len(text)
    while pos < n:
        nl = text.find("\n", pos)
        if nl == -1:
            spans.append((pos, n))
            break
        spans.append((pos, nl + 1))
        pos = nl + 1
    return spans or [(0, n)]


def _classify_runs(text: str) -> list[tuple[int, int, bool]]:
    """Contiguous (start, end, is_table) runs — table lines contain a '|' cell separator."""
    lines = _line_spans(text)
    runs: list[tuple[int, int, bool]] = []
    cur_s, cur_e = lines[0][0], lines[0][1]
    cur_table = "|" in text[cur_s:cur_e]
    for s, e in lines[1:]:
        is_table = "|" in text[s:e]
        if is_table == cur_table:
            cur_e = e
        else:
            runs.append((cur_s, cur_e, cur_table))
            cur_s, cur_e, cur_table = s, e, is_table
    runs.append((cur_s, cur_e, cur_table))
    return runs


# ── Per-run splitters (spans relative to the run text) ──────────────────────────

def _row_spans(text: str, encoder: tiktoken.Encoding, target: int) -> list[Span]:
    lines = _line_spans(text)
    out: list[Span] = []
    cur_start, cur_tok = lines[0][0], 0
    for s, e in lines:
        seg_tok = token_count(text[s:e], encoder)
        if cur_tok > 0 and cur_tok + seg_tok > target:
            out.append((cur_start, s))
            cur_start, cur_tok = s, seg_tok
        else:
            cur_tok += seg_tok
    out.append((cur_start, lines[-1][1]))
    return out


def _prose_spans(text: str, encoder: tiktoken.Encoding, target: int,
                 min_tokens: int, max_tokens: int, percentile: int) -> list[Span]:
    spans = sentence_spans(text)
    if len(spans) <= 1 or token_count(text, encoder) <= target:
        return [(0, len(text))]

    sentences = [text[s:e] for s, e in spans]
    embeddings = _get_model().encode(sentences, normalize_embeddings=True, show_progress_bar=False)
    sims = (embeddings[:-1] * embeddings[1:]).sum(axis=1)   # cosine (normalized)
    threshold = float(np.percentile(sims, percentile))

    out: list[Span] = []
    cur_start = spans[0][0]
    cur_tok = token_count(sentences[0], encoder)
    for i in range(1, len(spans)):
        s_tok = token_count(sentences[i], encoder)
        valley = sims[i - 1] < threshold and cur_tok >= min_tokens
        overflow = cur_tok + s_tok > max_tokens and cur_tok >= min_tokens
        if valley or overflow:
            out.append((cur_start, spans[i][0]))
            cur_start, cur_tok = spans[i][0], s_tok
        else:
            cur_tok += s_tok
    out.append((cur_start, spans[-1][1]))
    return out


# ── Floor / ceiling passes over parent-relative spans ───────────────────────────

def _merge_small(spans: list[Span], text: str, encoder: tiktoken.Encoding, min_tokens: int) -> list[Span]:
    """Grow a running span until it clears min_tokens; merge a sub-min tail back."""
    if len(spans) <= 1:
        return spans
    out: list[Span] = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if token_count(text[cur_s:cur_e], encoder) < min_tokens:
            cur_e = e
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    out.append((cur_s, cur_e))
    if len(out) > 1 and token_count(text[out[-1][0]:out[-1][1]], encoder) < min_tokens:
        ls, _ = out.pop()
        ps, _ = out[-1]
        out[-1] = (ps, spans[-1][1])
    return out


def _enforce_max(spans: list[Span], text: str, encoder: tiktoken.Encoding, max_tokens: int) -> list[Span]:
    """Last-resort word-boundary split for any span with no internal sentence/row break."""
    out: list[Span] = []
    for s, e in spans:
        if token_count(text[s:e], encoder) <= max_tokens:
            out.append((s, e))
            continue
        words = list(_WORD_RE.finditer(text, s, e))
        if len(words) <= 1:
            out.append((s, e))
            continue
        start, tok = s, 0
        for w in words:
            w_tok = token_count(w.group(), encoder)
            if tok > 0 and tok + w_tok > max_tokens:
                out.append((start, w.start()))
                start, tok = w.start(), w_tok
            else:
                tok += w_tok
        out.append((start, e))
    return out


def semantic_split(
    text: str,
    encoder: tiktoken.Encoding,
    target: int = CHILD_TARGET_TOKENS,
    min_tokens: int = CHILD_MIN_TOKENS,
    max_tokens: int = CHILD_MAX_TOKENS,
    percentile: int = SEMANTIC_PERCENTILE,
) -> list[Span]:
    """Offset spans partitioning `text` into semantic/table-aware children."""
    spans: list[Span] = []
    for rs, re_, is_table in _classify_runs(text):
        seg = text[rs:re_]
        sub = (_row_spans(seg, encoder, target) if is_table
               else _prose_spans(seg, encoder, target, min_tokens, max_tokens, percentile))
        spans.extend((rs + a, rs + b) for a, b in sub)

    spans = _enforce_max(spans, text, encoder, max_tokens)
    spans = _merge_small(spans, text, encoder, min_tokens)
    return spans
