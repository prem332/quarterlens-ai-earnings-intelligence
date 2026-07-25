"""
Sentence handling.

Two views of the same operation:
  - split_sentences: normalized list[str] (used for parent-block grouping)
  - sentence_spans:  (start, end) char offsets that PARTITION the text exactly,
                     so slices can be concatenated back into the original
                     byte-for-byte. Required by the semantic child splitter to
                     hold the parent-reconstruction invariant.
"""
from __future__ import annotations

import re

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "inc", "corp",
    "ltd", "co", "vs", "etc", "approx", "est", "avg",
}

# A sentence boundary: terminal punctuation + whitespace + capital/quote/paren.
_BOUNDARY_RE = re.compile(r"[.!?](\s+)(?=[A-Z\"(])")
_PROTECT_RE = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Corp|Ltd|Co|vs|etc|approx|est|avg)\.",
)
_TRAILING_WORD_RE = re.compile(r"([A-Za-z]+)$")


def split_sentences(text: str) -> list[str]:
    """Normalized sentence list (whitespace-collapsed). For parent grouping."""
    protected = _PROTECT_RE.sub(r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"(])", protected)
    parts = [p.replace("<DOT>", ".") for p in parts]
    return [p.strip() for p in parts if p.strip()]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """
    Contiguous (start, end) spans that partition `text` exactly:
    "".join(text[s:e] for s, e in spans) == text.

    Cuts at the END of the inter-sentence whitespace, so each sentence keeps its
    own trailing whitespace. Abbreviations (Inc., Corp., ...) are not treated as
    boundaries.
    """
    cuts = [0]
    for m in _BOUNDARY_RE.finditer(text):
        word = _TRAILING_WORD_RE.search(text[: m.start()])
        if word and word.group(1).lower() in _ABBREV:
            continue
        cuts.append(m.end())
    cuts.append(len(text))
    cuts = sorted(set(c for c in cuts if 0 <= c <= len(text)))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i] < cuts[i + 1]]
