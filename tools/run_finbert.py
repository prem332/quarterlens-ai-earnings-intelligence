"""
FinBERT (ProsusAI/finbert) sentiment analysis over transcript text.
Non-LLM path — deterministic per input, cheap, purpose-built for financial tone.

Design decisions:
  - Lazy singleton: model loads once per process on first call (~440MB, CPU-only)
  - 512-token hard limit per window: FinBERT's positional embedding constraint
  - Sentence-level windowing: splits on sentence boundaries to avoid cutting mid-phrase
  - Aggregation: weighted mean of per-window scores (weight = sentence count in window)

Tool signature (matches tool_registry.py):
    run_finbert(text, chunk_size=400)   # chunk_size in tokens, leave headroom for [CLS]/[SEP]

Returns:
    {
        "aggregate": {
            "label": "positive" | "negative" | "neutral",
            "scores": {"positive": float, "negative": float, "neutral": float}
        },
        "windows": [
            {
                "text_preview": str,   # first 120 chars
                "label": str,
                "scores": {"positive": float, "negative": float, "neutral": float},
                "sentence_count": int
            },
            ...
        ],
        "window_count": int
    }
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy singleton — model + tokenizer loaded once, on first call
# ---------------------------------------------------------------------------
_pipeline: Optional[object] = None
_MODEL_NAME = "ProsusAI/finbert"


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline  # type: ignore
        _pipeline = hf_pipeline(
            "text-classification",
            model=_MODEL_NAME,
            tokenizer=_MODEL_NAME,
            top_k=None,          # return all three label scores
            device=-1,           # CPU; set to 0 for GPU if available
            truncation=True,
            max_length=512,
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Sentence splitter (no NLTK dependency — keeps requirements lean)
# ---------------------------------------------------------------------------
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _build_windows(sentences: list[str], max_tokens: int) -> list[list[str]]:
    """
    Pack sentences into windows that stay under max_tokens (approximate by
    whitespace-split word count; FinBERT tokenizer expands ~1.3x on average,
    so we use max_tokens / 1.4 as the word budget to stay safely under 512).
    """
    word_budget = int(max_tokens / 1.4)
    windows: list[list[str]] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current and current_words + word_count > word_budget:
            windows.append(current)
            current = []
            current_words = 0
        current.append(sentence)
        current_words += word_count

    if current:
        windows.append(current)

    return windows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(window_results: list[dict]) -> dict:
    """
    Weighted mean of per-window scores, weighted by sentence count.
    Returns the label with the highest aggregate score.
    """
    total_weight = sum(w["sentence_count"] for w in window_results)
    if total_weight == 0:
        return {"label": "neutral", "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}}

    agg: dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for w in window_results:
        weight = w["sentence_count"] / total_weight
        for label, score in w["scores"].items():
            agg[label] += score * weight

    top_label = max(agg, key=lambda k: agg[k])
    return {"label": top_label, "scores": {k: round(v, 4) for k, v in agg.items()}}


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def run_finbert(text: str, chunk_size: int = 400) -> dict:
    """
    Run FinBERT sentiment over arbitrarily long transcript text.

    Args:
        text:       Transcript passage or full transcript body.
        chunk_size: Max tokens per window (≤ 512; default 400 leaves [CLS]/[SEP] headroom).

    Returns:
        dict with 'aggregate' scores, per-'windows' breakdown, and 'window_count'.
    """
    if not text or not text.strip():
        return {
            "aggregate": {"label": "neutral", "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}},
            "windows": [],
            "window_count": 0,
        }

    pipe = _get_pipeline()
    sentences = _split_sentences(text)
    windows = _build_windows(sentences, max_tokens=chunk_size)

    window_results: list[dict] = []
    for window_sentences in windows:
        window_text = " ".join(window_sentences)
        # hf pipeline returns [[{"label": ..., "score": ...}, ...]] when top_k=None
        raw = pipe(window_text)
        label_scores = {item["label"].lower(): round(item["score"], 4) for item in raw[0]}
        top_label = max(label_scores, key=lambda k: label_scores[k])

        window_results.append(
            {
                "text_preview": window_text[:120],
                "label": top_label,
                "scores": label_scores,
                "sentence_count": len(window_sentences),
            }
        )

    aggregate = _aggregate(window_results)

    return {
        "aggregate": aggregate,
        "windows": window_results,
        "window_count": len(window_results),
    }