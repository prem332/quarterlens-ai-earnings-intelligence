"""
Runs in parallel with comparison_agent (see build_graph.py).

Calls run_finbert() over transcript chunks from retrieval_results.
FinBERT is deterministic — no LLM involved here per ARCHITECTURE.md §3.

Tool: run_finbert(text) → {label: str, score: float}
"""

import time
from graph.state import GraphState, DecisionLogEntry, SentimentScore
from tools.run_finbert import run_finbert


# Only score transcript chunks; filing language is legal boilerplate
_TRANSCRIPT_DOC_TYPES = {"transcript", "earnings_call"}

# Cap per-passage length to keep FinBERT inference fast (model max is 512 tokens)
_MAX_PASSAGE_CHARS = 1200


def sentiment_agent(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    t0 = time.time()
    retrieval_results = state.get("retrieval_results") or []

    transcript_chunks = [
        r for r in retrieval_results
        if r.get("doc_type", "").lower() in _TRANSCRIPT_DOC_TYPES
    ]

    if not transcript_chunks:
        return _empty("no transcript chunks in retrieval_results", t0)

    scores: list[SentimentScore] = []

    for chunk in transcript_chunks:
        passage = chunk.get("content", "")[:_MAX_PASSAGE_CHARS]
        if not passage.strip():
            continue
        try:
            result = run_finbert(passage)
            scores.append(SentimentScore(
                label=result.get("label", "neutral"),
                score=float(result.get("score", 0.0)),
                passage=passage,
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"[sentiment_agent] run_finbert failed on chunk: {exc}")

    # Aggregate summary for log
    if scores:
        pos = sum(1 for s in scores if s["label"] == "positive")
        neg = sum(1 for s in scores if s["label"] == "negative")
        neu = len(scores) - pos - neg
        summary = f"{len(scores)} passages scored — pos={pos} neg={neg} neu={neu}"
    else:
        summary = "0 passages scored"

    entry: DecisionLogEntry = {
        "agent": "sentiment_agent",
        "tool_called": "run_finbert",
        "input_summary": f"{len(transcript_chunks)} transcript chunks",
        "output_summary": summary,
        "confidence": None,
        "tokens_used": None,           # FinBERT is not an LLM — no token cost
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }

    return {
        "sentiment_scores": scores,
        "decision_log_entries": [entry],
    }


def _empty(reason: str, t0: float) -> dict:
    entry: DecisionLogEntry = {
        "agent": "sentiment_agent",
        "tool_called": None,
        "input_summary": reason,
        "output_summary": "skipped",
        "confidence": None,
        "tokens_used": None,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
    return {
        "sentiment_scores": [],
        "decision_log_entries": [entry],
    }