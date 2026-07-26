"""
Runs in parallel with comparison_agent (see build_graph.py).

Calls run_finbert() over transcript chunks from transcript_retrieval_results.
FinBERT is deterministic — no LLM involved here per ARCHITECTURE.md §3.
FinBERT inference runs in a thread pool via asyncio.to_thread() so it
doesn't block the event loop during parallel execution with comparison_agent.

Reads from transcript_retrieval_results (not retrieval_results) so FinBERT
receives the full transcript candidate set rather than the globally reranked
top-5 which may be dominated by filing chunks for financial queries.

Tool: run_finbert(text) → {label: str, score: float}
"""

import asyncio
import re
import time
from graph.state import GraphState, DecisionLogEntry, SentimentScore
from tools.run_finbert import run_finbert
from data_pipeline.chunking.sentences import split_sentences


_TRANSCRIPT_DOC_TYPES = {"transcript", "earnings_call"}
_MAX_PASSAGE_CHARS    = 1200

# FinBERT used to score every transcript chunk with no query awareness — the
# highest-confidence passage anywhere in the call would win downstream selection
# even if it was about a completely different topic than what was asked. Restrict
# to the chunks most topically relevant to the query first.
_MAX_CANDIDATES = 8

# Then score at SENTENCE granularity, not whole-chunk. A chunk spans a full
# speaker turn (multiple sentences, up to 1200 chars) — FinBERT scoring the whole
# chunk aggregates sentiment across everything said in that turn, which can differ
# from the specific sentence a claim is actually about (e.g. a turn mixing "great
# quarter overall" with "though gaming revenue declined" can score neutral/positive
# even though the gaming-specific sentence is clearly negative). Split candidate
# chunks into sentences, rank by topical relevance, score the top few individually.
_MAX_SENTENCES = 6
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _topic_overlap(query: str, text: str) -> float:
    """Fraction of query's tokens also present in text — cheap topical relevance."""
    qt = _tokens(query)
    if not qt:
        return 0.0
    return len(qt & _tokens(text)) / len(qt)


async def sentiment_agent(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    t0 = time.time()
    query = state.get("query", "")

    # Read from transcript_retrieval_results — preserved by retrieval_agent
    # before global reranking so FinBERT sees all transcript candidates.
    transcript_chunks = list(state.get("transcript_retrieval_results") or [])

    # Fallback: if transcript_retrieval_results is empty (e.g. older pipeline
    # run without the new field), filter retrieval_results by doc_type.
    if not transcript_chunks:
        retrieval_results = state.get("retrieval_results") or []
        transcript_chunks = [
            r for r in retrieval_results
            if r.get("doc_type", "").lower() in _TRANSCRIPT_DOC_TYPES
        ]

    if not transcript_chunks:
        return _empty("no transcript chunks available", t0)

    # Narrow to the most query-relevant chunks before scoring — see module
    # docstring above. Still scores multiple candidates (not just top-1) so a
    # genuinely mixed-sentiment topic isn't flattened to one passage.
    if query and len(transcript_chunks) > _MAX_CANDIDATES:
        transcript_chunks = sorted(
            transcript_chunks,
            key=lambda c: _topic_overlap(query, c.get("content", "")),
            reverse=True,
        )[:_MAX_CANDIDATES]

    # Sentence-level passages when we have a query to rank against — see module
    # docstring. Falls back to whole chunks (old behavior) if there's no query to
    # rank sentences by, so this degrades gracefully rather than breaking.
    if query:
        candidate_sentences = [
            s for chunk in transcript_chunks
            for s in split_sentences(chunk.get("content", ""))
        ]
        passages = sorted(
            (s for s in candidate_sentences if s.strip()),
            key=lambda s: _topic_overlap(query, s),
            reverse=True,
        )[:_MAX_SENTENCES]
        if not passages:
            passages = [c.get("content", "") for c in transcript_chunks]
    else:
        passages = [c.get("content", "") for c in transcript_chunks]

    async def _score_passage(passage: str) -> SentimentScore | None:
        text = passage[:_MAX_PASSAGE_CHARS]
        if not text.strip():
            return None
        try:
            result = await asyncio.to_thread(run_finbert, text)
            # run_finbert returns {"aggregate": {"label": ..., "scores": {...}}, ...} —
            # label/score live under "aggregate", not at the top level. Reading them
            # at the top level (the previous version) silently defaulted to
            # "neutral"/0.0 on every single call via .get()'s fallback, regardless
            # of what FinBERT actually returned — this was the real cause of the
            # near-universal "neutral" mislabeling, not chunk/sentence selection.
            aggregate = result.get("aggregate", {})
            label = aggregate.get("label", "neutral")
            scores = aggregate.get("scores", {})
            return SentimentScore(
                label=label,
                score=float(scores.get(label, 0.0)),
                passage=text,
            )
        except Exception as exc:
            print(f"[sentiment_agent] run_finbert failed on passage: {exc}")
            return None

    tasks   = [_score_passage(p) for p in passages]
    results = await asyncio.gather(*tasks)
    scores: list[SentimentScore] = [r for r in results if r is not None]

    if scores:
        pos = sum(1 for s in scores if s["label"] == "positive")
        neg = sum(1 for s in scores if s["label"] == "negative")
        neu = len(scores) - pos - neg
        summary = f"{len(scores)} passages scored — pos={pos} neg={neg} neu={neu}"
    else:
        summary = "0 passages scored"

    entry: DecisionLogEntry = {
        "agent":         "sentiment_agent",
        "tool_called":   "run_finbert",
        "input_summary": f"{len(transcript_chunks)} transcript chunks (from transcript_retrieval_results)",
        "output_summary": summary,
        "confidence":    None,
        "tokens_used":   None,
        "latency_ms":    round((time.time() - t0) * 1000, 1),
    }

    return {
        "sentiment_scores":     scores,
        "decision_log_entries": [entry],
    }


def _empty(reason: str, t0: float) -> dict:
    entry: DecisionLogEntry = {
        "agent":         "sentiment_agent",
        "tool_called":   None,
        "input_summary": reason,
        "output_summary": "skipped",
        "confidence":    None,
        "tokens_used":   None,
        "latency_ms":    round((time.time() - t0) * 1000, 1),
    }
    return {
        "sentiment_scores":     [],
        "decision_log_entries": [entry],
    }