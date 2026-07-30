"""

Design notes:
- Each agent writes to its own output key — no write contention.
- decision_log_entries uses Annotated[list, operator.add] so every node can
  append audit entries concurrently without clobbering each other.
- Comparison and Sentiment run in parallel (see build_graph.py); their output
  keys (comparison_findings, sentiment_scores) are independent.
- retrieval_results: globally reranked evidence (filing + transcript) for
  comparison_agent, report_agent, numeric_validation_agent.
- transcript_retrieval_results: raw transcript candidates preserved for
  sentiment_agent — FinBERT needs maximum transcript coverage, not the
  globally reranked top-5 which may be dominated by filing chunks.
"""

import operator
from typing import Annotated
from typing_extensions import NotRequired, TypedDict


class DecisionLogEntry(TypedDict):
    agent: str
    tool_called: str | None
    input_summary: str
    output_summary: str
    confidence: float | None
    tokens_used: int | None
    latency_ms: float | None


class RetrievalResult(TypedDict):
    chunk_id: str
    content: str
    company: str
    quarter: str
    doc_type: str          # "10-Q" | "10-K" | "transcript"
    fiscal_label: str
    score: float
    accession: str         # SEC accession number — filing coordinate for precision/recall@k
    section: str           # parsed section key (e.g. "mda") — filing coordinate for precision/recall@k
    chunk_index: int       # position of chunk within its section (−1 if unknown) — enables adjacency/duplicate analysis
    chunk_total: int       # total chunks in the section (−1 if unknown)
    parent_id: str         # L2 parent block id (hierarchical retrieval) — siblings share it
    parent_index: int      # ordinal of this child within its parent
    parent_total: int      # total children in the parent
    parent_content: str    # parent block reconstructed at retrieval (small-to-big); "" until expanded


class ComparisonFinding(TypedDict):
    topic: str
    current_language: str
    prior_language: dict[str, str]   # {fiscal_label: excerpt}
    shift_detected: bool
    shift_description: str | None


class SentimentScore(TypedDict):
    label: str             # "positive" | "negative" | "neutral"
    score: float           # 0.0–1.0 confidence
    passage: str           # the text segment scored


class NumericValidation(TypedDict):
    claim: str             # verbatim claim from transcript
    metric: str            # e.g. "revenue_growth_yoy"
    claimed_value: float | None
    calculated_value: float | None
    match: bool
    delta_pct: float | None
    source_fiscal_label: str


class GraphState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    company: str                      # e.g. "AAPL"
    quarter: str                      # e.g. "Q2_FY2025"
    query: str                        # analyst's question or "full analysis"
    comparison_quarters: list[str]    # prior quarters to compare against

    # ── Model routing (Phase 2) ────────────────────────────────────────────
    model_tier: str                   # "primary" (gpt-5.4-mini) | "standard" (gpt-5-mini)

    # ── Agent outputs (one key per agent, no shared keys) ──────────────────
    retrieval_results: list[RetrievalResult]             # globally reranked — comparison/report/numeric
    transcript_retrieval_results: list[RetrievalResult]  # transcript candidates — sentiment_agent only
    comparison_findings: list[ComparisonFinding]
    sentiment_scores: list[SentimentScore]
    numeric_validations: list[NumericValidation]
    report: str                       # final drafted report text

    # ── Audit trail (append-only, reducer handles concurrent writes) ───────
    decision_log_entries: Annotated[list[DecisionLogEntry], operator.add]

    # ── Pipeline control ───────────────────────────────────────────────────
    error: str | None                 # set by any node on unrecoverable failure

    # ── Live token streaming (optional) ─────────────────────────────────────
    # An asyncio.Queue, set only by api/routes/analysis.py when a browser is
    # listening on the SSE stream endpoint; report_agent pushes draft/verify
    # progress events to it if present. NotRequired since it's absent on every
    # other invocation path — run_baseline_eval.py, tests, and any future
    # caller that just wants the final result with no live progress.
    # LangGraph validates state against this TypedDict schema and silently
    # drops keys it doesn't recognize, so this must be declared here even
    # though it carries a live object rather than serializable data — an
    # undeclared key was confirmed (empirically) not to survive a single node
    # hop, let alone the five in this graph.
    stream_queue: NotRequired[object]