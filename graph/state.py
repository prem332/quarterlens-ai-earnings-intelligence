"""

Design notes:
- Each agent writes to its own output key — no write contention.
- decision_log_entries uses Annotated[list, operator.add] so every node can
  append audit entries concurrently without clobbering each other.
- Comparison and Sentiment run in parallel (see build_graph.py); their output
  keys (comparison_findings, sentiment_scores) are independent.
"""

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict


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
    report_model_tier: str            # "primary" | "finetuned" — report_agent only; isolates fine-tuned model from other agents

    # ── Agent outputs (one key per agent, no shared keys) ──────────────────
    retrieval_results: list[RetrievalResult]
    comparison_findings: list[ComparisonFinding]
    sentiment_scores: list[SentimentScore]
    numeric_validations: list[NumericValidation]
    report: str                       # final drafted report text

    # ── Audit trail (append-only, reducer handles concurrent writes) ───────
    decision_log_entries: Annotated[list[DecisionLogEntry], operator.add]

    # ── Pipeline control ───────────────────────────────────────────────────
    error: str | None                 # set by any node on unrecoverable failure