from datetime import datetime
from typing import Any
from pydantic import BaseModel
from .shared import Company, Quarter, RunStatus


class RunStatusResponse(BaseModel):
    run_id: str
    status: RunStatus
    company: Company
    quarter: Quarter
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = None


# These three mirror the TypedDicts in graph/state.py exactly. The graph is the
# source of truth: the pipeline's output is written to blob storage verbatim and
# re-read here, so any field-name drift makes AnalysisResponse(**doc) raise
# ValidationError and 500 the endpoint. Keep them in sync.

class NumericValidation(BaseModel):
    claim: str                          # verbatim claim from the transcript
    metric: str = ""                    # e.g. "revenue_growth_yoy"
    claimed_value: float | None = None  # what the executive said
    calculated_value: float | None = None  # what the filing supports
    match: bool = False
    delta_pct: float | None = None
    source_fiscal_label: str = ""


class ComparisonFinding(BaseModel):
    topic: str
    current_language: str = ""
    prior_language: dict[str, str] = {}   # {fiscal_label: excerpt}
    shift_detected: bool = False
    shift_description: str | None = None


class SentimentScore(BaseModel):
    label: str          # positive / negative / neutral
    score: float
    passage: str = ""   # the text segment FinBERT scored


class AnalysisResponse(BaseModel):
    run_id: str
    company: Company
    quarter: Quarter
    status: RunStatus
    created_at: datetime
    completed_at: datetime | None = None
    report: str | None = None
    numeric_validations: list[NumericValidation] = []
    comparison_findings: list[ComparisonFinding] = []
    sentiment_scores: list[SentimentScore] = []
    retrieval_results: list[dict[str, Any]] = []
    error: str | None = None


class ReportSummary(BaseModel):
    run_id: str
    company: Company
    quarter: Quarter
    status: RunStatus
    created_at: datetime
    report_snippet: str | None = None   # first 200 chars of report
