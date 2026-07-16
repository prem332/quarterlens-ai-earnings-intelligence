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


class NumericValidation(BaseModel):
    claim: str
    filed_value: float | None
    stated_value: float | None
    verified: bool
    delta_pct: float | None = None


class ComparisonFinding(BaseModel):
    topic: str
    current: str
    prior: str
    quarter: Quarter
    shift_detected: bool


class SentimentScore(BaseModel):
    label: str          # positive / negative / neutral
    score: float
    excerpt: str


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


class ClaimEvidence(BaseModel):
    claim_id: str
    claim_text: str
    source_section: str
    source_paragraph: str
    confidence: float
    doc_type: str       # "10-Q", "10-K", or "transcript"
    quarter: Quarter