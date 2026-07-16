"""
Typed claim schema for the hand-verified golden evaluation set.
Discriminated union on `claim_type`. SIX variants, each with its own `edge_case`
enum so evaluation can slice metrics by stressor.

Ground-truth principle: anchors are index-independent (filing coordinates or
transcript coordinates). This schema validates SHAPE only — it cannot and does
not fabricate ground truth. Facts are supplied by the human labeler from real
sources.

"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class ClaimType(str, Enum):
    RETRIEVAL = "retrieval"
    COMPARISON = "comparison"        # merged: expected_shift bool carries +/- class
    NUMERIC = "numeric"
    OUT_OF_SCOPE = "out_of_scope"
    SENTIMENT = "sentiment"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Provenance(str, Enum):
    GOLDEN = "golden"        # hand-verified against a real source
    SYNTHETIC = "synthetic"  # Phase 2: mechanically derived from a golden claim


class SourceType(str, Enum):
    FILING = "filing"
    TRANSCRIPT = "transcript"


class DocType(str, Enum):
    FILING_10Q = "10-Q"
    FILING_10K = "10-K"
    TRANSCRIPT = "transcript"


# Per-variant edge_case enums ------------------------------------------------ #

class RetrievalEdge(str, Enum):
    DIRECT = "direct"
    TERMINOLOGY_MISMATCH = "terminology_mismatch"
    MULTI_SECTION = "multi_section"
    NEAR_DUPLICATE_BOILERPLATE = "near_duplicate_boilerplate"
    CROSS_DOCUMENT = "cross_document"
    TEMPORAL_DISAMBIGUATION = "temporal_disambiguation"


class ComparisonEdge(str, Enum):
    # expected_shift = True cases
    GUIDANCE_LANGUAGE = "guidance_language"
    RISK_FACTOR_ADDED = "risk_factor_added"
    RISK_FACTOR_DROPPED = "risk_factor_dropped"
    HEDGING_SHIFT = "hedging_shift"
    MAGNITUDE_SHIFT = "magnitude_shift"
    # expected_shift = False cases (the honesty backbone — over-flag guard)
    BOILERPLATE_REWORD = "boilerplate_reword"
    REORDERING = "reordering"
    NUMERIC_ONLY_UPDATE = "numeric_only_update"
    SYNONYM_SWAP = "synonym_swap"
    FORMATTING_ARTIFACT = "formatting_artifact"


SHIFT_TRUE_EDGES = {
    ComparisonEdge.GUIDANCE_LANGUAGE, ComparisonEdge.RISK_FACTOR_ADDED,
    ComparisonEdge.RISK_FACTOR_DROPPED, ComparisonEdge.HEDGING_SHIFT,
    ComparisonEdge.MAGNITUDE_SHIFT,
}
SHIFT_FALSE_EDGES = {
    ComparisonEdge.BOILERPLATE_REWORD, ComparisonEdge.REORDERING,
    ComparisonEdge.NUMERIC_ONLY_UPDATE, ComparisonEdge.SYNONYM_SWAP,
    ComparisonEdge.FORMATTING_ARTIFACT,
}


class NumericEdge(str, Enum):
    EXACT_MATCH = "exact_match"
    ROUNDING_BAND = "rounding_band"
    UNIT_SCALE = "unit_scale"
    DERIVED_METRIC = "derived_metric"
    MISMATCH_TRUE = "mismatch_true"
    PERIOD_MISMATCH = "period_mismatch"


class NumericVerdict(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"


class OutOfScopeEdge(str, Enum):
    UNANSWERABLE = "unanswerable"   # right doc retrieved, answer genuinely absent
    ADVICE_BAIT = "advice_bait"     # Buy/Sell/Hold — no doc answers this (ARCH §1)


class ExpectedBehavior(str, Enum):
    REFUSE = "refuse"
    DECLINE_ADVICE = "decline_advice"


class SentimentEdge(str, Enum):
    HEDGED_POSITIVE = "hedged_positive"
    NEGATIVE_IN_POLITE_FRAMING = "negative_in_polite_framing"
    ANALYST_PUSHBACK = "analyst_pushback"
    NEUTRAL_BOILERPLATE = "neutral_boilerplate"
    DOMAIN_INVERSION = "domain_inversion"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# --------------------------------------------------------------------------- #
# Anchors — index-independent coordinates
# --------------------------------------------------------------------------- #

class FilingLocator(BaseModel):
    section: str = Field(
        ...,
        description="Canonical section key emitted by document_parser.py (the "
                    "'section' field). Validated against config/section_ids.json, "
                    "form-scoped, at load time.",
    )
    paragraph_hint: Optional[str] = Field(
        None,
        description="Short verbatim phrase (<=12 words) to relocate the target "
                    "paragraph after re-parse. A pointer, not a quote.",
    )


class TranscriptLocator(BaseModel):
    """
    Transcripts are a single flat `text` blob (per transcript_fetcher.py output) —
    no turn segmentation. Speaker is read off the inline 'Name: ...' prefix, and
    quote_span is the only real locator: text-matched at eval time, not index-
    resolved. A verbatim span is unambiguous within one transcript.
    """
    speaker: str = Field(..., description="Speaker name as it appears inline in the transcript.")
    quote_span: str = Field(
        ...,
        max_length=320,
        description="Verbatim span (<=~40 words) locating the claim. Pointer, not reproduction.",
    )


class FilingAnchor(BaseModel):
    source_type: Literal[SourceType.FILING] = SourceType.FILING
    cik: str
    accession: str
    fiscal_label: str
    form: DocType
    locator: FilingLocator


class TranscriptAnchor(BaseModel):
    source_type: Literal[SourceType.TRANSCRIPT] = SourceType.TRANSCRIPT
    cik: str
    company: str = Field(..., description="Ticker, e.g. AAPL.")
    fiscal_label: str
    locator: TranscriptLocator


SourceAnchor = Annotated[
    Union[FilingAnchor, TranscriptAnchor],
    Field(discriminator="source_type"),
]


# --------------------------------------------------------------------------- #
# Shared envelope
# --------------------------------------------------------------------------- #

class ClaimBase(BaseModel):
    claim_id: str = Field(..., description="Stable unique id, e.g. AAPL_FY2025-Q3_num_001.")
    company: str = Field(..., description="Ticker.")
    fiscal_label: str = Field(..., description="Primary quarter under test.")
    difficulty: Difficulty
    notes: str = Field(..., min_length=1, description="Labeler rationale — why this is ground truth.")
    provenance: Provenance = Provenance.GOLDEN
    derived_from: Optional[str] = Field(
        None,
        description="Phase 2: golden claim_id this synthetic claim derives from. "
                    "Must be None when provenance=golden.",
    )

    @model_validator(mode="after")
    def _provenance_consistency(self):
        if self.provenance == Provenance.GOLDEN and self.derived_from is not None:
            raise ValueError("golden claims must not set derived_from")
        if self.provenance == Provenance.SYNTHETIC and self.derived_from is None:
            raise ValueError("synthetic claims must set derived_from")
        return self


# --------------------------------------------------------------------------- #
# 1 — retrieval
# --------------------------------------------------------------------------- #

class RetrievalPayload(BaseModel):
    query: str
    doc_type: DocType
    expected_answer_gist: str = Field(..., description="What a correct retrieval enables answering.")


class RetrievalGroundTruth(BaseModel):
    relevant_anchors: list[SourceAnchor] = Field(..., min_length=1)
    distractor_note: Optional[str] = Field(
        None, description="Why near-miss sections could mislead retrieval."
    )


class RetrievalClaim(ClaimBase):
    claim_type: Literal[ClaimType.RETRIEVAL] = ClaimType.RETRIEVAL
    edge_case: RetrievalEdge
    payload: RetrievalPayload
    ground_truth: RetrievalGroundTruth

    @model_validator(mode="after")
    def _multi_anchor_edges(self):
        if self.edge_case in (RetrievalEdge.MULTI_SECTION, RetrievalEdge.CROSS_DOCUMENT):
            if len(self.ground_truth.relevant_anchors) < 2:
                raise ValueError(f"{self.edge_case.value} requires >=2 relevant_anchors")
        if self.edge_case == RetrievalEdge.CROSS_DOCUMENT:
            kinds = {a.source_type for a in self.ground_truth.relevant_anchors}
            if len(kinds) < 2:
                raise ValueError(
                    "cross_document requires anchors spanning both a filing and a transcript"
                )
        return self


# --------------------------------------------------------------------------- #
# 2 — comparison  (merged: expected_shift carries the +/- class)
# --------------------------------------------------------------------------- #

class ComparisonPayload(BaseModel):
    current_quarter_lang: str
    prior_quarter_lang: str
    shift_description: Optional[str] = Field(
        None, description="What substantively changed. Required when expected_shift=True."
    )
    why_not: Optional[str] = Field(
        None, description="Why this is NOT a substantive shift. Required when expected_shift=False."
    )


class ComparisonGroundTruth(BaseModel):
    current_anchor: SourceAnchor
    prior_anchor: SourceAnchor
    expected_shift: bool = Field(
        ...,
        description="True = real substantive language shift. False = looks like a "
                    "shift but isn't (over-flag guard). The false class must stay "
                    "near-balanced with the true class or the shift metric is "
                    "unfalsifiable.",
    )


class ComparisonClaim(ClaimBase):
    claim_type: Literal[ClaimType.COMPARISON] = ClaimType.COMPARISON
    edge_case: ComparisonEdge
    payload: ComparisonPayload
    ground_truth: ComparisonGroundTruth

    @model_validator(mode="after")
    def _edge_matches_expected_shift(self):
        shift = self.ground_truth.expected_shift
        if shift and self.edge_case not in SHIFT_TRUE_EDGES:
            raise ValueError(
                f"edge_case '{self.edge_case.value}' is a no-shift case but "
                f"expected_shift=True"
            )
        if not shift and self.edge_case not in SHIFT_FALSE_EDGES:
            raise ValueError(
                f"edge_case '{self.edge_case.value}' is a real-shift case but "
                f"expected_shift=False"
            )
        if shift and not self.payload.shift_description:
            raise ValueError("expected_shift=True requires payload.shift_description")
        if not shift and not self.payload.why_not:
            raise ValueError("expected_shift=False requires payload.why_not")
        return self


# --------------------------------------------------------------------------- #
# 3 — numeric  (zero-tolerance; XBRL-sourced, never a filing section)
# --------------------------------------------------------------------------- #

class NumericPayload(BaseModel):
    verbal_claim: str = Field(..., description="What the exec stated (verbatim-ish).")
    metric: str = Field(..., description="e.g. 'total net sales', 'gross margin %'.")
    stated_value: str = Field(..., description="Value as stated; string preserves '~18%'.")


class NumericSource(BaseModel):
    xbrl_tag: Optional[str] = Field(
        None, description="us-gaap fact tag when the filed value comes from XBRL."
    )
    accession: str
    detail: Optional[str] = Field(None, description="Extra locating detail for the filed value.")


class NumericGroundTruth(BaseModel):
    filed_value: str = Field(..., description="Real filed value. Human-verified.")
    unit: str = Field(..., description="e.g. 'USD', 'USD millions', 'percent'.")
    source: NumericSource
    verdict: NumericVerdict
    tolerance_rule: str = Field(
        ...,
        description="Per-claim, never global. 'exact' for reported figures; an "
                    "explicit band ONLY where the exec verbally rounds, e.g. "
                    "'abs<=0.5pp; exec said ~18%'.",
    )
    derived_inputs: Optional[list[str]] = Field(
        None,
        description="For derived_metric: the real filed values recomputed from. "
                    "Makes the arithmetic re-checkable.",
    )
    transcript_anchor: Optional[TranscriptAnchor] = Field(
        None, description="Where the exec stated it. Optional but strongly encouraged."
    )


class NumericClaim(ClaimBase):
    claim_type: Literal[ClaimType.NUMERIC] = ClaimType.NUMERIC
    edge_case: NumericEdge
    payload: NumericPayload
    ground_truth: NumericGroundTruth

    @model_validator(mode="after")
    def _edge_verdict_consistency(self):
        gt = self.ground_truth
        if self.edge_case == NumericEdge.MISMATCH_TRUE and gt.verdict != NumericVerdict.MISMATCH:
            raise ValueError("mismatch_true must have verdict=mismatch")
        if self.edge_case == NumericEdge.EXACT_MATCH and gt.verdict != NumericVerdict.MATCH:
            raise ValueError("exact_match must have verdict=match")
        if self.edge_case == NumericEdge.DERIVED_METRIC and not gt.derived_inputs:
            raise ValueError("derived_metric must supply derived_inputs")
        if self.edge_case == NumericEdge.EXACT_MATCH and gt.tolerance_rule.strip().lower() != "exact":
            raise ValueError("exact_match must use tolerance_rule='exact'")
        return self


# --------------------------------------------------------------------------- #
# 4 — out_of_scope  (conditional anchor)
# --------------------------------------------------------------------------- #

class OutOfScopePayload(BaseModel):
    query: str


class OutOfScopeGroundTruth(BaseModel):
    expected_behavior: ExpectedBehavior
    refusal_reason: str = Field(..., min_length=1, description="Why the system must decline.")
    anchor: Optional[SourceAnchor] = Field(
        None,
        description="REQUIRED for unanswerable: the plausible-but-insufficient source "
                    "the system should retrieve and still refuse from — this is what "
                    "distinguishes 'refused correctly' from 'refused because retrieval "
                    "failed'. MUST be None for advice_bait: no document answers "
                    "'should I buy NVDA?'.",
    )


class OutOfScopeClaim(ClaimBase):
    claim_type: Literal[ClaimType.OUT_OF_SCOPE] = ClaimType.OUT_OF_SCOPE
    edge_case: OutOfScopeEdge
    payload: OutOfScopePayload
    ground_truth: OutOfScopeGroundTruth

    @model_validator(mode="after")
    def _anchor_conditionality(self):
        gt = self.ground_truth
        if self.edge_case == OutOfScopeEdge.UNANSWERABLE:
            if gt.anchor is None:
                raise ValueError(
                    "unanswerable requires an anchor (the insufficient source it must refuse from)"
                )
            if gt.expected_behavior != ExpectedBehavior.REFUSE:
                raise ValueError("unanswerable must have expected_behavior=refuse")
        if self.edge_case == OutOfScopeEdge.ADVICE_BAIT:
            if gt.anchor is not None:
                raise ValueError("advice_bait must not carry an anchor — no document answers it")
            if gt.expected_behavior != ExpectedBehavior.DECLINE_ADVICE:
                raise ValueError("advice_bait must have expected_behavior=decline_advice")
        return self


# --------------------------------------------------------------------------- #
# 5 — sentiment  (transcript-anchored, three-class)
# --------------------------------------------------------------------------- #

class SentimentPayload(BaseModel):
    span: str = Field(
        ..., max_length=320,
        description="Verbatim transcript span (<=~40 words) under test.",
    )
    speaker: str


class SentimentGroundTruth(BaseModel):
    label: SentimentLabel = Field(
        ...,
        description="Three-class, matching FinBERT's native output space. A human "
                    "can defensibly assign a class; a human cannot defensibly assign "
                    "a score band — labeling a number would be fabricated precision.",
    )
    rationale: str = Field(
        ..., min_length=1,
        description="REQUIRED. Hand-labeled tone is the most subjective ground truth "
                    "in the set; without written justification it's an assertion, not "
                    "evidence. Also makes disagreement auditable.",
    )
    anchor: TranscriptAnchor


class SentimentClaim(ClaimBase):
    claim_type: Literal[ClaimType.SENTIMENT] = ClaimType.SENTIMENT
    edge_case: SentimentEdge
    payload: SentimentPayload
    ground_truth: SentimentGroundTruth

    @model_validator(mode="after")
    def _boilerplate_is_neutral(self):
        if (self.edge_case == SentimentEdge.NEUTRAL_BOILERPLATE
                and self.ground_truth.label != SentimentLabel.NEUTRAL):
            raise ValueError("neutral_boilerplate must be labeled neutral")
        return self


# --------------------------------------------------------------------------- #
# Discriminated union + collection
# --------------------------------------------------------------------------- #

Claim = Annotated[
    Union[
        RetrievalClaim,
        ComparisonClaim,
        NumericClaim,
        OutOfScopeClaim,
        SentimentClaim,
    ],
    Field(discriminator="claim_type"),
]


class GoldenSet(BaseModel):
    """Top-level container for validating a claims/*.json file."""
    claims: list[Claim]