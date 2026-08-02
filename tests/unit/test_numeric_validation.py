"""
Unit tests for agents/numeric_validation_agent.py.

Covers the pure deterministic helpers directly (no mocking needed — this is
exactly the "all arithmetic is done by calculate_metric()/these helpers,
never by the LLM" contract the module's own docstring states), plus the
full agent function with calculate_metric()/openai_client.chat() mocked so
no live Azure call happens. Several of these tests pin down specific bugs
documented in the module's own comments (delta-vs-level category error,
oversized-chunk truncation, unverifiable-claim leakage) — regressions of
those, not just generic coverage.
"""
from unittest.mock import patch

from agents.numeric_validation_agent import (
    _rescale_claimed,
    _normalize_period,
    _concat_transcript,
    _compare,
    numeric_validation_agent,
)


# ── _rescale_claimed ────────────────────────────────────────────────────────

def test_rescale_percent_to_percentage_points():
    assert _rescale_claimed(49.3, "percent", "%") == 49.3


def test_rescale_basis_points_to_percentage_points():
    assert _rescale_claimed(340, "basis_points", "pp") == 3.4


def test_rescale_usd_millions_to_raw_usd():
    assert _rescale_claimed(44867, "usd_millions", "USD") == 44867 * 1e6


def test_rescale_usd_billions_to_raw_usd():
    assert _rescale_claimed(95.4, "usd_billions", "USD") == 95.4 * 1e9


def test_rescale_unknown_unit_falls_through_identity():
    # Documented behavior: unknown/absent unit never invents a conversion
    # the model didn't state.
    assert _rescale_claimed(12.3, "widgets", "%") == 12.3


def test_rescale_none_claimed_stays_none():
    assert _rescale_claimed(None, "percent", "%") is None


# ── _normalize_period ───────────────────────────────────────────────────────

def test_normalize_period_valid_quarter():
    assert _normalize_period("Q2_FY2025") == "FY2025-Q2"


def test_normalize_period_unparseable_falls_back_to_input():
    assert _normalize_period("not-a-quarter") == "not-a-quarter"


# ── _concat_transcript ──────────────────────────────────────────────────────

def test_concat_transcript_filters_to_transcript_doc_type():
    chunks = [
        {"doc_type": "10-Q", "content": "filing text"},
        {"doc_type": "transcript", "content": "call text"},
    ]
    assert _concat_transcript(chunks) == "call text"


def test_concat_transcript_skips_not_breaks_on_oversized_chunk():
    # Regression for the documented bug: an oversized chunk used to `break`
    # and discard every chunk behind it. It must now be skipped, letting
    # smaller chunks after it still contribute.
    chunks = [
        {"doc_type": "transcript", "content": "x" * 100},   # over budget, skip
        {"doc_type": "transcript", "content": "small one"},  # fits, keep
    ]
    result = _concat_transcript(chunks, max_chars=50)
    assert result == "small one"


def test_concat_transcript_empty_when_no_transcript_chunks():
    chunks = [{"doc_type": "10-Q", "content": "filing only"}]
    assert _concat_transcript(chunks) == ""


# ── _compare ─────────────────────────────────────────────────────────────

def test_compare_match_within_percentage_tolerance():
    match, delta = _compare(49.3, 49.27, "percentage")
    assert match
    assert delta < 0.5


def test_compare_mismatch_outside_tolerance():
    match, delta = _compare(49.3, 40.0, "percentage")
    assert not match
    assert delta > 0.5


def test_compare_absolute_uses_wider_tolerance():
    # 1% tolerance for absolutes vs 0.5% for percentages.
    match, _ = _compare(100.7, 100.0, "absolute")
    assert match


def test_compare_none_claimed_or_calculated_never_matches():
    assert _compare(None, 100.0, "absolute") == (False, None)
    assert _compare(100.0, None, "absolute") == (False, None)


def test_compare_zero_calculated_value():
    assert _compare(0.0, 0.0, "absolute") == (True, None)
    assert _compare(5.0, 0.0, "absolute") == (False, None)


# ── numeric_validation_agent (full function, mocked boundaries) ────────────

_STATE_BASE = {
    "company": "MSFT",
    "quarter": "FY2026-Q3",
    "retrieval_results": [],
    "transcript_retrieval_results": [
        {"doc_type": "transcript", "content": "Gross margin was 49.3% this quarter."}
    ],
}


def test_agent_returns_empty_when_no_transcript_content():
    state = {**_STATE_BASE, "transcript_retrieval_results": [], "retrieval_results": []}
    result = numeric_validation_agent(state)
    assert result["numeric_validations"] == []


def test_agent_returns_empty_when_upstream_error_set():
    state = {**_STATE_BASE, "error": "retrieval_agent failed"}
    result = numeric_validation_agent(state)
    assert result == {}


def test_agent_matches_claim_against_calculated_value():
    fake_claims = [{
        "claim": "Gross margin was 49.3% this quarter.",
        "metric": "gross_margin",
        "claimed_value": 49.3,
        "claimed_unit": "percent",
        "value_type": "percentage",
    }]
    with patch("agents.numeric_validation_agent._extract_claims", return_value=fake_claims), \
         patch("agents.numeric_validation_agent.calculate_metric",
               return_value={"value": 49.27, "unit": "%"}):
        result = numeric_validation_agent(dict(_STATE_BASE))

    validations = result["numeric_validations"]
    assert len(validations) == 1
    assert validations[0]["match"] is True
    assert validations[0]["metric"] == "gross_margin"


def test_agent_skips_claim_with_no_claimed_value():
    # Guard documented in the module: a claim extracted with no actual
    # number must be skipped before calculate_metric is even called, not
    # reported as a false mismatch.
    fake_claims = [{
        "claim": "you can see that in the OpEx numbers",
        "metric": "operating_expenses",
        "claimed_value": None,
        "claimed_unit": "",
        "value_type": "absolute",
    }]
    with patch("agents.numeric_validation_agent._extract_claims", return_value=fake_claims), \
         patch("agents.numeric_validation_agent.calculate_metric") as mock_calc:
        result = numeric_validation_agent(dict(_STATE_BASE))

    assert result["numeric_validations"] == []
    mock_calc.assert_not_called()


def test_agent_skips_unverifiable_claim():
    fake_claims = [{
        "claim": "Azure grew 30%",
        "metric": "azure_revenue_growth",
        "claimed_value": 30.0,
        "claimed_unit": "percent",
        "value_type": "percentage",
    }]
    with patch("agents.numeric_validation_agent._extract_claims", return_value=fake_claims), \
         patch("agents.numeric_validation_agent.calculate_metric",
               return_value={"value": None, "unit": None, "error": "segment metric not filed"}):
        result = numeric_validation_agent(dict(_STATE_BASE))

    assert result["numeric_validations"] == []


def test_agent_skips_basis_points_vs_level_category_error():
    # Documented deterministic backstop: a basis-point CHANGE claim
    # compared against a filed LEVEL is a category error, not a real
    # mismatch, and must be skipped rather than scored.
    fake_claims = [{
        "claim": "margin improved 110 bps",
        "metric": "gross_margin",
        "claimed_value": 110,
        "claimed_unit": "basis_points",
        "value_type": "percentage",
    }]
    with patch("agents.numeric_validation_agent._extract_claims", return_value=fake_claims), \
         patch("agents.numeric_validation_agent.calculate_metric",
               return_value={"value": 49.27, "unit": "%"}):
        result = numeric_validation_agent(dict(_STATE_BASE))

    assert result["numeric_validations"] == []
