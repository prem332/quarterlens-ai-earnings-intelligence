"""
Unit tests for agents/router.py, agents/comparison_agent.py, and
agents/sentiment_agent.py.

Pure routing/context-building logic is tested directly. The two async
agent functions are tested end-to-end with their LLM/model boundaries
mocked (openai_client.achat_tiered, fetch_prior_quarter, run_finbert) so
no live Azure/model call happens.
"""
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from agents.router import classify_query
from agents.comparison_agent import comparison_agent, _ranked_context, _resolve_quarters_back
from agents.sentiment_agent import sentiment_agent, _topic_overlap


# ── router.classify_query ────────────────────────────────────────────────

def test_classify_query_simple_fact_lookup_is_standard():
    assert classify_query("What was revenue in Q3?") == "standard"


def test_classify_query_comparison_is_primary():
    assert classify_query("What was revenue and how did guidance change?") == "primary"


def test_classify_query_empty_defaults_primary():
    assert classify_query("") == "primary"
    assert classify_query("   ") == "primary"


def test_classify_query_ambiguous_defaults_primary():
    assert classify_query("Tell me about the quarter.") == "primary"


def test_classify_query_primary_override_wins_over_standard_pattern():
    # Contains "what was" (standard) AND "guidance" (primary override) —
    # override must win per the documented precedence.
    assert classify_query("What was the guidance for next quarter?") == "primary"


# ── comparison_agent._ranked_context ────────────────────────────────────────

def test_ranked_context_prefers_parent_content():
    chunks = [{"parent_content": "expanded parent block", "content": "raw chunk"}]
    assert _ranked_context(chunks) == "expanded parent block"


def test_ranked_context_falls_back_to_content():
    chunks = [{"content": "raw chunk text"}]
    assert _ranked_context(chunks) == "raw chunk text"


def test_ranked_context_skips_not_breaks_on_oversized_chunk():
    # Same documented bug class as numeric_validation_agent's _concat_transcript.
    chunks = [
        {"content": "x" * 100},       # over budget, skip
        {"content": "small chunk"},   # fits, keep
    ]
    result = _ranked_context(chunks, max_chars=50)
    assert result == "small chunk"


def test_ranked_context_preserves_input_order():
    chunks = [{"content": "first"}, {"content": "second"}]
    assert _ranked_context(chunks) == "first\n\nsecond"


# ── comparison_agent._resolve_quarters_back ─────────────────────────────────

def test_resolve_quarters_back_yoy():
    result = _resolve_quarters_back("FY2026-Q3", ["FY2025-Q3"])
    assert result["FY2025-Q3"] == 4


def test_resolve_quarters_back_qoq():
    result = _resolve_quarters_back("FY2026-Q3", ["FY2026-Q2"])
    assert result["FY2026-Q2"] == 1


def test_resolve_quarters_back_falls_back_to_position_on_unparseable():
    result = _resolve_quarters_back("not-a-quarter", ["also-not-a-quarter"])
    assert result["also-not-a-quarter"] == 1


# ── comparison_agent (full async function, mocked LLM/tool boundaries) ─────

_COMPARISON_STATE_BASE = {
    "company": "MSFT",
    "quarter": "FY2026-Q3",
    "query": "How did cloud margin commentary change?",
    "comparison_quarters": ["FY2026-Q2"],
    "retrieval_results": [{"content": "Cloud margin was strong this quarter."}],
}


def _mock_chat_response(content: dict) -> MagicMock:
    import json
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=json.dumps(content)))]
    resp.usage = MagicMock(total_tokens=42)
    return resp


def test_comparison_agent_returns_empty_without_retrieval_results():
    state = {**_COMPARISON_STATE_BASE, "retrieval_results": []}
    result = asyncio.run(comparison_agent(state))
    assert result["comparison_findings"] == []


def test_comparison_agent_returns_empty_without_comparison_quarters():
    state = {**_COMPARISON_STATE_BASE, "comparison_quarters": []}
    result = asyncio.run(comparison_agent(state))
    assert result["comparison_findings"] == []


def test_comparison_agent_returns_empty_on_upstream_error():
    state = {**_COMPARISON_STATE_BASE, "error": "retrieval_agent failed"}
    result = asyncio.run(comparison_agent(state))
    assert result == {}


def test_comparison_agent_detects_shift_end_to_end():
    extract_response = _mock_chat_response({
        "topic": "cloud margin",
        "current_statement": "Cloud margin was strong this quarter.",
        "prior_statements": {"FY2026-Q2": "Cloud margin was under pressure last quarter."},
    })
    compare_response = _mock_chat_response({
        "shift_detected": True,
        "shift_description": "Margin characterization improved from pressured to strong.",
    })

    with patch("agents.comparison_agent.fetch_prior_quarter",
               return_value={"results": [{"content": "Cloud margin was under pressure last quarter."}]}), \
         patch.object(
             __import__("agents.comparison_agent", fromlist=["openai_client"]).openai_client,
             "achat_tiered", new=AsyncMock(side_effect=[extract_response, compare_response]),
         ):
        result = asyncio.run(comparison_agent(dict(_COMPARISON_STATE_BASE)))

    findings = result["comparison_findings"]
    assert len(findings) == 1
    assert findings[0]["shift_detected"] is True
    assert findings[0]["topic"] == "cloud margin"


def test_comparison_agent_no_topic_match_returns_empty():
    extract_response = _mock_chat_response({
        "topic": "",
        "current_statement": None,
        "prior_statements": {},
    })

    with patch("agents.comparison_agent.fetch_prior_quarter",
               return_value={"results": [{"content": "unrelated prior text"}]}), \
         patch.object(
             __import__("agents.comparison_agent", fromlist=["openai_client"]).openai_client,
             "achat_tiered", new=AsyncMock(return_value=extract_response),
         ):
        result = asyncio.run(comparison_agent(dict(_COMPARISON_STATE_BASE)))

    assert result["comparison_findings"] == []


# ── sentiment_agent._topic_overlap ──────────────────────────────────────────

def test_topic_overlap_full_match():
    assert _topic_overlap("cloud margin", "cloud margin was strong") == 1.0


def test_topic_overlap_no_match():
    assert _topic_overlap("cloud margin", "iphone sales grew") == 0.0


def test_topic_overlap_empty_query_returns_zero():
    assert _topic_overlap("", "any text") == 0.0


def test_topic_overlap_partial_match():
    # "cloud" matches, "margin" and "growth" don't — 1/3.
    assert _topic_overlap("cloud margin growth", "cloud revenue was strong") == pytest.approx(1 / 3)


# ── sentiment_agent (full async function, mocked run_finbert) ──────────────

def test_sentiment_agent_returns_empty_without_transcript_chunks():
    state = {"query": "How positive was leadership?", "transcript_retrieval_results": [],
             "retrieval_results": []}
    result = asyncio.run(sentiment_agent(state))
    assert result["sentiment_scores"] == []


def test_sentiment_agent_returns_empty_on_upstream_error():
    state = {"error": "retrieval_agent failed"}
    result = asyncio.run(sentiment_agent(state))
    assert result == {}


def test_sentiment_agent_scores_transcript_passages():
    state = {
        "query": "How positive was leadership about growth?",
        "transcript_retrieval_results": [
            {"doc_type": "transcript", "content": "We are thrilled about our growth this quarter."}
        ],
        "retrieval_results": [],
    }
    fake_finbert_result = {
        "aggregate": {"label": "positive", "scores": {"positive": 0.95, "negative": 0.02, "neutral": 0.03}}
    }
    with patch("agents.sentiment_agent.run_finbert", return_value=fake_finbert_result):
        result = asyncio.run(sentiment_agent(state))

    scores = result["sentiment_scores"]
    assert len(scores) >= 1
    assert scores[0]["label"] == "positive"
    assert scores[0]["score"] == 0.95


def test_sentiment_agent_falls_back_to_retrieval_results_when_transcript_pool_empty():
    state = {
        "query": "sentiment check",
        "transcript_retrieval_results": [],
        "retrieval_results": [
            {"doc_type": "transcript", "content": "Solid quarter overall."},
            {"doc_type": "10-Q", "content": "filing text, not a transcript"},
        ],
    }
    fake_finbert_result = {
        "aggregate": {"label": "neutral", "scores": {"positive": 0.3, "negative": 0.2, "neutral": 0.5}}
    }
    with patch("agents.sentiment_agent.run_finbert", return_value=fake_finbert_result):
        result = asyncio.run(sentiment_agent(state))

    assert len(result["sentiment_scores"]) >= 1
