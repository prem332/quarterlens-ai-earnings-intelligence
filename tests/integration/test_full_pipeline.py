"""
Integration test for the compiled LangGraph pipeline (graph/build_graph.py).

Each agent's own internal logic (LLM calls, retrieval, FinBERT, SQL) is
already covered by tests/unit/*.py with that agent's own boundaries mocked.
This file tests a different thing: the GRAPH WIRING itself — that
supervisor_init's routing, the retrieval -> [comparison || sentiment ||
numeric_validation] fan-out/fan-in, and the decision_log_entries reducer
all behave correctly when real agent functions run end to end. So agent
functions here are replaced wholesale with fakes matching their real
return-shape contract, not partially mocked at the tool level — that
keeps this test about the graph, not a duplicate of the unit tests.

Patches target graph.build_graph's own names (`graph.build_graph.retrieval_agent`,
etc.), not agents.*'s — build_graph.py imports each agent function at module
level and closes over that reference inside build_graph(), so patching the
origin module after graph.build_graph has already been imported would not
affect a freshly-built graph. build_graph() is called fresh inside each test
(not the module-level `compiled_graph` singleton) so the patches actually
take effect.
"""
import asyncio
from unittest.mock import patch

from graph.build_graph import build_graph
from graph.state import GraphState


def _initial_state(**overrides) -> GraphState:
    base: GraphState = {
        "company": "MSFT",
        "quarter": "FY2026-Q3",
        "query": "Summarize key earnings findings and verify management claims.",
        "comparison_quarters": [],
        "retrieval_results": [],
        "transcript_retrieval_results": [],
        "comparison_findings": [],
        "sentiment_scores": [],
        "numeric_validations": [],
        "report": "",
        "decision_log_entries": [],
        "model_tier": "primary",
        "error": None,
    }
    base.update(overrides)
    return base


async def _fake_retrieval_agent(state):
    return {
        "retrieval_results": [{"chunk_id": "c1", "content": "Revenue grew 12% YoY."}],
        "transcript_retrieval_results": [{"chunk_id": "t1", "doc_type": "transcript",
                                           "content": "We are pleased with results."}],
        "decision_log_entries": [{"agent": "retrieval_agent", "tool_called": "search_documents",
                                   "input_summary": "x", "output_summary": "1 chunk",
                                   "confidence": None, "tokens_used": None, "latency_ms": 1.0}],
    }


async def _fake_comparison_agent(state):
    return {
        "comparison_findings": [{"topic": "revenue", "current_language": "grew 12%",
                                  "prior_language": {}, "shift_detected": False,
                                  "shift_description": None}],
        "decision_log_entries": [{"agent": "comparison_agent", "tool_called": None,
                                   "input_summary": "x", "output_summary": "1 finding",
                                   "confidence": None, "tokens_used": None, "latency_ms": 1.0}],
    }


async def _fake_sentiment_agent(state):
    return {
        "sentiment_scores": [{"label": "positive", "score": 0.9, "passage": "pleased with results"}],
        "decision_log_entries": [{"agent": "sentiment_agent", "tool_called": "run_finbert",
                                   "input_summary": "x", "output_summary": "1 scored",
                                   "confidence": None, "tokens_used": None, "latency_ms": 1.0}],
    }


def _fake_numeric_validation_agent(state):
    # Real numeric_validation_agent is sync — LangGraph supports mixed
    # sync/async nodes in one graph, and this fake must match that.
    return {
        "numeric_validations": [{"claim": "revenue grew 12%", "metric": "revenue_growth_yoy",
                                  "claimed_value": 12.0, "calculated_value": 12.0,
                                  "match": True, "delta_pct": 0.0, "source_fiscal_label": "FY2026-Q3"}],
        "decision_log_entries": [{"agent": "numeric_validation_agent", "tool_called": "calculate_metric",
                                   "input_summary": "x", "output_summary": "1 validated",
                                   "confidence": 1.0, "tokens_used": 0, "latency_ms": 1.0}],
    }


async def _fake_report_agent(state):
    return {
        "report": "# Report\nRevenue grew 12% YoY, consistent with management commentary.",
        "decision_log_entries": [{"agent": "report_agent", "tool_called": None,
                                   "input_summary": "x", "output_summary": "drafted",
                                   "confidence": None, "tokens_used": 100, "latency_ms": 1.0}],
    }


_PATCHES = {
    "graph.build_graph.retrieval_agent": _fake_retrieval_agent,
    "graph.build_graph.comparison_agent": _fake_comparison_agent,
    "graph.build_graph.sentiment_agent": _fake_sentiment_agent,
    "graph.build_graph.numeric_validation_agent": _fake_numeric_validation_agent,
    "graph.build_graph.report_agent": _fake_report_agent,
}


def _build_graph_with_fakes():
    patchers = [patch(target, new=fn) for target, fn in _PATCHES.items()]
    for p in patchers:
        p.start()
    try:
        return build_graph()
    finally:
        for p in patchers:
            p.stop()


# ── Structural: the graph compiles with the expected nodes ─────────────────

def test_build_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_build_graph_has_all_expected_nodes():
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "supervisor_init", "retrieval_agent", "comparison_agent", "sentiment_agent",
        "numeric_validation_agent", "report_agent", "supervisor_finalize", "error_exit",
    }
    assert expected.issubset(node_names)


# ── Happy path: fan-out/fan-in + reducer wiring ─────────────────────────────

def test_full_pipeline_happy_path():
    graph = _build_graph_with_fakes()
    result = asyncio.run(graph.ainvoke(_initial_state()))

    assert result["error"] is None
    assert result["report"].startswith("# Report")
    assert len(result["retrieval_results"]) == 1
    assert len(result["transcript_retrieval_results"]) == 1
    assert len(result["comparison_findings"]) == 1
    assert len(result["sentiment_scores"]) == 1
    assert len(result["numeric_validations"]) == 1

    # decision_log_entries uses Annotated[list, operator.add] (graph/state.py)
    # so every node's entry should have accumulated, not overwritten one
    # another — this is the actual wiring behavior under test.
    agents_logged = {e["agent"] for e in result["decision_log_entries"]}
    assert agents_logged == {
        "supervisor_init", "retrieval_agent", "comparison_agent", "sentiment_agent",
        "numeric_validation_agent", "report_agent", "supervisor_finalize",
    }


def test_full_pipeline_parallel_agents_all_see_retrieval_output():
    # comparison_agent/sentiment_agent/numeric_validation_agent all fan out
    # from retrieval_agent directly (not from each other) — verify each
    # fake, which only returns data when given a real state dict, actually
    # ran (rather than one being silently skipped by a wiring mistake).
    graph = _build_graph_with_fakes()
    result = asyncio.run(graph.ainvoke(_initial_state()))

    assert result["comparison_findings"][0]["topic"] == "revenue"
    assert result["sentiment_scores"][0]["label"] == "positive"
    assert result["numeric_validations"][0]["match"] is True


# ── Error path: supervisor_init -> error_exit -> supervisor_finalize ───────

def test_full_pipeline_missing_required_field_routes_to_error_exit():
    graph = _build_graph_with_fakes()
    incomplete_state = _initial_state(company="")  # required field blanked
    result = asyncio.run(graph.ainvoke(incomplete_state))

    assert result["error"] is not None
    assert "company" in result["error"]
    # Downstream agents must never have run.
    assert result["report"] == ""
    assert result["retrieval_results"] == []

    agents_logged = {e["agent"] for e in result["decision_log_entries"]}
    assert "supervisor_init" in agents_logged
    assert "supervisor_finalize" in agents_logged
    assert "retrieval_agent" not in agents_logged
    assert "report_agent" not in agents_logged
