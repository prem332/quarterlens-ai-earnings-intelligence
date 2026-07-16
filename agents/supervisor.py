"""
Supervisor node for QuarterLens LangGraph pipeline.

routing is a static DAG:
    supervisor_init
        → retrieval_agent
        → [comparison_agent ‖ sentiment_agent]   (parallel fan-out)
        → numeric_validation_agent
        → report_agent
        → supervisor_finalize

The supervisor owns two nodes:
  - supervisor_init: validates input, sets defaults, writes initial log entry
  - supervisor_finalize: checks for errors, writes Decision Log to Cosmos DB
"""

import time
import uuid
from graph.state import GraphState, DecisionLogEntry
from azure_clients.cosmos_client import cosmos_decision_log


def supervisor_init(state: GraphState) -> dict:
    """Entry node. Validates required fields and initialises defaults."""
    t0 = time.time()

    missing = [f for f in ("company", "quarter", "query") if not state.get(f)]
    if missing:
        return {
            "error": f"Missing required input fields: {missing}",
            "decision_log_entries": [_log_entry("supervisor_init", None,
                f"input={state.get('company')}/{state.get('quarter')}",
                f"FAILED — missing {missing}", None, None, _ms(t0))],
        }

    defaults: dict = {}
    if not state.get("comparison_quarters"):
        defaults["comparison_quarters"] = []
    if not state.get("retrieval_results"):
        defaults["retrieval_results"] = []
    if not state.get("comparison_findings"):
        defaults["comparison_findings"] = []
    if not state.get("sentiment_scores"):
        defaults["sentiment_scores"] = []
    if not state.get("numeric_validations"):
        defaults["numeric_validations"] = []
    if not state.get("report"):
        defaults["report"] = ""
    if state.get("error") is None:
        defaults["error"] = None

    entry: DecisionLogEntry = _log_entry(
        agent="supervisor_init",
        tool_called=None,
        input_summary=f"company={state['company']} quarter={state['quarter']} query_len={len(state['query'])}",
        output_summary="pipeline initialised",
        confidence=None,
        tokens_used=None,
        latency_ms=_ms(t0),
    )

    return {**defaults, "decision_log_entries": [entry]}


def supervisor_finalize(state: GraphState) -> dict:
    """
    Exit node. Persists the full decision log to Cosmos DB.
    Returns state unchanged (mutations already committed by prior nodes).
    """
    t0 = time.time()

    if state.get("error"):
        summary = f"pipeline FAILED: {state['error']}"
        status = "error"
    else:
        summary = (
            f"pipeline complete — "
            f"{len(state.get('retrieval_results', []))} chunks retrieved, "
            f"{len(state.get('comparison_findings', []))} comparisons, "
            f"{len(state.get('numeric_validations', []))} validations, "
            f"report_len={len(state.get('report', ''))}"
        )
        status = "success"

    final_entry: DecisionLogEntry = _log_entry(
        agent="supervisor_finalize",
        tool_called=None,
        input_summary=f"company={state['company']} quarter={state['quarter']}",
        output_summary=summary,
        confidence=None,
        tokens_used=None,
        latency_ms=_ms(t0),
    )

    # Persist audit trail to Cosmos DB using the correct log() method.
    # fire-and-forget — failure is logged, not fatal.
    try:
        run_id = f"{state['company']}_{state['quarter']}_{int(time.time())}"
        cosmos_decision_log.log(
            run_id=run_id,
            agent="supervisor_finalize",
            tool_called="pipeline_complete",
            result_summary=summary,
            status=status,
            tool_args={
                "company": state["company"],
                "quarter": state["quarter"],
                "query": state.get("query", ""),
            },
        )
    except Exception as exc:
        print(f"[supervisor_finalize] Cosmos write failed (non-fatal): {exc}")

    return {"decision_log_entries": [final_entry]}


# ── Routing helper (used by build_graph.py conditional edge) ──────────────────

def route_after_init(state: GraphState) -> str:
    """After init, go to retrieval unless there's a hard error."""
    return "error_exit" if state.get("error") else "retrieval_agent"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_entry(
    agent: str,
    tool_called: str | None,
    input_summary: str,
    output_summary: str,
    confidence: float | None,
    tokens_used: int | None,
    latency_ms: float | None,
) -> DecisionLogEntry:
    return DecisionLogEntry(
        agent=agent,
        tool_called=tool_called,
        input_summary=input_summary,
        output_summary=output_summary,
        confidence=confidence,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
    )


def _ms(t0: float) -> float:
    return round((time.time() - t0) * 1000, 1)