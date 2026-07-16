"""
observability/decision_log.py
Cosmos DB decision log writer for QuarterLens AI.

Writes per-agent audit records to Cosmos DB for every pipeline run.
This is the regulated-domain audit trail — what each agent decided,
which tools it called, confidence, cost, and latency.

Distinct from Phoenix (live span tracing) and MLflow (experiment metrics).
This layer answers: "For run X, what exactly did the comparison agent do?"

Usage:
    from observability.decision_log import log_agent_decision, log_run_summary

    log_agent_decision(
        run_id="run-123",
        agent="retrieval_agent",
        input_query="What was Apple's revenue growth?",
        retrieved_chunks=[...],
        output="...",
        tool_calls=[{"tool": "search_documents", "args": {...}}],
        confidence=0.91,
        latency_ms=430,
        token_cost={"prompt": 512, "completion": 128},
    )
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from azure_clients.cosmos_client import cosmos_client

log = logging.getLogger(__name__)


def log_agent_decision(
    run_id: str,
    agent: str,
    input_query: str,
    output: str,
    tool_calls: list[dict[str, Any]] | None = None,
    retrieved_chunks: list[str] | None = None,
    confidence: float | None = None,
    latency_ms: int | None = None,
    token_cost: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Write a per-agent decision record to the Cosmos DB Decision Log.

    Args:
        run_id:           Unique identifier for the pipeline run.
        agent:            Agent name (e.g. "retrieval_agent", "numeric_validation_agent").
        input_query:      The query or input this agent received.
        output:           The agent's output or decision.
        tool_calls:       List of tool invocations with args and results.
        retrieved_chunks: Chunk IDs or summaries of retrieved context.
        confidence:       Agent's self-reported confidence score (0–1).
        latency_ms:       Wall-clock time for this agent's execution in ms.
        token_cost:       Dict with "prompt" and "completion" token counts.
        extra:            Any additional agent-specific metadata.
    """
    record = {
        "run_id": run_id,
        "agent": agent,
        "timestamp_ms": int(time.time() * 1000),
        "input_query": input_query,
        "output": output,
        "tool_calls": tool_calls or [],
        "retrieved_chunks": retrieved_chunks or [],
        "confidence": confidence,
        "latency_ms": latency_ms,
        "token_cost": token_cost or {},
        **(extra or {}),
    }

    try:
        cosmos_client.log(run_id=run_id, agent=agent, data=record)
        log.debug(
            "Decision logged: run=%s agent=%s latency=%sms",
            run_id, agent, latency_ms,
        )
    except Exception as exc:
        # Decision log failure must never crash the pipeline.
        log.warning(
            "Failed to write decision log (run=%s agent=%s): %s",
            run_id, agent, exc,
        )


def log_run_summary(
    run_id: str,
    company: str,
    fiscal_label: str,
    total_latency_ms: int,
    total_tokens: dict[str, int],
    agent_sequence: list[str],
    final_verdict: str,
    error: str | None = None,
) -> None:
    """
    Write a run-level summary record after the full pipeline completes.

    Args:
        run_id:            Unique identifier for the pipeline run.
        company:           Ticker symbol (e.g. "AAPL").
        fiscal_label:      Filing label (e.g. "FY2025-Q3").
        total_latency_ms:  Total wall-clock time for the full run.
        total_tokens:      Aggregated token counts across all agents.
        agent_sequence:    Ordered list of agents that executed.
        final_verdict:     Summary outcome (e.g. "verified", "mismatch_found").
        error:             Error message if the run failed, else None.
    """
    record = {
        "run_id": run_id,
        "record_type": "run_summary",
        "company": company,
        "fiscal_label": fiscal_label,
        "timestamp_ms": int(time.time() * 1000),
        "total_latency_ms": total_latency_ms,
        "total_tokens": total_tokens,
        "agent_sequence": agent_sequence,
        "final_verdict": final_verdict,
        "error": error,
        "status": "error" if error else "success",
    }

    try:
        cosmos_client.log(run_id=run_id, agent="run_summary", data=record)
        log.info(
            "Run summary logged: run=%s %s/%s status=%s latency=%dms",
            run_id, company, fiscal_label,
            record["status"], total_latency_ms,
        )
    except Exception as exc:
        log.warning("Failed to write run summary (run=%s): %s", run_id, exc)


def get_run_log(run_id: str) -> list[dict]:
    """
    Retrieve all decision records for a given run_id.

    Args:
        run_id: The run identifier to fetch.

    Returns:
        List of decision records in insertion order.
    """
    try:
        return cosmos_client.get_run_log(run_id)
    except Exception as exc:
        log.warning("Failed to retrieve run log (run=%s): %s", run_id, exc)
        return []