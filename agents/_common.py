"""
agents/_common.py

Shared decision-log helpers for the LangGraph agent nodes.

Every agent emits DecisionLogEntry records into the append-only
`decision_log_entries` channel (see graph/state.py), and every agent has a
degraded path that returns an empty output plus a "skipped" audit entry. Those
three shapes — the latency stamp, the entry itself, and the skip result — were
duplicated across all five agent modules; they live here instead.

Behaviour is unchanged: each agent still defines its own `_empty(reason, t0)`
wrapper, so call sites and the per-agent output key stay exactly as they were.
"""
from __future__ import annotations

import time
from typing import Any

from graph.state import DecisionLogEntry


def ms(t0: float) -> float:
    """Milliseconds elapsed since t0, rounded — the latency stamp on every entry."""
    return round((time.time() - t0) * 1000, 1)


def log_entry(
    agent: str,
    tool_called: str | None,
    input_summary: str,
    output_summary: str,
    confidence: float | None,
    tokens_used: int | None,
    latency_ms: float | None,
) -> DecisionLogEntry:
    """Build one audit-trail entry."""
    return DecisionLogEntry(
        agent=agent,
        tool_called=tool_called,
        input_summary=input_summary,
        output_summary=output_summary,
        confidence=confidence,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
    )


def skipped(agent: str, output_key: str, empty: Any, reason: str, t0: float) -> dict:
    """
    Degraded-path result: the agent's empty output plus a "skipped" audit entry.

    `empty` is passed in rather than assumed: report_agent's empty output is ""
    (a string) while the other agents' are [] (a list).
    """
    return {
        output_key: empty,
        "decision_log_entries": [
            log_entry(agent, None, reason, "skipped", None, None, ms(t0))
        ],
    }
