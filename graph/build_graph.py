"""
pipeline (static DAG):

    supervisor_init
        ↓
    retrieval_agent
        ↓  (fan-out — both run in parallel)
    [comparison_agent, sentiment_agent]
        ↓  (fan-in — both must complete before proceeding)
    numeric_validation_agent
        ↓
    report_agent
        ↓
    supervisor_finalize

Parallel execution is LangGraph's native behaviour when two nodes are both
reachable from the same predecessor without a conditional edge between them.

The error_exit node is a no-op sink — it reaches supervisor_finalize so the
audit trail is still persisted even on failure.

"""

import inspect
import logging
import sys
import time

from langgraph.graph import StateGraph, END

from graph.state import GraphState
from agents.supervisor import supervisor_init, supervisor_finalize, route_after_init
from agents.retrieval_agent import retrieval_agent
from agents.comparison_agent import comparison_agent
from agents.sentiment_agent import sentiment_agent
from agents.numeric_validation_agent import numeric_validation_agent
from agents.report_agent import report_agent

logger = logging.getLogger(__name__)


def _emit(state, event: dict) -> None:
    """Push a stage event to the SSE queue if a browser is listening.

    Best-effort by design: this is progress telemetry, and a failure here must
    never take down an analysis run. put_nowait (not await) so the sync-node
    branch below can call it too -- LangGraph may execute sync nodes off the
    event loop thread, where awaiting the queue is not available.
    """
    queue = state.get("stream_queue") if isinstance(state, dict) else None
    if queue is None:
        return
    try:
        queue.put_nowait(event)
    except Exception:
        pass


def _traced(name, fn):
    """Wrap a node with entry/exit timing.

    Two outputs, same instrumentation:
      * stdout NODE_START/NODE_END lines -- readable via
        `az containerapp logs show`, which is the only stack-level visibility
        available on Container Apps' Consumption tier (it does not grant
        CAP_SYS_PTRACE, confirmed by a real "Permission Denied" from py-spy).
      * stage_start/stage_end events on the SSE queue, so the UI can show
        which stage is ACTUALLY running. The progress list used to advance on
        hardcoded per-stage seconds, which meant it reported a stage as
        finished while it was still running -- and hid, for example, a
        numeric_validation step that really took 53s against a 3s guess.

    Pure observability: does not touch any agent's logic, inputs, or outputs.
    """
    is_async = inspect.iscoroutinefunction(fn)

    if is_async:
        async def wrapped(state):
            t0 = time.time()
            print(f"NODE_START {name} t=0.0s", flush=True)
            sys.stdout.flush()
            _emit(state, {"type": "stage_start", "stage": name})
            try:
                result = await fn(state)
                elapsed = time.time() - t0
                print(f"NODE_END {name} t={elapsed:.1f}s", flush=True)
                _emit(state, {"type": "stage_end", "stage": name, "seconds": round(elapsed, 1)})
                return result
            except Exception:
                elapsed = time.time() - t0
                print(f"NODE_ERROR {name} t={elapsed:.1f}s", flush=True)
                _emit(state, {"type": "stage_error", "stage": name, "seconds": round(elapsed, 1)})
                raise
        return wrapped
    else:
        def wrapped(state):
            t0 = time.time()
            print(f"NODE_START {name} t=0.0s", flush=True)
            sys.stdout.flush()
            _emit(state, {"type": "stage_start", "stage": name})
            try:
                result = fn(state)
                elapsed = time.time() - t0
                print(f"NODE_END {name} t={elapsed:.1f}s", flush=True)
                _emit(state, {"type": "stage_end", "stage": name, "seconds": round(elapsed, 1)})
                return result
            except Exception:
                elapsed = time.time() - t0
                print(f"NODE_ERROR {name} t={elapsed:.1f}s", flush=True)
                _emit(state, {"type": "stage_error", "stage": name, "seconds": round(elapsed, 1)})
                raise
        return wrapped


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node("supervisor_init", _traced("supervisor_init", supervisor_init))
    graph.add_node("retrieval_agent", _traced("retrieval_agent", retrieval_agent))
    graph.add_node("comparison_agent", _traced("comparison_agent", comparison_agent))
    graph.add_node("sentiment_agent", _traced("sentiment_agent", sentiment_agent))
    graph.add_node("numeric_validation_agent", _traced("numeric_validation_agent", numeric_validation_agent))
    graph.add_node("report_agent", _traced("report_agent", report_agent))
    graph.add_node("supervisor_finalize", _traced("supervisor_finalize", supervisor_finalize))
    graph.add_node("error_exit", _error_exit)

    # ── Entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("supervisor_init")

    # ── Conditional edge: init → retrieval (happy path) or error_exit ────
    graph.add_conditional_edges(
        "supervisor_init",
        route_after_init,
        {
            "retrieval_agent": "retrieval_agent",
            "error_exit": "error_exit",
        },
    )

    # ── Retrieval → three-way parallel fan-out ────────────────────────────
    # numeric_validation joins comparison and sentiment here rather than
    # running after them. It reads only company, quarter, retrieval_results
    # and transcript_retrieval_results -- never comparison_findings or
    # sentiment_scores -- so everything it needs exists the moment retrieval
    # finishes. Sequencing it behind the fan-in put its full duration on the
    # critical path for no dependency reason: ~1-2s warm, and up to ~9s (or
    # far worse) whenever the Serverless SQL database is resuming.
    #
    # Pure scheduling change: the three write disjoint state keys
    # (comparison_findings / sentiment_scores / numeric_validations), so
    # results are unchanged.
    graph.add_edge("retrieval_agent", "comparison_agent")
    graph.add_edge("retrieval_agent", "sentiment_agent")
    graph.add_edge("retrieval_agent", "numeric_validation_agent")

    # ── Fan-in → report ───────────────────────────────────────────────────
    # report_agent consumes all three outputs, so LangGraph's "wait for every
    # incoming edge" semantics give the correct barrier.
    graph.add_edge("comparison_agent", "report_agent")
    graph.add_edge("sentiment_agent", "report_agent")
    graph.add_edge("numeric_validation_agent", "report_agent")

    graph.add_edge("report_agent", "supervisor_finalize")
    graph.add_edge("supervisor_finalize", END)

    # ── Error path also reaches finalize so audit trail is persisted ──────
    graph.add_edge("error_exit", "supervisor_finalize")

    return graph.compile()


def _error_exit(state: GraphState) -> dict:
    """No-op sink — error already set by supervisor_init."""
    return {}


# ── Convenience: pre-compiled singleton for import by API layer ───────────────
compiled_graph = build_graph()