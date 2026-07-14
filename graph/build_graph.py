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

from langgraph.graph import StateGraph, END

from graph.state import GraphState
from agents.supervisor import supervisor_init, supervisor_finalize, route_after_init
from agents.retrieval_agent import retrieval_agent
from agents.comparison_agent import comparison_agent
from agents.sentiment_agent import sentiment_agent
from agents.numeric_validation_agent import numeric_validation_agent
from agents.report_agent import report_agent


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node("supervisor_init", supervisor_init)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("comparison_agent", comparison_agent)
    graph.add_node("sentiment_agent", sentiment_agent)
    graph.add_node("numeric_validation_agent", numeric_validation_agent)
    graph.add_node("report_agent", report_agent)
    graph.add_node("supervisor_finalize", supervisor_finalize)
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

    # ── Retrieval → parallel fan-out ──────────────────────────────────────
    graph.add_edge("retrieval_agent", "comparison_agent")
    graph.add_edge("retrieval_agent", "sentiment_agent")

    # ── Parallel fan-in → numeric validation ─────────────────────────────
    # LangGraph waits for all incoming edges before executing the target node.
    graph.add_edge("comparison_agent", "numeric_validation_agent")
    graph.add_edge("sentiment_agent", "numeric_validation_agent")

    # ── Sequential tail ───────────────────────────────────────────────────
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