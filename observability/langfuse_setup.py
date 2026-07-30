"""
observability/langfuse_setup.py
Langfuse LLM observability for QuarterLens AI.

Sits alongside phoenix_setup.py — both instrument the same pipeline.
Phoenix = primary (in ARCHITECTURE.md spec).
Langfuse = secondary (resume flexibility — some JDs specify Langfuse).

Langfuse traces every LLM call, retrieval step, and agent execution.
View traces at: https://us.cloud.langfuse.com

Usage:
    from observability.langfuse_setup import setup_langfuse, get_langfuse_client

    setup_langfuse()  # call once at app startup

    # Manual tracing (optional — OTEL auto-instrumentation handles LangChain):
    client = get_langfuse_client()
    trace = client.trace(name="pipeline-run", metadata={"company": "AAPL"})
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)

_langfuse_client = None


def setup_langfuse(
    project_name: str = "quarterlens-ai",
    debug: bool = False,
) -> bool:
    """
    Initialise Langfuse and instrument the LangChain/LangGraph pipeline
    via OpenTelemetry auto-instrumentation.

    Args:
        project_name: Langfuse project name (for display only).
        debug:        Enable verbose Langfuse SDK logging.

    Returns:
        True if setup succeeded, False if Langfuse is unavailable.
    """
    global _langfuse_client

    try:
        from langfuse import Langfuse
        
    except ImportError:
        log.warning(
            "langfuse not installed — Langfuse tracing disabled. "
            "Install with: pip install langfuse"
        )
        return False

    try:
        from azure_clients.key_vault_client import kv

        public_key = kv.get_secret("LANGFUSE-PUBLIC-KEY")
        secret_key = kv.get_secret("LANGFUSE-SECRET-KEY")
        host = kv.get_secret("LANGFUSE-HOST")

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            debug=debug,
        )

        # Verify connectivity
        _langfuse_client.auth_check()

        log.info(
            "Langfuse tracing enabled — project=%s host=%s",
            project_name,
            host,
        )
        return True

    except Exception as exc:
        log.warning(
            "Langfuse setup failed — tracing disabled. Error: %s", exc
        )
        return False


def get_langfuse_client():
    """
    Return the initialised Langfuse client, or None if not set up.
    Use for manual trace/span creation where auto-instrumentation
    doesn't cover a step (e.g. custom retrieval logic, numeric validation).
    """
    return _langfuse_client


def flush_langfuse() -> None:
    """
    Flush pending Langfuse events — call at app shutdown or end of eval run
    to ensure all traces are sent before the process exits.
    """
    client = get_langfuse_client()
    if client:
        try:
            client.flush()
            log.debug("Langfuse events flushed.")
        except Exception as exc:
            log.warning("Langfuse flush failed: %s", exc)