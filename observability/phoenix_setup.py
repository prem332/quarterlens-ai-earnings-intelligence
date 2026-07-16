"""
observability/phoenix_setup.py
Phoenix tracing instrumentation for QuarterLens AI.

Instruments the LangGraph pipeline so every agent span — token cost,
latency, tool calls, retrieved chunks — is visible in the Phoenix UI.

Usage:
    from observability.phoenix_setup import setup_phoenix
    setup_phoenix()   # call once at app startup, before any graph invocation
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)

# Phoenix collector endpoint — override via env var for remote collector.
# Default points to a locally-running Phoenix server (phoenix serve).
_DEFAULT_COLLECTOR = "http://localhost:6006"


def setup_phoenix(
    project_name: str = "quarterlens-ai",
    collector_endpoint: str | None = None,
) -> None:
    """
    Instrument the LangGraph pipeline with Phoenix OTEL tracing.

    Args:
        project_name:        Phoenix project name (groups runs in the UI).
        collector_endpoint:  OTLP collector URL. Defaults to localhost:6006
                             or PHOENIX_COLLECTOR_ENDPOINT env var.
    """
    try:
        from phoenix.otel import register
    except ImportError:
        log.warning(
            "arize-phoenix-otel not installed — Phoenix tracing disabled. "
            "Install with: pip install arize-phoenix-otel"
        )
        return

    endpoint = (
        collector_endpoint
        or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        or _DEFAULT_COLLECTOR
    )

    # register() sets up the OTEL TracerProvider and points it at Phoenix.
    tracer_provider = register(
        project_name=project_name,
        endpoint=f"{endpoint}/v1/traces",
        verbose=False,
    )

    # Instrument LangChain / LangGraph automatically.
    _instrument_langchain(tracer_provider)

    log.info(
        "Phoenix tracing enabled — project=%s endpoint=%s",
        project_name,
        endpoint,
    )


def _instrument_langchain(tracer_provider) -> None:
    """Attach the LangChain auto-instrumentor if available."""
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        log.debug("LangChain instrumentor attached")
    except ImportError:
        log.warning(
            "openinference-instrumentation-langchain not installed — "
            "LangChain spans will not be traced. "
            "Install with: pip install openinference-instrumentation-langchain"
        )