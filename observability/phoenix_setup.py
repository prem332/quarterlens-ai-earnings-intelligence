"""
observability/phoenix_setup.py
Phoenix tracing instrumentation for QuarterLens AI.

Instruments the LangGraph pipeline so every agent span — token cost,
latency, tool calls, retrieved chunks — is visible in the Phoenix UI.

Supports both local Phoenix server and Phoenix Cloud (app.phoenix.arize.com).

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

_DEFAULT_LOCAL_ENDPOINT = "http://localhost:6006"
_instrumented = False  # guard against double instrumentation


def setup_phoenix(
    project_name: str = "quarterlens-ai",
    collector_endpoint: str | None = None,
) -> None:
    """
    Instrument the LangGraph pipeline with Phoenix OTEL tracing.

    Resolution order for endpoint:
      1. collector_endpoint argument
      2. PHOENIX_COLLECTOR_ENDPOINT env var
      3. Key Vault PHOENIX-ENDPOINT secret
      4. localhost:6006 (local fallback)

    Resolution order for API key:
      1. PHOENIX_API_KEY env var
      2. Key Vault PHOENIX-API-KEY secret
      3. None (local Phoenix, no auth required)

    Args:
        project_name:        Phoenix project name (groups runs in the UI).
        collector_endpoint:  Override endpoint URL.
    """
    global _instrumented
    if _instrumented:
        log.debug("Phoenix already instrumented — skipping duplicate setup_phoenix() call")
        return

    try:
        from phoenix.otel import register
    except ImportError:
        log.warning(
            "arize-phoenix-otel not installed — Phoenix tracing disabled. "
            "Install with: pip install arize-phoenix-otel"
        )
        return

    # ── Resolve endpoint ──────────────────────────────────────────────────────
    endpoint = collector_endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT")

    if not endpoint:
        try:
            from azure_clients.key_vault_client import kv
            endpoint = kv.get_secret("PHOENIX-ENDPOINT")
        except Exception:
            pass

    endpoint = endpoint or _DEFAULT_LOCAL_ENDPOINT

    # ── Resolve API key ───────────────────────────────────────────────────────
    api_key = os.getenv("PHOENIX_API_KEY")

    if not api_key:
        try:
            from azure_clients.key_vault_client import kv
            api_key = kv.get_secret("PHOENIX-API-KEY")
        except Exception:
            pass

    # Set env vars — phoenix.otel.register reads these automatically
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = endpoint
    if api_key:
        os.environ["PHOENIX_API_KEY"] = api_key

    # ── Register tracer provider ──────────────────────────────────────────────
    try:
        tracer_provider = register(
            project_name=project_name,
            endpoint=f"{endpoint}/v1/traces",
            protocol="http/protobuf",
            batch=True,
            verbose=False,
        )
    except Exception as exc:
        log.warning(
            "Phoenix register() failed — tracing disabled. Error: %s", exc
        )
        return

    # ── Instrument LangChain / LangGraph ─────────────────────────────────────
    _instrument_langchain(tracer_provider)

    # ── Instrument raw OpenAI calls ───────────────────────────────────────────
    _instrument_openai(tracer_provider)

    _instrumented = True

    is_cloud = "arize.com" in endpoint
    log.info(
        "Phoenix tracing enabled — project=%s endpoint=%s mode=%s",
        project_name,
        endpoint,
        "cloud" if is_cloud else "local",
    )


def _instrument_openai(tracer_provider) -> None:
    """Attach the OpenAI auto-instrumentor if available."""
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        log.debug("OpenAI instrumentor attached")
    except ImportError:
        log.warning(
            "openinference-instrumentation-openai not installed — "
            "raw OpenAI spans will not be traced."
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
            "LangChain spans will not be traced."
        )