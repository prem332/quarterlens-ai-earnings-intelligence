"""
observability/azure_monitor_setup.py
Azure Monitor Application Insights for QuarterLens AI.

Instruments the FastAPI app and pipeline with Azure Monitor OpenTelemetry.
Captures:
  - API request latency, error rates, dependency calls (FastAPI auto-instrumentation)
  - Exceptions and logs streamed to Log Analytics workspace

Resume framing:
  "Instrumented FastAPI with Azure Application Insights — tracked p50/p95
   latency per endpoint for pipeline observability"

Usage:
    from observability.azure_monitor_setup import setup_azure_monitor
    setup_azure_monitor()   # call once at app startup
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)

_monitor_client = None
_setup_done = False


def setup_azure_monitor() -> bool:
    """
    Configure Azure Monitor OpenTelemetry for the FastAPI app.

    Reads connection string from Key Vault (APPLICATIONINSIGHTS-CONNECTION-STRING).
    Falls back gracefully if not available.

    Returns:
        True if setup succeeded, False otherwise.
    """
    global _setup_done

    if _setup_done:
        log.debug("Azure Monitor already configured — skipping duplicate call")
        return True

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
    except ImportError:
        log.warning(
            "azure-monitor-opentelemetry not installed — Azure Monitor disabled. "
            "Install with: pip install azure-monitor-opentelemetry"
        )
        return False

    try:
        from azure_clients.key_vault_client import kv
        connection_string = kv.get_secret("APPLICATIONINSIGHTS-CONNECTION-STRING")
    except Exception as exc:
        log.warning("Could not read Application Insights connection string: %s", exc)
        return False

    try:
        configure_azure_monitor(
            connection_string=connection_string,
            logger_name="quarterlens",      # capture logs from this logger namespace
        )
        _setup_done = True
        log.info("Azure Monitor Application Insights configured — telemetry streaming to Azure")
        return True

    except Exception as exc:
        log.warning("Azure Monitor setup failed: %s", exc)
        return False
