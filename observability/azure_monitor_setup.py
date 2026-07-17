"""
observability/azure_monitor_setup.py
Azure Monitor Application Insights for QuarterLens AI.

Instruments the FastAPI app and pipeline with Azure Monitor OpenTelemetry.
Captures:
  - API request latency, error rates, dependency calls (FastAPI auto-instrumentation)
  - Custom metrics: RAGAS scores, numeric pass rate, pipeline latency
  - Exceptions and logs streamed to Log Analytics workspace

Resume framing:
  "Instrumented FastAPI with Azure Application Insights — tracked p50/p95
   latency per endpoint and published RAGAS faithfulness + numeric accuracy
   as custom metrics with threshold alerting for model quality monitoring"

Usage:
    from observability.azure_monitor_setup import setup_azure_monitor, track_eval_metrics
    setup_azure_monitor()   # call once at app startup
    track_eval_metrics(faithfulness=0.87, numeric_pass_rate=0.65)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

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


def track_eval_metrics(
    run_name: str = "baseline",
    faithfulness: float | None = None,
    answer_relevancy: float | None = None,
    numeric_pass_rate: float | None = None,
    llm_judge_mean: float | None = None,
    context_precision: float | None = None,
    context_recall: float | None = None,
    pipeline_errors: int = 0,
) -> None:
    """
    Publish evaluation scores as Azure Monitor custom metrics.

    These appear in Azure Monitor → Metrics → Custom Namespace,
    and can be used to set threshold alerts for model quality degradation.

    Args:
        run_name:          MLflow run name for this eval (e.g. "baseline-v4").
        faithfulness:      RAGAS faithfulness score (0-1).
        answer_relevancy:  RAGAS answer relevancy score (0-1).
        numeric_pass_rate: Zero-tolerance numeric validation pass rate (0-1).
        llm_judge_mean:    LLM-as-judge mean score (1-5, normalised to 0-1 here).
        context_precision: RAGAS context precision (0-1).
        context_recall:    RAGAS context recall (0-1).
        pipeline_errors:   Number of pipeline errors in this eval run.
    """
    try:
        from opentelemetry import metrics
        meter = metrics.get_meter("quarterlens.eval")

        def _record(name: str, value: float | None, description: str) -> None:
            if value is None:
                return
            gauge = meter.create_gauge(
                name=f"quarterlens.eval.{name}",
                description=description,
                unit="1",
            )
            gauge.set(value, {"run_name": run_name})
            log.debug("Azure Monitor metric: quarterlens.eval.%s = %.4f", name, value)

        _record("faithfulness", faithfulness, "RAGAS faithfulness score")
        _record("answer_relevancy", answer_relevancy, "RAGAS answer relevancy score")
        _record("numeric_pass_rate", numeric_pass_rate, "Numeric validation pass rate")
        _record("context_precision", context_precision, "RAGAS context precision")
        _record("context_recall", context_recall, "RAGAS context recall")

        # LLM judge is 1-5 scale — normalise to 0-1 for consistent metric range
        if llm_judge_mean is not None:
            normalised_judge = (llm_judge_mean - 1) / 4
            _record("llm_judge_normalised", normalised_judge, "LLM-as-judge score (normalised 0-1)")

        # Pipeline errors as a counter
        if pipeline_errors > 0:
            counter = meter.create_counter(
                name="quarterlens.eval.pipeline_errors",
                description="Pipeline errors in eval run",
                unit="1",
            )
            counter.add(pipeline_errors, {"run_name": run_name})

        log.info(
            "Azure Monitor eval metrics published — run=%s faithfulness=%.4f numeric=%.4f",
            run_name,
            faithfulness or 0.0,
            numeric_pass_rate or 0.0,
        )

    except Exception as exc:
        log.warning("Failed to publish Azure Monitor eval metrics: %s", exc)


def track_pipeline_run(
    company: str,
    fiscal_label: str,
    latency_ms: int,
    chunks_retrieved: int,
    error: str | None = None,
) -> None:
    """
    Track a single pipeline run as a custom metric event.

    Args:
        company:          Ticker symbol.
        fiscal_label:     Filing label (e.g. "FY2025-Q3").
        latency_ms:       Total pipeline latency in milliseconds.
        chunks_retrieved: Number of chunks retrieved by the retrieval agent.
        error:            Error message if pipeline failed, else None.
    """
    try:
        from opentelemetry import metrics
        meter = metrics.get_meter("quarterlens.pipeline")

        latency_gauge = meter.create_gauge(
            name="quarterlens.pipeline.latency_ms",
            description="Pipeline run latency in milliseconds",
            unit="ms",
        )
        latency_gauge.set(latency_ms, {
            "company": company,
            "fiscal_label": fiscal_label,
            "status": "error" if error else "success",
        })

        chunks_gauge = meter.create_gauge(
            name="quarterlens.pipeline.chunks_retrieved",
            description="Chunks retrieved per pipeline run",
            unit="1",
        )
        chunks_gauge.set(chunks_retrieved, {
            "company": company,
            "fiscal_label": fiscal_label,
        })

        if error:
            error_counter = meter.create_counter(
                name="quarterlens.pipeline.errors",
                description="Pipeline error count",
                unit="1",
            )
            error_counter.add(1, {"company": company, "error_type": error[:50]})

    except Exception as exc:
        log.warning("Failed to track pipeline run metrics: %s", exc)