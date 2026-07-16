"""
observability/mlflow_tracking.py
MLflow experiment tracking wrapper for QuarterLens AI.

Provides a thin, opinionated wrapper around MLflow so every eval run
logs params + metrics in a consistent schema. Each optimization variant
(baseline, hybrid, reranker, etc.) becomes a separate MLflow run under
the same experiment — enabling valid before/after ablation comparison.

Usage:
    from observability.mlflow_tracking import start_run, log_eval_results

    with start_run(run_name="baseline", tags={"phase": "1"}):
        # ... run evaluation ...
        log_eval_results(metrics, params)
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow

log = logging.getLogger(__name__)

# Experiment name — all QuarterLens runs live under this experiment.
EXPERIMENT_NAME = "quarterlens-eval"

# Tracking URI — local ./mlruns by default; override for remote MLflow server.
_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def _ensure_experiment() -> str:
    """Get or create the QuarterLens experiment. Returns experiment_id."""
    mlflow.set_tracking_uri(_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        log.info("Created MLflow experiment '%s' (id=%s)", EXPERIMENT_NAME, experiment_id)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


@contextmanager
def start_run(
    run_name: str,
    tags: dict[str, str] | None = None,
    nested: bool = False,
):
    """
    Context manager that opens an MLflow run and closes it cleanly.

    Args:
        run_name:  Human-readable name for this run (e.g. "baseline", "hybrid-v1").
        tags:      Optional key-value tags (e.g. {"phase": "1", "variant": "baseline"}).
        nested:    True to nest inside an existing active run.

    Example:
        with start_run("baseline", tags={"phase": "1"}):
            log_eval_results(metrics, params)
    """
    experiment_id = _ensure_experiment()
    _tags = {"project": "quarterlens-ai", **(tags or {})}

    with mlflow.start_run(
        run_name=run_name,
        experiment_id=experiment_id,
        tags=_tags,
        nested=nested,
    ) as run:
        log.info("MLflow run started: %s (id=%s)", run_name, run.info.run_id)
        try:
            yield run
        except Exception:
            mlflow.set_tag("run_status", "failed")
            raise
        else:
            mlflow.set_tag("run_status", "completed")
        log.info("MLflow run completed: %s", run_name)


def log_eval_results(
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
) -> None:
    """
    Log evaluation metrics, params, and artifact paths to the active run.

    Args:
        metrics:    Dict of metric name → float value.
                    e.g. {"faithfulness": 0.91, "precision_at_5": 0.82}
        params:     Dict of hyperparameters / config values for this run.
                    e.g. {"retrieval_top_k": 5, "model": "gpt-5-mini"}
        artifacts:  Dict of label → local file path to log as MLflow artifact.
                    e.g. {"eval_report": "evaluation/reports/baseline.json"}
    """
    if not mlflow.active_run():
        log.warning("log_eval_results called outside an active MLflow run — skipping.")
        return

    if params:
        mlflow.log_params(params)

    if metrics:
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            log.info("  %s: %.4f", name, value)

    if artifacts:
        for label, path in artifacts.items():
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path=label)
            else:
                log.warning("Artifact not found, skipping: %s (%s)", label, path)


def log_per_claim_results(claim_results: list[dict[str, Any]]) -> None:
    """
    Log per-claim evaluation results as a JSON artifact for debugging.

    Args:
        claim_results: List of per-claim dicts from run_baseline_eval.py.
    """
    import json
    import tempfile

    if not mlflow.active_run():
        log.warning("log_per_claim_results called outside an active MLflow run — skipping.")
        return

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="per_claim_"
    ) as f:
        json.dump(claim_results, f, indent=2)
        tmp_path = f.name

    mlflow.log_artifact(tmp_path, artifact_path="per_claim")
    os.unlink(tmp_path)
    log.info("Logged %d per-claim results to MLflow", len(claim_results))