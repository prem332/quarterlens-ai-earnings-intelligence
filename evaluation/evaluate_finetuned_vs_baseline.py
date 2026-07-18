"""
evaluation/evaluate_finetuned_vs_baseline.py

Compares three model variants against the locked golden dataset:
  1. baseline      — gpt-5.4-mini (primary, no fine-tuning) [baseline-v4 locked]
  2. finetuned_sft — gpt-4o-mini fine-tuned via Azure SFT (report_agent only)
  3. qlora_gemma   — Gemma-2-2B QLoRA (Kaggle, metrics from qlora_metrics.csv)

For each runnable claim in the golden dataset, runs the full LangGraph pipeline
with the relevant model variant for report_agent, then scores with:
  - RAGAS (faithfulness, answer_relevancy)
  - Precision@k / Recall@k
  - LLM-as-judge (3 dimensions, claim-type weighted)
  - Numeric pass rate (zero tolerance)

All results logged to MLflow. QLoRA Gemma metrics are logged from CSV
(Gemma runs on Kaggle — not locally callable) with a clear note.

Baseline-v4 locked metrics (never re-run, used as reference only):
  ragas_faithfulness=0.5867, ragas_answer_relevancy=0.6493,
  llm_judge_mean=2.6333, numeric_pass_rate=0.4231,
  precision_at_5=0.0, recall_at_5=0.0

Usage:
    python evaluation/evaluate_finetuned_vs_baseline.py
    python evaluation/evaluate_finetuned_vs_baseline.py --variant finetuned_sft
    python evaluation/evaluate_finetuned_vs_baseline.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Observability init — must precede openai_client import ───────────────────
# Langfuse patches AzureOpenAI at import time — must run first.
_obs_log = logging.getLogger("observability")

try:
    from observability.langfuse_setup import setup_langfuse
    setup_langfuse()
    _obs_log.info("Langfuse initialized successfully")
except Exception as _lf_exc:
    _obs_log.warning("Langfuse init failed (non-fatal): %s", _lf_exc)

try:
    from observability.phoenix_setup import setup_phoenix
    setup_phoenix()
    _obs_log.info("Phoenix initialized successfully")
except Exception as _px_exc:
    _obs_log.warning("Phoenix init failed (non-fatal): %s", _px_exc)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("evaluate_finetuned_vs_baseline")

# ── Locked baseline-v4 metrics (never re-run) ─────────────────────────────────
_BASELINE_V4: dict[str, Any] = {
    "ragas_faithfulness":    0.5867,
    "ragas_answer_relevancy": 0.6493,
    "llm_judge_mean":        2.6333,
    "numeric_pass_rate":     0.4231,
    "precision_at_5":        0.0,
    "recall_at_5":           0.0,
    "pipeline_errors":       0,
    "total_claims":          75,
    "run_name":              "baseline-v4",
    "model":                 "gpt-5.4-mini",
}

# ── QLoRA Gemma locked metrics (from Kaggle run) ─────────────────────────────
_QLORA_METRICS_PATH = Path(__file__).resolve().parent.parent / "finetuning" / "qlora_metrics.csv"

_RUNNABLE_TYPES = {"retrieval", "comparison", "numeric", "out_of_scope", "sentiment"}


# ── Reuse helpers from run_baseline_eval ─────────────────────────────────────

def _load_claims(claims_dir: Path) -> list[dict]:
    claims = []
    for path in sorted(claims_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            claims.extend(data) if isinstance(data, list) else claims.append(data)
        except Exception as e:
            log.warning("Skipping %s — parse error: %s", path.name, e)
    log.info("Loaded %d claims from %s", len(claims), claims_dir)
    return claims


def _build_query(claim: dict) -> str:
    payload = claim.get("payload") or {}
    claim_type = claim.get("claim_type", "")
    if claim_type == "numeric":
        return payload.get("verbal_claim", "")
    elif claim_type == "retrieval":
        return payload.get("query", "")
    elif claim_type == "comparison":
        lang = payload.get("current_quarter_lang", "")
        return f"Did this language shift from the prior quarter? {lang}" if lang else ""
    elif claim_type == "out_of_scope":
        return payload.get("query", "")
    elif claim_type == "sentiment":
        span = payload.get("span", "")
        speaker = payload.get("speaker", "")
        return f'What is the sentiment of this statement by {speaker}: "{span}"' if span else ""
    return ""


def _build_ground_truth(claim: dict) -> str:
    gt = claim.get("ground_truth") or {}
    claim_type = claim.get("claim_type", "")
    if claim_type == "numeric":
        return f"Filed value: {gt.get('filed_value', '')} {gt.get('unit', '')}. Verdict: {gt.get('verdict', '')}."
    elif claim_type == "retrieval":
        return claim.get("payload", {}).get("expected_answer_gist", "")
    elif claim_type == "comparison":
        return f"Expected shift: {gt.get('expected_shift')}. {claim.get('payload', {}).get('shift_description', '')}"
    elif claim_type == "out_of_scope":
        return f"Expected behavior: {gt.get('expected_behavior', 'refuse')}. {gt.get('refusal_reason', '')}"
    elif claim_type == "sentiment":
        return f"Expected sentiment: {gt.get('label', '')}. {gt.get('rationale', '')}"
    return str(gt)


def _extract_gt_anchors(claim: dict) -> list[dict]:
    gt = claim.get("ground_truth") or {}
    claim_type = claim.get("claim_type", "")
    anchors = []

    def _extract(a: dict) -> dict | None:
        acc = a.get("accession")
        section = (a.get("locator") or {}).get("section")
        return {"accession": acc, "section": section} if acc and section else None

    if claim_type == "retrieval":
        for a in gt.get("relevant_anchors", []):
            e = _extract(a)
            if e:
                anchors.append(e)
    elif claim_type == "comparison":
        for key in ("current_anchor", "prior_anchor"):
            a = gt.get(key)
            if a:
                e = _extract(a)
                if e:
                    anchors.append(e)
    elif claim_type == "out_of_scope":
        a = gt.get("anchor")
        if a:
            e = _extract(a)
            if e:
                anchors.append(e)
    return anchors


# Floor/ceiling qualifiers that mean the stated value is NOT an exact figure.
# Golden dataset deliberately tests that these are NOT accepted as equivalent.
_FLOOR_WORDS = ("exceeded", "over", "more than", "greater than", "at least", "north of", "above")


def _parse_value(v) -> float | None:
    """
    Parse numeric values to float, normalised to USD millions for money.
    Handles: '$94 billion', '94,036', '46.5%', '-8%', '$10 billion / 22%'.
    Percentages: '%' stripped, value kept as-is (e.g. '46.5%' -> 46.5).
    Money: 'billion' -> *1000 (to millions), 'million' -> *1.
    For compound strings ('$10 billion / 22%'), the first money value wins.
    Returns None if no number can be parsed.
    """
    if v is None:
        return None
    s = str(v).strip().lower().replace(',', '').strip()

    # Compound like "$10 billion / 22%" — take the substring before the first '/'
    if '/' in s:
        s = s.split('/')[0].strip()

    is_percent = '%' in s
    s = s.replace('%', '').replace('$', '').strip()

    multiplier = 1.0
    if 'billion' in s:
        multiplier = 1_000.0          # billions -> millions
        s = s.replace('billion', '').strip()
    elif 'million' in s:
        multiplier = 1.0
        s = s.replace('million', '').strip()

    # Pull the first numeric token (handles leading text like "exceeded 54")
    import re
    m = re.search(r'-?\d+\.?\d*', s)
    if not m:
        return None
    try:
        val = float(m.group()) * multiplier
        return val
    except ValueError:
        return None


def _score_numeric(claim: dict, answer: str) -> dict:
    """
    Zero-tolerance numeric validation.
    Compares payload.stated_value (exec's verbal claim) against
    ground_truth.filed_value. Does NOT use the pipeline answer —
    this measures whether the exec's stated figure matches the filing.

    Handles:
      - Absolute money (USD millions), percentages (pp), exact and banded tolerances.
      - Floor statements ('exceeded $54B'): correctly FAIL when tolerance is 'exact',
        because a floor is not an exact figure (golden dataset tests this explicitly).
    """
    gt = claim.get("ground_truth") or {}
    payload = claim.get("payload") or {}
    filed_raw = gt.get("filed_value")
    stated_raw = payload.get("stated_value")
    tolerance = gt.get("tolerance_rule", "exact")

    if filed_raw is None or stated_raw is None:
        return {"numeric_pass": None, "numeric_delta": None,
                "filed_value": filed_raw, "stated_value": stated_raw,
                "tolerance_rule": tolerance,
                "note": "missing filed_value or stated_value"}

    stated_str = str(stated_raw).lower()
    is_floor_statement = any(w in stated_str for w in _FLOOR_WORDS)

    unit = str(gt.get("unit", "")).lower()

    filed_f = _parse_value(filed_raw)
    stated_f = _parse_value(stated_raw)

    # Unit normalization: if the filed value's unit is expressed in billions
    # (e.g. "billion people", "billion USD") but the raw filed number is bare
    # (3.54, not 3540), scale it to match _parse_value's billion handling,
    # which converts "3.5 billion" -> 3500. This keeps delta meaningful.
    if "billion" in unit and filed_f is not None and abs(filed_f) < 1000:
        filed_f = filed_f * 1_000.0

    if filed_f is None or stated_f is None:
        return {"numeric_pass": None, "numeric_delta": None,
                "filed_value": filed_raw, "stated_value": stated_raw,
                "tolerance_rule": tolerance,
                "note": f"could not parse: filed={filed_raw} stated={stated_raw}"}

    delta = abs(filed_f - stated_f)

    # Floor statement + exact tolerance = FAIL. A floor ('exceeded $54B') is not
    # an exact figure; accepting it as equal to $54.5B infers beyond what was said.
    if is_floor_statement and (tolerance == "exact" or str(tolerance).strip().startswith("exact")):
        return {
            "numeric_pass": False,
            "numeric_delta": round(delta, 4),
            "filed_value": filed_f,
            "stated_value": stated_f,
            "tolerance_rule": tolerance,
            "note": "floor statement not accepted as exact",
        }

    # Determine the tolerance band.
    # Rules seen: "exact", "abs<=500M ...", "abs<=0.5pp ...",
    #             "abs<=500M for absolute; abs<=0.5pp for %; ..."
    tol_str = str(tolerance).lower()
    passed = False

    if tol_str.startswith("exact"):
        passed = delta == 0.0
    elif "abs<=" in tol_str:
        # Choose the % band if the stated value is a percentage, else the money band.
        # For compound values ('$10 billion / 22%'), classify by the portion actually
        # parsed — the part before '/'. A money value there means use the money band.
        stated_portion = str(stated_raw).split('/')[0].strip()
        stated_is_pct = ("%" in stated_portion) and not any(
            u in stated_portion.lower() for u in ("billion", "million", "$")
        )
        band = None
        # Collect all "abs<=N(pp|m)" occurrences
        import re
        for m in re.finditer(r"abs<=\s*([\d.]+)\s*(pp|m)?", tol_str):
            num = float(m.group(1))
            kind = m.group(2)  # 'pp', 'm', or None
            if stated_is_pct and kind == "pp":
                band = num
                break
            if (not stated_is_pct) and kind == "m":
                band = num
                break
            # Fallback: first band if kind ambiguous
            if band is None:
                band = num
        passed = band is not None and delta <= band
    else:
        passed = delta == 0.0

    return {
        "numeric_pass": passed,
        "numeric_delta": round(delta, 4),
        "filed_value": filed_f,
        "stated_value": stated_f,
        "tolerance_rule": tolerance,
    }


# ── Pipeline runner with variant routing ─────────────────────────────────────

async def _run_pipeline(
    query: str,
    company: str,
    fiscal_label: str,
    variant: str,
) -> dict[str, Any]:
    """
    Run LangGraph pipeline with model variant override for report_agent.

    variant:
        "finetuned_sft" — injects AZURE-OPENAI-DEPLOYMENT-NAME-FINETUNED
                          as the primary deployment for report_agent
        "baseline"      — uses standard primary deployment (gpt-5.4-mini)
    """
    from graph.build_graph import compiled_graph
    from graph.state import GraphState

    initial_state: GraphState = {
        "company": company,
        "quarter": fiscal_label,
        "query": query,
        "comparison_quarters": [],
        "retrieval_results": [],
        "comparison_findings": [],
        "sentiment_scores": [],
        "numeric_validations": [],
        "report": "",
        "decision_log_entries": [],
        "model_tier": "primary",           # all agents always use primary tier
        "report_model_tier": "finetuned" if variant == "finetuned_sft" else "primary",  # report_agent only
        "error": None,
    }

    try:
        result = await compiled_graph.ainvoke(initial_state)
        chunks = result.get("retrieval_results") or []
        contexts = [c.get("content", "") for c in chunks if isinstance(c, dict)]
        return {
            "answer": result.get("report") or "",
            "contexts": contexts,
            "chunks": chunks,
            "error": result.get("error"),
        }
    except Exception as e:
        log.warning("Pipeline error [%s] for '%s': %s", variant, query[:60], e)
        return {"answer": "", "contexts": [], "chunks": [], "error": str(e)}


# ── QLoRA metrics loader ──────────────────────────────────────────────────────

def _load_qlora_metrics() -> dict[str, Any]:
    """Load QLoRA Gemma metrics from the CSV saved during Kaggle training."""
    if not _QLORA_METRICS_PATH.exists():
        log.warning("qlora_metrics.csv not found at %s", _QLORA_METRICS_PATH)
        return {}
    try:
        with open(_QLORA_METRICS_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            return {
                "model":                    row.get("model", "gemma-2-2b-it-qlora"),
                "final_valid_loss":         float(row.get("final_valid_loss", 0)),
                "final_valid_token_acc":    float(row.get("final_valid_token_accuracy", 0)),
                "train_loss_reduction_pct": float(row.get("train_loss_reduction_pct", 0)),
                "valid_loss_reduction_pct": float(row.get("valid_loss_reduction_pct", 0)),
                "trainable_pct":            float(row.get("trainable_pct", 0)),
                "lora_r":                   int(row.get("lora_r", 16)),
                "lora_alpha":               int(row.get("lora_alpha", 32)),
                "n_epochs":                 int(row.get("n_epochs", 3)),
                "training_time_seconds":    int(row.get("training_time_seconds", 0)),
                "note": "QLoRA Gemma-2-2B trained on Kaggle T4 — golden dataset eval not applicable (model not deployed); training metrics only",
            }
    except Exception as e:
        log.warning("Failed to load qlora_metrics.csv: %s", e)
        return {}


# ── Main eval loop ────────────────────────────────────────────────────────────

async def run_finetuned_eval(
    claims_dir: Path,
    variant: str = "finetuned_sft",
    k: int = 5,
    run_name: str = "finetuned-eval-v1",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run evaluation for a fine-tuned model variant against the golden dataset.
    Logs results + baseline-v4 reference + QLoRA training metrics to MLflow.
    """
    from evaluation.ragas_eval import run_ragas_eval
    from evaluation.precision_recall_at_k import compute_batch_retrieval_metrics
    from evaluation.llm_as_judge import judge_batch
    from observability.mlflow_tracking import start_run, log_eval_results, log_per_claim_results

    claims = _load_claims(claims_dir)
    runnable = [c for c in claims if c.get("claim_type") in _RUNNABLE_TYPES]
    log.info("%d/%d claims runnable", len(runnable), len(claims))

    if dry_run:
        log.info("Dry run — no pipeline invocation.")
        return {"dry_run": True, "total_claims": len(claims), "runnable_claims": len(runnable)}

    ragas_samples: list[dict] = []
    retrieval_batch: list[dict] = []
    judge_samples: list[dict] = []
    numeric_results: list[dict] = []
    per_claim_results: list[dict] = []

    for claim in runnable:
        claim_id = claim.get("claim_id", str(uuid.uuid4()))
        claim_type = claim.get("claim_type", "retrieval")
        query = _build_query(claim)
        ground_truth = _build_ground_truth(claim)
        company = claim.get("company", "")
        fiscal_label = claim.get("fiscal_label", "")

        if not query:
            log.warning("Claim %s has no query — skipping", claim_id)
            continue

        log.info("Claim %s (%s | %s/%s)", claim_id, claim_type, company, fiscal_label)

        t0 = time.time()
        pipeline_out = await _run_pipeline(query, company, fiscal_label, variant)
        latency_ms = int((time.time() - t0) * 1000)
        time.sleep(3)

        answer = pipeline_out["answer"]
        contexts = pipeline_out["contexts"]
        chunks = pipeline_out["chunks"]
        pipeline_error = pipeline_out["error"]

        ragas_samples.append({
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

        gt_anchors = _extract_gt_anchors(claim)
        if gt_anchors:
            retrieval_batch.append({
                "claim_id": claim_id,
                "retrieved_chunks": chunks,
                "ground_truth_anchors": gt_anchors,
            })

        judge_samples.append({
            "claim_id": claim_id,
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "claim_type": claim_type,
        })

        if claim_type == "numeric":
            num_result = _score_numeric(claim, answer)
            num_result["claim_id"] = claim_id
            numeric_results.append(num_result)

        per_claim_results.append({
            "claim_id": claim_id,
            "claim_type": claim_type,
            "company": company,
            "fiscal_label": fiscal_label,
            "latency_ms": latency_ms,
            "pipeline_error": pipeline_error,
            "answer_length": len(answer),
        })

    # ── Scoring ───────────────────────────────────────────────────────────────
    log.info("Scoring %d samples...", len(ragas_samples))

    ragas_scores = run_ragas_eval(
        ragas_samples, metrics=["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ) if ragas_samples else {}

    retrieval_scores = (
        compute_batch_retrieval_metrics(retrieval_batch, k=k)
        if retrieval_batch else {}
    )

    judge_scores_list, mean_judge = (
        judge_batch(judge_samples) if judge_samples else ([], 0.0)
    )

    numeric_pass_rate = 0.0
    if numeric_results:
        passed = sum(1 for r in numeric_results if r.get("numeric_pass") is True)
        numeric_pass_rate = round(passed / len(numeric_results), 4)

    # ── Build metrics dict ────────────────────────────────────────────────────
    metrics = {
        **{f"ragas_{k_}": v for k_, v in ragas_scores.items()},
        f"precision_at_{k}": retrieval_scores.get("mean_precision_at_k", 0.0),
        f"recall_at_{k}": retrieval_scores.get("mean_recall_at_k", 0.0),
        "llm_judge_mean": mean_judge,
        "numeric_pass_rate": numeric_pass_rate,
        "total_claims": len(runnable),
        "pipeline_errors": sum(1 for r in per_claim_results if r["pipeline_error"]),
    }

    # ── Delta vs baseline-v4 ──────────────────────────────────────────────────
    delta_metrics = {}
    for key in ["ragas_faithfulness", "ragas_answer_relevancy", "llm_judge_mean",
                "numeric_pass_rate", f"precision_at_{k}", f"recall_at_{k}"]:
        if key in metrics and key in _BASELINE_V4:
            delta = round(metrics[key] - _BASELINE_V4[key], 4)
            pct = round((delta / _BASELINE_V4[key]) * 100, 2) if _BASELINE_V4[key] else 0.0
            delta_metrics[f"delta_{key}"] = delta
            delta_metrics[f"delta_pct_{key}"] = pct

    params = {
        "variant": variant,
        "run_name": run_name,
        "retrieval_k": k,
        "phase": "2",
        "finetuned_model": "gpt-4o-mini-2024-07-18.ft-90ba5d7aafc94e59b3c6a87277edcabf",
        "finetuned_deployment": "gpt-4o-mini-finetuned",
        "ft_n_epochs": 3,
        "ft_batch_size": 1,
        "ft_lr_multiplier": 1.0,
        "ft_seed": 42,
        "ft_training_type": "globalStandard",
        "ft_trained_tokens": 1176690,
        "ft_valid_token_accuracy": 0.827,
        "ft_valid_loss_final": 0.495,
        # QLoRA Gemma reference params
        "qlora_model": "gemma-2-2b-it",
        "qlora_lora_r": 16,
        "qlora_lora_alpha": 32,
        "qlora_trainable_pct": 0.7881,
        "qlora_valid_token_accuracy": 0.8714,
        "qlora_valid_loss_final": 0.5608,
        "qlora_train_loss_reduction_pct": 66.2,
        "qlora_note": "Trained on Kaggle T4, metrics from qlora_metrics.csv",
        # Baseline reference
        "baseline_run": "baseline-v4",
        "baseline_faithfulness": _BASELINE_V4["ragas_faithfulness"],
        "baseline_judge_mean": _BASELINE_V4["llm_judge_mean"],
    }

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    with start_run(run_name=run_name, tags={"phase": "2", "variant": variant}):
        log_eval_results(
            metrics={**metrics, **delta_metrics},
            params=params,
        )
        log_per_claim_results(per_claim_results)

        # Log QLoRA metrics as a nested artifact
        qlora_metrics = _load_qlora_metrics()
        if qlora_metrics:
            import mlflow
            mlflow.log_dict(qlora_metrics, "qlora_gemma_training_metrics.json")
            log.info("QLoRA Gemma training metrics logged to MLflow artifact.")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"FINETUNED EVAL COMPLETE — variant={variant}")
    print("=" * 65)
    print(f"\n{'Metric':<35} {'Baseline-v4':>12} {'This run':>12} {'Delta':>10}")
    print("-" * 70)
    for key in ["ragas_faithfulness", "ragas_answer_relevancy",
                "ragas_context_precision", "ragas_context_recall",
                "llm_judge_mean", "numeric_pass_rate",
                f"precision_at_{k}", f"recall_at_{k}"]:
        baseline_val = _BASELINE_V4.get(key, 0.0)
        this_val = metrics.get(key, 0.0)
        delta = this_val - baseline_val
        print(f"  {key:<33} {baseline_val:>12.4f} {this_val:>12.4f} {delta:>+10.4f}")

    print(f"\n  pipeline_errors : {metrics['pipeline_errors']}")
    print(f"  total_claims    : {metrics['total_claims']}")
    print("\nQLoRA Gemma-2-2B (training metrics, Kaggle T4):")
    print(f"  valid_token_accuracy : 87.14%")
    print(f"  valid_loss           : 0.5608")
    print(f"  train_loss_reduction : 66.2%")
    print(f"  trainable_params     : 0.79% (20.7M / 2.6B)")
    print("=" * 65)

    return {**metrics, **delta_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model vs baseline-v4 on golden dataset."
    )
    parser.add_argument(
        "--claims-dir", default="golden_dataset/claims",
        help="Directory containing golden claim JSON files.",
    )
    parser.add_argument(
        "--variant", default="finetuned_sft",
        choices=["finetuned_sft", "baseline"],
        help="Model variant to evaluate.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--run-name", default="finetuned-eval-v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_finetuned_eval(
        claims_dir=Path(args.claims_dir),
        variant=args.variant,
        k=args.k,
        run_name=args.run_name,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()