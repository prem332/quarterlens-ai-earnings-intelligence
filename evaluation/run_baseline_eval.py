"""
evaluation/run_baseline_eval.py
Phase 1 baseline evaluation runner for QuarterLens AI.

Loads the golden dataset, runs each claim through the compiled LangGraph
pipeline, collects (question, answer, contexts, ground_truth) tuples, then
scores with RAGAS + precision/recall@k + LLM-as-judge. Logs everything to
MLflow as a single "baseline" run.

This is the Phase 1 baseline. Every Phase 2 optimization is measured
against this run in MLflow — never against a moving target.

Usage:
    python evaluation/run_baseline_eval.py
    python evaluation/run_baseline_eval.py --claims-dir golden_dataset/claims
    python evaluation/run_baseline_eval.py --k 5 --run-name baseline-v1
    python evaluation/run_baseline_eval.py --dry-run   # schema check only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("run_baseline_eval")

# ── Claim type routing ────────────────────────────────────────────────────────
_RUNNABLE_TYPES = {"retrieval", "comparison", "numeric", "out_of_scope", "sentiment"}


def _load_claims(claims_dir: Path) -> list[dict]:
    """Glob all *.json files in claims_dir and return parsed claim dicts."""
    claims = []
    for path in sorted(claims_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                claims.extend(data)
            else:
                claims.append(data)
        except Exception as e:
            log.warning("Skipping %s — parse error: %s", path.name, e)
    log.info("Loaded %d claims from %s", len(claims), claims_dir)
    return claims


def _build_query(claim: dict) -> str:
    """Extract the runnable query string from a claim's payload."""
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
    """Extract the expected answer string from a claim's ground_truth."""
    gt = claim.get("ground_truth") or {}
    claim_type = claim.get("claim_type", "")

    if claim_type == "numeric":
        filed = gt.get("filed_value", "")
        verdict = gt.get("verdict", "")
        unit = gt.get("unit", "")
        return f"Filed value: {filed} {unit}. Verdict: {verdict}."
    elif claim_type == "retrieval":
        return claim.get("payload", {}).get("expected_answer_gist", "")
    elif claim_type == "comparison":
        shift = gt.get("expected_shift")
        desc = claim.get("payload", {}).get("shift_description", "")
        return f"Expected shift: {shift}. {desc}"
    elif claim_type == "out_of_scope":
        return f"Expected behavior: {gt.get('expected_behavior', 'refuse')}. {gt.get('refusal_reason', '')}"
    elif claim_type == "sentiment":
        return f"Expected sentiment: {gt.get('label', '')}. {gt.get('rationale', '')}"
    return str(gt)


def _extract_ground_truth_anchors(claim: dict) -> list[dict]:
    """
    Extract filing-coordinate anchors from a claim for precision/recall@k.
    Returns list of {"accession": str, "section": str} dicts.
    """
    gt = claim.get("ground_truth") or {}
    claim_type = claim.get("claim_type", "")
    anchors = []

    def _extract_anchor(anchor_dict: dict) -> dict | None:
        acc = anchor_dict.get("accession")
        section = (anchor_dict.get("locator") or {}).get("section")
        if acc and section:
            return {"accession": acc, "section": section}
        return None

    if claim_type == "retrieval":
        for a in gt.get("relevant_anchors", []):
            extracted = _extract_anchor(a)
            if extracted:
                anchors.append(extracted)

    elif claim_type == "comparison":
        for key in ("current_anchor", "prior_anchor"):
            a = gt.get(key)
            if a:
                extracted = _extract_anchor(a)
                if extracted:
                    anchors.append(extracted)

    elif claim_type == "out_of_scope":
        a = gt.get("anchor")
        if a:
            extracted = _extract_anchor(a)
            if extracted:
                anchors.append(extracted)

    elif claim_type == "numeric":
        # Numeric facts come from XBRL, not filing sections.
        # precision/recall@k is not applicable for pure numeric claims.
        pass

    return anchors


def _run_pipeline(query: str, company: str, fiscal_label: str) -> dict[str, Any]:
    """
    Invoke the compiled LangGraph graph for one claim.

    Returns dict with:
        answer:   str — the pipeline's final answer
        contexts: list[str] — retrieved chunk texts
        chunks:   list[dict] — full chunk dicts (for precision/recall@k)
        error:    str | None
    """
    from graph.build_graph import compiled_graph
    from graph.state import GraphState

    initial_state: GraphState = {
        "company": company,
        "quarter": fiscal_label,          # GraphState uses 'quarter', not 'fiscal_label'
        "query": query,
        "comparison_quarters": [],
        "retrieval_results": [],
        "comparison_findings": [],
        "sentiment_scores": [],
        "numeric_validations": [],
        "report": "",
        "decision_log_entries": [],
        "error": None,
    }

    try:
        result = compiled_graph.invoke(initial_state)
        chunks = result.get("retrieval_results") or []
        contexts = [c.get("content", "") for c in chunks if isinstance(c, dict)]
        return {
            "answer": result.get("report") or "",
            "contexts": contexts,
            "chunks": chunks,
            "error": result.get("error"),
        }
    except Exception as e:
        log.warning("Pipeline error for query '%s': %s", query[:60], e)
        return {"answer": "", "contexts": [], "chunks": [], "error": str(e)}


def _parse_value(v: Any) -> float | None:
    """
    Parse numeric values like '$94 billion', '94,036', '94036' to float.
    All monetary values are normalised to USD millions to match filed values.
    """
    if v is None:
        return None
    s = str(v).strip().lower().replace(',', '').replace('$', '').strip()
    multiplier = 1.0
    if 'billion' in s:
        multiplier = 1_000.0   # convert billions → millions
        s = s.replace('billion', '').strip()
    elif 'million' in s:
        multiplier = 1.0
        s = s.replace('million', '').strip()
    try:
        return float(s) * multiplier
    except ValueError:
        return None


def _score_numeric(claim: dict, answer: str) -> dict[str, Any]:
    """
    Zero-tolerance numeric validation. Returns pass/fail + delta.
    Filed value comes from ground_truth.filed_value (USD millions).
    Stated value comes from payload.stated_value (may include units like '$94 billion').
    """
    gt = claim.get("ground_truth") or {}
    payload = claim.get("payload") or {}
    filed_raw = gt.get("filed_value")
    stated_raw = payload.get("stated_value")
    tolerance = gt.get("tolerance_rule", "exact")

    if filed_raw is None or stated_raw is None:
        return {"numeric_pass": None, "numeric_delta": None,
                "note": "missing filed_value or stated_value"}

    filed_f = _parse_value(filed_raw)
    stated_f = _parse_value(stated_raw)

    if filed_f is None or stated_f is None:
        return {"numeric_pass": None, "numeric_delta": None,
                "note": f"could not parse: filed={filed_raw} stated={stated_raw}"}

    delta = abs(filed_f - stated_f)

    if tolerance == "exact":
        passed = delta == 0.0
    elif isinstance(tolerance, str) and "abs<=" in tolerance:
        # e.g. "abs<=500M; exec rounded to nearest billion"
        try:
            band_str = tolerance.split("abs<=")[1].split(";")[0].strip()
            # Handle "500M" or "0.5pp" style bands
            band_str = band_str.lower().replace('m', '').replace('pp', '').strip()
            band = float(band_str)
            passed = delta <= band
        except (ValueError, IndexError):
            passed = False
    else:
        passed = delta == 0.0

    return {
        "numeric_pass": passed,
        "numeric_delta": round(delta, 4),
        "filed_value": filed_f,
        "stated_value": stated_f,
        "tolerance_rule": tolerance,
    }


def run_eval(
    claims_dir: Path,
    k: int = 5,
    run_name: str = "baseline",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Full evaluation loop. Returns aggregated metrics dict.
    """
    from evaluation.ragas_eval import run_ragas_eval
    from evaluation.precision_recall_at_k import compute_batch_retrieval_metrics
    from evaluation.llm_as_judge import judge_batch
    from observability.mlflow_tracking import start_run, log_eval_results, log_per_claim_results

    claims = _load_claims(claims_dir)
    runnable = [c for c in claims if c.get("claim_type") in _RUNNABLE_TYPES]
    log.info("%d/%d claims are runnable in Phase 1", len(runnable), len(claims))

    if dry_run:
        log.info("Dry run — schema check only, pipeline not invoked.")
        return {"dry_run": True, "total_claims": len(claims),
                "runnable_claims": len(runnable)}

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
            log.warning("Claim %s has no question — skipping", claim_id)
            continue

        log.info("Running claim %s (%s | %s/%s)", claim_id, claim_type,
                 company, fiscal_label)

        t0 = time.time()
        pipeline_out = _run_pipeline(query, company, fiscal_label)
        latency_ms = int((time.time() - t0) * 1000)

        time.sleep(3)  # Rate limit headroom between pipeline calls
        answer = pipeline_out["answer"]
        contexts = pipeline_out["contexts"]
        chunks = pipeline_out["chunks"]
        pipeline_error = pipeline_out["error"]

        # RAGAS sample (all runnable types)
        ragas_samples.append({
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

        # Retrieval metrics (retrieval + comparison + out_of_scope have anchors)
        gt_anchors = _extract_ground_truth_anchors(claim)
        if gt_anchors:
            retrieval_batch.append({
                "claim_id": claim_id,
                "retrieved_chunks": chunks,
                "ground_truth_anchors": gt_anchors,
            })

        # LLM-as-judge
        judge_samples.append({
            "claim_id": claim_id,
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "claim_type": claim_type,
        })

        # Numeric zero-tolerance check
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

    # ── Score aggregation ─────────────────────────────────────────────────────
    log.info("Scoring %d samples...", len(ragas_samples))

    ragas_scores = run_ragas_eval(ragas_samples, metrics=["faithfulness", "answer_relevancy"]) if ragas_samples else {}
    retrieval_scores = (
        compute_batch_retrieval_metrics(retrieval_batch, k=k)
        if retrieval_batch else {}
    )
    judge_scores_list, mean_judge = judge_batch(judge_samples) if judge_samples else ([], 0.0)

    # Numeric accuracy — zero tolerance
    numeric_pass_rate = 0.0
    if numeric_results:
        passed = sum(1 for r in numeric_results if r.get("numeric_pass") is True)
        numeric_pass_rate = round(passed / len(numeric_results), 4)
        log.info("Numeric pass rate: %.4f (%d/%d)", numeric_pass_rate,
                 passed, len(numeric_results))

    metrics = {
        **{f"ragas_{k_}": v for k_, v in ragas_scores.items()},
        f"precision_at_{k}": retrieval_scores.get("mean_precision_at_k", 0.0),
        f"recall_at_{k}": retrieval_scores.get("mean_recall_at_k", 0.0),
        "llm_judge_mean": mean_judge,
        "numeric_pass_rate": numeric_pass_rate,
        "total_claims": len(runnable),
        "pipeline_errors": sum(1 for r in per_claim_results if r["pipeline_error"]),
    }

    params = {
        "retrieval_k": k,
        "run_name": run_name,
        "phase": "1",
        "model": "gpt-5-mini",
        "embedding_model": "text-embedding-3-small",
        "claims_dir": str(claims_dir),
        "num_claims": len(runnable),
    }

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    with start_run(run_name=run_name, tags={"phase": "1", "variant": run_name}):
        log_eval_results(metrics=metrics, params=params)
        log_per_claim_results(per_claim_results)

    log.info("Baseline eval complete. Key metrics:")
    for name, val in metrics.items():
        if isinstance(val, float):
            log.info("  %s: %.4f", name, val)
        else:
            log.info("  %s: %s", name, val)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 baseline evaluation for QuarterLens AI."
    )
    parser.add_argument(
        "--claims-dir",
        default="golden_dataset/claims",
        help="Directory containing golden claim JSON files.",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Retrieval cutoff rank for precision@k / recall@k.",
    )
    parser.add_argument(
        "--run-name", default="baseline",
        help="MLflow run name for this evaluation.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate claim schema only — do not invoke the pipeline.",
    )
    args = parser.parse_args()

    metrics = run_eval(
        claims_dir=Path(args.claims_dir),
        k=args.k,
        run_name=args.run_name,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print("\n=== Baseline Eval Results ===")
        for name, val in metrics.items():
            if isinstance(val, float):
                print(f"  {name}: {val:.4f}")
            else:
                print(f"  {name}: {val}")


if __name__ == "__main__":
    main()