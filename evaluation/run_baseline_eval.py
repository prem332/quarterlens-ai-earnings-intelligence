"""
evaluation/run_baseline_eval.py

Baseline evaluation runner for QuarterLens AI.

Supports phased evaluation (cost control):
    --max-claims 5   → run 5 stratified claims
    --max-claims 25  → run 25 stratified claims
    --max-claims 75  → full golden dataset (default)

Stratified sampling ensures all claim types are represented even in small runs.

Usage:
    python evaluation/run_baseline_eval.py --dry-run
    python evaluation/run_baseline_eval.py --max-claims 5 --run-name baseline-recursive-5
    python evaluation/run_baseline_eval.py --max-claims 25 --run-name baseline-recursive-25
    python evaluation/run_baseline_eval.py --run-name baseline-recursive-v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Observability — must initialize before openai_client is imported ──────────
_obs_log = logging.getLogger("observability")
try:
    from observability.langfuse_setup import setup_langfuse
    setup_langfuse()
    _obs_log.info("Langfuse initialized")
except Exception as _lf_exc:
    _obs_log.warning("Langfuse init failed (non-fatal): %s", _lf_exc)

try:
    from observability.phoenix_setup import setup_phoenix
    setup_phoenix()
    _obs_log.info("Phoenix initialized")
except Exception as _px_exc:
    _obs_log.warning("Phoenix init failed (non-fatal): %s", _px_exc)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("run_baseline_eval")

_RUNNABLE_TYPES = {"retrieval", "comparison", "numeric", "out_of_scope", "sentiment"}

# Per-type minimum for stratified sampling
_TYPE_WEIGHTS = {
    "numeric":      0.35,  # 26/75
    "retrieval":    0.20,
    "comparison":   0.17,
    "sentiment":    0.15,
    "out_of_scope": 0.13,
}


# ── Stratified sampling ───────────────────────────────────────────────────────

def _stratified_sample(claims: list[dict], n: int, seed: int = 42) -> list[dict]:
    """
    Return n claims sampled proportionally across claim types.
    Ensures all claim types present in small runs.
    Fixed seed for reproducibility.
    """
    if n >= len(claims):
        return claims

    random.seed(seed)
    by_type: dict[str, list[dict]] = {}
    for c in claims:
        ct = c.get("claim_type", "retrieval")
        by_type.setdefault(ct, []).append(c)

    # Shuffle each bucket
    for bucket in by_type.values():
        random.shuffle(bucket)

    # Allocate slots proportionally, min 1 per type present
    result: list[dict] = []
    remaining = n
    types = list(by_type.keys())

    # First pass: allocate proportionally
    alloc: dict[str, int] = {}
    for ct in types:
        weight = _TYPE_WEIGHTS.get(ct, 1.0 / len(types))
        alloc[ct] = max(1, round(n * weight))

    # Adjust to hit exactly n
    total_alloc = sum(alloc.values())
    diff = n - total_alloc
    if diff != 0:
        # Add/remove from the largest bucket
        largest = max(alloc, key=lambda k: alloc[k])
        alloc[largest] += diff

    for ct, count in alloc.items():
        bucket = by_type.get(ct, [])
        result.extend(bucket[:count])

    random.shuffle(result)
    return result[:n]


# ── Claim helpers ─────────────────────────────────────────────────────────────

def _load_claims(claims_dir: Path) -> list[dict]:
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


def _extract_ground_truth_anchors(claim: dict) -> list[dict]:
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
            e = _extract_anchor(a)
            if e: anchors.append(e)
    elif claim_type == "comparison":
        for key in ("current_anchor", "prior_anchor"):
            a = gt.get(key)
            if a:
                e = _extract_anchor(a)
                if e: anchors.append(e)
    elif claim_type == "out_of_scope":
        a = gt.get("anchor")
        if a:
            e = _extract_anchor(a)
            if e: anchors.append(e)
    return anchors


# ── Numeric scoring (improved — handles %, pp, floor statements, unit normalization) ──

_FLOOR_WORDS = ("exceeded", "over", "more than", "greater than", "at least", "north of", "above")


def _parse_value(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip().lower().replace(',', '').strip()
    if '/' in s:
        s = s.split('/')[0].strip()
    s = s.replace('%', '').replace('$', '').strip()
    multiplier = 1.0
    if 'billion' in s:
        multiplier = 1_000.0
        s = s.replace('billion', '').strip()
    elif 'million' in s:
        s = s.replace('million', '').strip()
    m = re.search(r'-?\d+\.?\d*', s)
    if not m:
        return None
    try:
        return float(m.group()) * multiplier
    except ValueError:
        return None


def _score_numeric(claim: dict, answer: str) -> dict:
    gt = claim.get("ground_truth") or {}
    payload = claim.get("payload") or {}
    filed_raw = gt.get("filed_value")
    stated_raw = payload.get("stated_value")
    tolerance = gt.get("tolerance_rule", "exact")
    unit = str(gt.get("unit", "")).lower()

    if filed_raw is None or stated_raw is None:
        return {"numeric_pass": None, "numeric_delta": None,
                "filed_value": filed_raw, "stated_value": stated_raw,
                "tolerance_rule": tolerance, "note": "missing filed_value or stated_value"}

    stated_str = str(stated_raw).lower()
    is_floor_statement = any(w in stated_str for w in _FLOOR_WORDS)

    filed_f = _parse_value(filed_raw)
    stated_f = _parse_value(stated_raw)

    # Unit normalization: bare billions (e.g. 3.54 in "billion people" unit)
    if "billion" in unit and filed_f is not None and abs(filed_f) < 1000:
        filed_f = filed_f * 1_000.0

    if filed_f is None or stated_f is None:
        return {"numeric_pass": None, "numeric_delta": None,
                "filed_value": filed_raw, "stated_value": stated_raw,
                "tolerance_rule": tolerance,
                "note": f"could not parse: filed={filed_raw} stated={stated_raw}"}

    delta = abs(filed_f - stated_f)

    if is_floor_statement and str(tolerance).strip().startswith("exact"):
        return {"numeric_pass": False, "numeric_delta": round(delta, 4),
                "filed_value": filed_f, "stated_value": stated_f,
                "tolerance_rule": tolerance, "note": "floor statement not accepted as exact"}

    tol_str = str(tolerance).lower()
    if tol_str.startswith("exact"):
        passed = delta == 0.0
    elif "abs<=" in tol_str:
        stated_portion = str(stated_raw).split('/')[0].strip()
        stated_is_pct = ("%" in stated_portion) and not any(
            u in stated_portion.lower() for u in ("billion", "million", "$")
        )
        band = None
        for m in re.finditer(r"abs<=\s*([\d.]+)\s*(pp|m)?", tol_str):
            num = float(m.group(1))
            kind = m.group(2)
            if stated_is_pct and kind == "pp":
                band = num
                break
            if (not stated_is_pct) and kind == "m":
                band = num
                break
            if band is None:
                band = num
        passed = band is not None and delta <= band
    else:
        passed = delta == 0.0

    return {"numeric_pass": passed, "numeric_delta": round(delta, 4),
            "filed_value": filed_f, "stated_value": stated_f, "tolerance_rule": tolerance}


# ── Pipeline runner ───────────────────────────────────────────────────────────

async def _run_pipeline(query: str, company: str, fiscal_label: str) -> dict[str, Any]:
    from graph.build_graph import compiled_graph
    from graph.state import GraphState
    from azure_clients.redis_client import get_report_cached, set_report_cached

    cached_report = get_report_cached(query, company, fiscal_label)
    if cached_report:
        return {"answer": cached_report, "contexts": [], "chunks": [],
                "error": None, "cache_hit": True}

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
        "model_tier": "primary",
        "report_model_tier": "primary",
        "error": None,
    }

    try:
        result = await compiled_graph.ainvoke(initial_state)
        chunks = result.get("retrieval_results") or []
        contexts = [c.get("content", "") for c in chunks if isinstance(c, dict)]
        report = result.get("report") or ""
        if report and not result.get("error"):
            set_report_cached(query, company, fiscal_label, report)
        return {"answer": report, "contexts": contexts, "chunks": chunks,
                "error": result.get("error"), "cache_hit": False}
    except Exception as e:
        log.warning("Pipeline error for query '%s': %s", query[:60], e)
        return {"answer": "", "contexts": [], "chunks": [], "error": str(e), "cache_hit": False}


# ── Main eval loop ────────────────────────────────────────────────────────────

async def run_eval(
    claims_dir: Path,
    k: int = 5,
    run_name: str = "baseline",
    dry_run: bool = False,
    max_claims: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    from evaluation.ragas_eval import run_ragas_eval
    from evaluation.precision_recall_at_k import compute_batch_retrieval_metrics
    from evaluation.llm_as_judge import judge_batch
    from observability.mlflow_tracking import start_run, log_eval_results, log_per_claim_results

    claims = _load_claims(claims_dir)
    runnable = [c for c in claims if c.get("claim_type") in _RUNNABLE_TYPES]
    log.info("%d/%d claims runnable", len(runnable), len(claims))

    # Stratified sampling if max_claims specified
    if max_claims and max_claims < len(runnable):
        runnable = _stratified_sample(runnable, max_claims, seed=seed)
        log.info("Stratified sample: %d claims selected", len(runnable))
        # Log type distribution
        by_type: dict[str, int] = {}
        for c in runnable:
            ct = c.get("claim_type", "unknown")
            by_type[ct] = by_type.get(ct, 0) + 1
        log.info("Sample distribution: %s", by_type)

    if dry_run:
        log.info("Dry run — pipeline not invoked.")
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
        pipeline_out = await _run_pipeline(query, company, fiscal_label)
        latency_ms = int((time.time() - t0) * 1000)
        time.sleep(3)

        answer = pipeline_out["answer"]
        contexts = pipeline_out["contexts"]
        chunks = pipeline_out["chunks"]
        pipeline_error = pipeline_out["error"]

        ragas_samples.append({"question": query, "answer": answer,
                               "contexts": contexts, "ground_truth": ground_truth})

        gt_anchors = _extract_ground_truth_anchors(claim)
        if gt_anchors:
            retrieval_batch.append({"claim_id": claim_id, "retrieved_chunks": chunks,
                                    "ground_truth_anchors": gt_anchors})

        judge_samples.append({"claim_id": claim_id, "question": query, "answer": answer,
                               "contexts": contexts, "ground_truth": ground_truth,
                               "claim_type": claim_type})

        if claim_type == "numeric":
            num_result = _score_numeric(claim, answer)
            num_result["claim_id"] = claim_id
            numeric_results.append(num_result)

        per_claim_results.append({
            "claim_id": claim_id, "claim_type": claim_type,
            "company": company, "fiscal_label": fiscal_label,
            "latency_ms": latency_ms, "pipeline_error": pipeline_error,
            "answer_length": len(answer),
        })

    # ── Scoring ───────────────────────────────────────────────────────────────
    log.info("Scoring %d samples...", len(ragas_samples))

    ragas_scores = run_ragas_eval(
        ragas_samples,
        metrics=["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ) if ragas_samples else {}

    retrieval_scores = (
        compute_batch_retrieval_metrics(retrieval_batch, k=k)
        if retrieval_batch else {}
    )
    judge_scores_list, mean_judge = judge_batch(judge_samples) if judge_samples else ([], 0.0)

    numeric_pass_rate = 0.0
    if numeric_results:
        passed = sum(1 for r in numeric_results if r.get("numeric_pass") is True)
        numeric_pass_rate = round(passed / len(numeric_results), 4)
        log.info("Numeric pass rate: %.4f (%d/%d)", numeric_pass_rate, passed, len(numeric_results))

    from azure_clients.redis_client import get_cache_stats
    cache_stats = get_cache_stats()

    metrics = {
        **{f"ragas_{k_}": v for k_, v in ragas_scores.items()},
        f"precision_at_{k}": retrieval_scores.get("mean_precision_at_k", 0.0),
        f"recall_at_{k}": retrieval_scores.get("mean_recall_at_k", 0.0),
        "llm_judge_mean": mean_judge,
        "numeric_pass_rate": numeric_pass_rate,
        "total_claims": len(runnable),
        "pipeline_errors": sum(1 for r in per_claim_results if r["pipeline_error"]),
        **{f"cache_{k_}": v for k_, v in cache_stats.items()},
    }

    params = {
        "retrieval_k": k,
        "run_name": run_name,
        "phase": "2",
        "model": "gpt-5.4-mini",
        "chunking": "recursive",
        "embedding_model": "text-embedding-3-small",
        "claims_dir": str(claims_dir),
        "num_claims": len(runnable),
        "max_claims_filter": max_claims or "all",
        "stratified_seed": seed,
    }

    with start_run(run_name=run_name, tags={"phase": "2", "variant": run_name}):
        log_eval_results(metrics=metrics, params=params)
        log_per_claim_results(per_claim_results)

    print("\n" + "=" * 55)
    print(f"EVAL COMPLETE — {run_name}")
    print("=" * 55)
    for name, val in metrics.items():
        if isinstance(val, float):
            print(f"  {name}: {val:.4f}")
        else:
            print(f"  {name}: {val}")
    print("=" * 55)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation for QuarterLens AI."
    )
    parser.add_argument("--claims-dir", default="golden_dataset/claims")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--run-name", default="baseline-recursive-v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-claims", type=int, default=None,
        help="Limit to N stratified claims for cost control (e.g. 5, 10, 25, 50, 75)."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asyncio.run(run_eval(
        claims_dir=Path(args.claims_dir),
        k=args.k,
        run_name=args.run_name,
        dry_run=args.dry_run,
        max_claims=args.max_claims,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()