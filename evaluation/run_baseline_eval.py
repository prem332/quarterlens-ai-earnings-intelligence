"""
evaluation/run_baseline_eval.py

Baseline evaluation runner for QuarterLens AI.

Supports phased evaluation (cost control):
    --max-claims 5   → run 5 stratified claims
    --max-claims 25  → run 25 stratified claims
    --max-claims 75  → full golden dataset (default)

Stratified sampling ensures all claim types are represented even in small runs.

Phase 3 addition: retrieval error analysis integrated into every eval run.
For each claim with ground truth anchors, top-5 chunks are classified as:
  exact_match, same_accession, same_company, irrelevant
Duplicate density and adjacent chunk rate also computed and logged to MLflow.

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
    """
    Extract filing-coordinate anchors reachable by retrieval_results.

    retrieval_results is scoped to the claim's own fiscal_label — the pipeline
    only ever queries company/fiscal_label for the claim under test (see
    _run_pipeline). Comparison claims carry a current_anchor (same fiscal_label
    as the claim) and a prior_anchor (an earlier fiscal_label fetched
    separately by comparison_agent.fetch_prior_quarter and never merged into
    retrieval_results). Including prior_anchor here would structurally cap
    recall@k below 1.0 for every comparison claim, independent of retrieval
    quality, since no prior-quarter chunk can ever appear in retrieval_results.
    """
    gt = claim.get("ground_truth") or {}
    claim_type = claim.get("claim_type", "")
    claim_fiscal_label = claim.get("fiscal_label", "")
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
            # Only the anchor matching the claim's own fiscal_label is reachable
            # by retrieval_results — the prior-quarter anchor never is.
            if a and a.get("fiscal_label") == claim_fiscal_label:
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


# ── Retrieval error analysis helpers (Phase 3) ────────────────────────────────

def _classify_chunk(chunk: dict, gt_anchors: list[dict], claim: dict) -> str:
    """
    Classify a retrieved chunk against ground truth anchors.
    Returns: exact_match | same_accession | same_company | irrelevant
    """
    chunk_acc = chunk.get("accession", "").strip()
    chunk_sec = chunk.get("section", "").strip().lower()
    chunk_company = chunk.get("company", "").strip().upper()
    chunk_quarter = chunk.get("quarter", "").strip()

    claim_company = claim.get("company", "").strip().upper()
    claim_quarter = claim.get("fiscal_label", "").strip()

    gt_accs = {a["accession"].strip() for a in gt_anchors}
    gt_keys = {(a["accession"].strip(), a["section"].strip().lower()) for a in gt_anchors}

    if (chunk_acc, chunk_sec) in gt_keys:
        return "exact_match"
    if chunk_acc in gt_accs:
        return "same_accession"
    if chunk_company == claim_company and chunk_quarter == claim_quarter:
        return "same_company"
    return "irrelevant"


def _duplicate_density(chunks: list[dict]) -> float:
    """Fraction of top-k chunks sharing accession+section with another chunk."""
    if len(chunks) <= 1:
        return 0.0
    keys = [(c.get("accession", ""), c.get("section", "").lower()) for c in chunks]
    seen: set = set()
    dupes = 0
    for k in keys:
        if k in seen:
            dupes += 1
        seen.add(k)
    return round(dupes / len(chunks), 4)


def _adjacent_chunk_rate(chunks: list[dict]) -> float:
    """
    Fraction of chunk pairs with abs(chunk_index diff) <= 2 and same accession.
    High rate → MMR not diversifying within a document.
    """
    if len(chunks) <= 1:
        return 0.0
    pairs = 0
    adjacent = 0
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            pairs += 1
            if chunks[i].get("accession") == chunks[j].get("accession"):
                idx_i = chunks[i].get("chunk_index", -1)
                idx_j = chunks[j].get("chunk_index", -1)
                if idx_i >= 0 and idx_j >= 0 and abs(idx_i - idx_j) <= 2:
                    adjacent += 1
    return round(adjacent / pairs, 4) if pairs > 0 else 0.0


def _compute_chunk_pair_similarities(chunks: list[dict]) -> list[dict]:
    """
    Compute pairwise cosine similarities between top-k chunks for duplicate analysis.
    Uses the rerank_score as a proxy when embeddings aren't available.
    Returns list of pair dicts with classification.
    """
    pairs = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            a, b = chunks[i], chunks[j]
            same_acc = a.get("accession", "") == b.get("accession", "")
            same_sec = a.get("section", "").lower() == b.get("section", "").lower()
            same_key = same_acc and same_sec
            idx_dist = (
                abs(a.get("chunk_index", -1) - b.get("chunk_index", -1))
                if a.get("chunk_index", -1) >= 0 and b.get("chunk_index", -1) >= 0
                else -1
            )
            # Classify the duplicate pair type
            if same_key:
                if idx_dist >= 0 and idx_dist <= 2:
                    pair_type = "adjacent_same_section"
                else:
                    pair_type = "non_adjacent_same_section"
            elif same_acc:
                pair_type = "same_filing_diff_section"
            elif a.get("company") == b.get("company") and a.get("quarter") == b.get("quarter"):
                pair_type = "same_company_diff_filing"
            else:
                pair_type = "different_company"

            pairs.append({
                "rank_a": i + 1,
                "rank_b": j + 1,
                "chunk_id_a": a.get("chunk_id", ""),
                "chunk_id_b": b.get("chunk_id", ""),
                "accession_a": a.get("accession", ""),
                "accession_b": b.get("accession", ""),
                "section_a": a.get("section", ""),
                "section_b": b.get("section", ""),
                "doc_type_a": a.get("doc_type", ""),
                "doc_type_b": b.get("doc_type", ""),
                "chunk_index_a": a.get("chunk_index", -1),
                "chunk_index_b": b.get("chunk_index", -1),
                "chunk_index_distance": idx_dist,
                "same_accession": same_acc,
                "same_section": same_sec,
                "same_key": same_key,
                "pair_type": pair_type,
                "content_preview_a": a.get("content", "")[:200],
                "content_preview_b": b.get("content", "")[:200],
            })
    return pairs


def _build_detail_report(
    claim: dict,
    chunks: list[dict],
    gt_anchors: list[dict],
    classifications: list[str],
    answer: str,
    k: int,
) -> dict:
    """Build a full per-claim detail record for the --detail-report output."""
    top_k = chunks[:k]
    chunk_details = []
    for rank, (chunk, cls) in enumerate(zip(top_k, classifications), start=1):
        chunk_details.append({
            "rank":          rank,
            "classification": cls,
            "chunk_id":      chunk.get("chunk_id", ""),
            "accession":     chunk.get("accession", ""),
            "section":       chunk.get("section", ""),
            "doc_type":      chunk.get("doc_type", ""),
            "chunk_index":   chunk.get("chunk_index", -1),
            "chunk_total":   chunk.get("chunk_total", -1),
            "score":         round(float(chunk.get("score", 0.0)), 4),
            "content_preview": chunk.get("content", "")[:300],
        })

    return {
        "claim_id":            claim.get("claim_id", ""),
        "claim_type":          claim.get("claim_type", ""),
        "company":             claim.get("company", ""),
        "fiscal_label":        claim.get("fiscal_label", ""),
        "difficulty":          claim.get("difficulty", ""),
        "edge_case":           claim.get("edge_case", ""),
        "query":               _build_query(claim)[:200],
        "ground_truth_anchors": gt_anchors,
        "precision_at_k":      round(
            sum(1 for c in classifications if c == "exact_match") / k, 4
        ) if k else 0.0,
        "duplicate_density":   _duplicate_density(top_k),
        "adjacent_chunk_rate": _adjacent_chunk_rate(top_k),
        "chunks":              chunk_details,
        "chunk_pairs":         _compute_chunk_pair_similarities(top_k),
        "answer_preview":      answer[:300],
    }
    """
    Aggregate retrieval error classifications across all claims.
    Returns summary dict suitable for MLflow logging.
    """
    if not error_analysis_batch:
        return {}

    total_chunks = 0
    counts = {"exact_match": 0, "same_accession": 0, "same_company": 0, "irrelevant": 0}
    dup_densities = []
    adj_rates = []

    for item in error_analysis_batch:
        for cls in item["classifications"]:
            counts[cls] = counts.get(cls, 0) + 1
            total_chunks += 1
        dup_densities.append(item["duplicate_density"])
        adj_rates.append(item["adjacent_chunk_rate"])

    rates = {k: round(v / total_chunks, 4) if total_chunks else 0.0
             for k, v in counts.items()}

    dominant = max(
        {"same_accession": counts["same_accession"],
         "same_company": counts["same_company"],
         "irrelevant": counts["irrelevant"]},
        key=lambda k: {"same_accession": counts["same_accession"],
                       "same_company": counts["same_company"],
                       "irrelevant": counts["irrelevant"]}[k]
    )

    return {
        "retrieval_exact_match_rate":    rates["exact_match"],
        "retrieval_same_accession_rate": rates["same_accession"],
        "retrieval_same_company_rate":   rates["same_company"],
        "retrieval_irrelevant_rate":     rates["irrelevant"],
        "retrieval_duplicate_density":   round(sum(dup_densities) / len(dup_densities), 4),
        "retrieval_adjacent_chunk_rate": round(sum(adj_rates) / len(adj_rates), 4),
        "retrieval_dominant_failure":    dominant,
    }


def _compute_error_analysis(error_analysis_batch: list[dict]) -> dict:
    """
    Aggregate retrieval error classifications across all claims.
    Returns summary dict suitable for MLflow logging.
    """
    if not error_analysis_batch:
        return {}

    total_chunks = 0
    counts = {"exact_match": 0, "same_accession": 0, "same_company": 0, "irrelevant": 0}
    dup_densities = []
    adj_rates = []

    for item in error_analysis_batch:
        for cls in item["classifications"]:
            counts[cls] = counts.get(cls, 0) + 1
            total_chunks += 1
        dup_densities.append(item["duplicate_density"])
        adj_rates.append(item["adjacent_chunk_rate"])

    rates = {k: round(v / total_chunks, 4) if total_chunks else 0.0
             for k, v in counts.items()}

    dominant = max(
        {"same_accession": counts["same_accession"],
         "same_company": counts["same_company"],
         "irrelevant": counts["irrelevant"]},
        key=lambda k: {"same_accession": counts["same_accession"],
                       "same_company": counts["same_company"],
                       "irrelevant": counts["irrelevant"]}[k]
    )

    return {
        "retrieval_exact_match_rate":    rates["exact_match"],
        "retrieval_same_accession_rate": rates["same_accession"],
        "retrieval_same_company_rate":   rates["same_company"],
        "retrieval_irrelevant_rate":     rates["irrelevant"],
        "retrieval_duplicate_density":   round(sum(dup_densities) / len(dup_densities), 4),
        "retrieval_adjacent_chunk_rate": round(sum(adj_rates) / len(adj_rates), 4),
        "retrieval_dominant_failure":    dominant,
    }


# ── RAGAS per-claim-type aggregation (measurement correction) ─────────────────

# Claim types that carry filing-coordinate anchors — the exact population
# precision@5 is measured over (see _extract_ground_truth_anchors). Reporting
# context_precision on this subset makes it comparable to precision@5 instead of
# being diluted by numeric/sentiment claims whose terse categorical ground_truth
# ("Filed value: …", "Expected sentiment: …") has no chunk-level relevance signal.
_ANCHOR_CLAIM_TYPES = {"retrieval", "comparison", "out_of_scope"}


def _aggregate_ragas_by_type(
    samples: list[dict], per_sample: list[dict[str, float]]
) -> dict[str, float]:
    """
    Break RAGAS per-sample scores down by claim type + a retrieval-relevant subset.

    Uses the per-sample scores already computed by run_ragas_eval — no extra LLM
    calls. Returns a flat {metric_key: value} dict (floats + counts) for MLflow.
    """
    by_type: dict[str, dict[str, list[float]]] = {}
    subset: dict[str, list[float]] = {}

    for sample, ps in zip(samples, per_sample):
        ct = sample.get("claim_type", "unknown")
        for metric, val in ps.items():
            by_type.setdefault(ct, {}).setdefault(metric, []).append(val)
            if ct in _ANCHOR_CLAIM_TYPES:
                subset.setdefault(metric, []).append(val)

    out: dict[str, float] = {}
    for ct, metrics in by_type.items():
        for metric, vals in metrics.items():
            out[f"ragas_{metric}_{ct}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
        out[f"ragas_n_{ct}"] = len(next(iter(metrics.values()), []))

    for metric, vals in subset.items():
        out[f"ragas_{metric}_retrieval_subset"] = (
            round(sum(vals) / len(vals), 4) if vals else 0.0
        )
    out["ragas_n_retrieval_subset"] = len(next(iter(subset.values()), []))
    return out


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
        "transcript_retrieval_results": [],
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
    detail_report: bool = False,
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
    error_analysis_batch: list[dict] = []   # Phase 3: retrieval error classification
    detail_reports: list[dict] = []          # --detail-report: per-claim chunk detail

    for claim_idx, claim in enumerate(runnable, start=1):
        claim_id = claim.get("claim_id", str(uuid.uuid4()))
        claim_type = claim.get("claim_type", "retrieval")
        query = _build_query(claim)
        ground_truth = _build_ground_truth(claim)
        company = claim.get("company", "")
        fiscal_label = claim.get("fiscal_label", "")

        if not query:
            log.warning("Claim %s has no query — skipping", claim_id)
            continue

        log.info(
            "Claim %s (%s | %s/%s) [%d/%d — %d remaining]",
            claim_id, claim_type, company, fiscal_label,
            claim_idx, len(runnable), len(runnable) - claim_idx,
        )

        t0 = time.time()
        pipeline_out = await _run_pipeline(query, company, fiscal_label)
        latency_ms = int((time.time() - t0) * 1000)
        time.sleep(3)

        answer = pipeline_out["answer"]
        contexts = pipeline_out["contexts"]
        chunks = pipeline_out["chunks"]
        pipeline_error = pipeline_out["error"]

        ragas_samples.append({"question": query, "answer": answer,
                               "contexts": contexts, "ground_truth": ground_truth,
                               "claim_type": claim_type})

        gt_anchors = _extract_ground_truth_anchors(claim)
        if gt_anchors:
            retrieval_batch.append({"claim_id": claim_id, "retrieved_chunks": chunks,
                                    "ground_truth_anchors": gt_anchors})

            # Phase 3: classify each top-k chunk against ground truth
            top_k_chunks = chunks[:k]
            classifications = [
                _classify_chunk(c, gt_anchors, claim) for c in top_k_chunks
            ]
            error_analysis_batch.append({
                "claim_id":           claim_id,
                "claim_type":         claim_type,
                "classifications":    classifications,
                "duplicate_density":  _duplicate_density(top_k_chunks),
                "adjacent_chunk_rate": _adjacent_chunk_rate(top_k_chunks),
            })

            # --detail-report: build full per-claim chunk detail record
            if detail_report:
                detail_reports.append(_build_detail_report(
                    claim=claim,
                    chunks=chunks,
                    gt_anchors=gt_anchors,
                    classifications=classifications,
                    answer=answer,
                    k=k,
                ))

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

    if ragas_samples:
        ragas_scores, ragas_per_sample = run_ragas_eval(
            ragas_samples,
            metrics=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
            return_per_sample=True,
        )
        ragas_by_type = _aggregate_ragas_by_type(ragas_samples, ragas_per_sample)
        log.info(
            "context_precision — overall=%.4f retrieval_subset=%.4f",
            ragas_scores.get("context_precision", 0.0),
            ragas_by_type.get("ragas_context_precision_retrieval_subset", 0.0),
        )
    else:
        ragas_scores, ragas_by_type = {}, {}

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

    # Phase 3: aggregate retrieval error analysis
    error_analysis_metrics = _compute_error_analysis(error_analysis_batch)
    if error_analysis_metrics:
        log.info(
            "Retrieval error analysis: exact=%.3f same_acc=%.3f same_co=%.3f irrelevant=%.3f "
            "dup_density=%.3f adj_rate=%.3f dominant=%s",
            error_analysis_metrics.get("retrieval_exact_match_rate", 0),
            error_analysis_metrics.get("retrieval_same_accession_rate", 0),
            error_analysis_metrics.get("retrieval_same_company_rate", 0),
            error_analysis_metrics.get("retrieval_irrelevant_rate", 0),
            error_analysis_metrics.get("retrieval_duplicate_density", 0),
            error_analysis_metrics.get("retrieval_adjacent_chunk_rate", 0),
            error_analysis_metrics.get("retrieval_dominant_failure", "unknown"),
        )

    from azure_clients.redis_client import get_cache_stats
    cache_stats = get_cache_stats()

    # Separate string fields from numeric fields for MLflow compatibility
    error_analysis_numeric = {k: v for k, v in error_analysis_metrics.items()
                               if isinstance(v, (int, float))}
    error_analysis_string = {k: v for k, v in error_analysis_metrics.items()
                              if isinstance(v, str)}

    metrics = {
        **{f"ragas_{k_}": v for k_, v in ragas_scores.items()},
        **ragas_by_type,   # per-claim-type + retrieval-subset context_precision breakdown
        f"precision_at_{k}": retrieval_scores.get("mean_precision_at_k", 0.0),
        f"recall_at_{k}": retrieval_scores.get("mean_recall_at_k", 0.0),
        "llm_judge_mean": mean_judge,
        "numeric_pass_rate": numeric_pass_rate,
        "total_claims": len(runnable),
        "pipeline_errors": sum(1 for r in per_claim_results if r["pipeline_error"]),
        **{f"cache_{k_}": v for k_, v in cache_stats.items()},
        **error_analysis_numeric,   # Phase 3: numeric retrieval error metrics only
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
        **error_analysis_string,    # Phase 3: string fields (dominant_failure) go to params
    }

    with start_run(run_name=run_name, tags={"phase": "2", "variant": run_name}):
        log_eval_results(metrics=metrics, params=params)
        log_per_claim_results(per_claim_results)

    # Save detail report if requested
    if detail_report and detail_reports:
        # Summarize pair type distribution across all claims
        pair_type_counts: dict[str, int] = {}
        for dr in detail_reports:
            for pair in dr.get("chunk_pairs", []):
                pt = pair.get("pair_type", "unknown")
                pair_type_counts[pt] = pair_type_counts.get(pt, 0) + 1

        detail_output = {
            "run_name": run_name,
            "total_claims": len(detail_reports),
            "pair_type_summary": pair_type_counts,
            "claims": detail_reports,
        }
        detail_path = Path(f"evaluation/detail_report_{run_name}.json")
        detail_path.write_text(
            json.dumps(detail_output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Detail report saved to %s", detail_path)
        print(f"\nDetail report: {detail_path}")
        print(f"Pair type distribution: {pair_type_counts}")

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
    parser.add_argument(
        "--detail-report", action="store_true",
        help="Save per-claim chunk detail + duplicate pair analysis to JSON. "
             "Use with --max-claims 25 for cost control."
    )
    args = parser.parse_args()

    asyncio.run(run_eval(
        claims_dir=Path(args.claims_dir),
        k=args.k,
        run_name=args.run_name,
        dry_run=args.dry_run,
        max_claims=args.max_claims,
        seed=args.seed,
        detail_report=args.detail_report,
    ))


if __name__ == "__main__":
    main()