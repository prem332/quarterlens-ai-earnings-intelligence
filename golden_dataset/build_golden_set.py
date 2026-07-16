"""
CANNOT fabricate ground truth. Validates shape and referential integrity only:

  1. Schema validation       — every claim parses against schema.py
  2. Anchor integrity        — filing anchors resolve in parsed_manifest.json
  3. Section vocabulary      — filing anchor `section` is real for that form
                               (config/section_ids.json)
  4. Prior-quarter adjacency — comparison prior_anchor is genuinely earlier than
                               current_anchor for the SAME company, by report_date
                               (never fiscal_label string — NVDA runs a year ahead)
  5. Duplicate claim_ids
  6. Coverage report         — per-type actual vs FLOOR (minimums, not caps)

Anchor rules by type:
  retrieval    — >=1 anchor (filing and/or transcript)
  comparison   — current + prior anchor, date-adjacency enforced
  numeric      — no section anchor; filed value sourced by xbrl_tag + accession
                 (narrative/numeric split: XBRL facts never scraped from filing HTML)
  sentiment    — transcript anchor only
  out_of_scope — CONDITIONAL: `unanswerable` carries an anchor (the plausible-but-
                 insufficient source it must refuse from); `advice_bait` carries none

Usage:
    python build_golden_set.py            # validate + coverage report
    python build_golden_set.py --strict   # non-zero exit on error or below-floor

Facts inside claims are the labeler's responsibility and are NOT checked for
correctness — only for structural well-formedness.

"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from schema import ClaimType, GoldenSet, SourceType

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CLAIMS_DIR = HERE / "claims"
SECTION_CONFIG = HERE / "config" / "section_ids.json"
PARSED_MANIFEST = REPO / "data" / "parsed" / "parsed_manifest.json"

# --------------------------------------------------------------------------- #
# Locked per-type FLOORS — minimums, NOT caps. Floor total 60, target 75+.
# Exceeding a floor is welcome; only shortfalls are gaps.
# --------------------------------------------------------------------------- #

FLOORS = {
    "numeric": 15,
    "comparison": 15,
    "retrieval": 12,
    "out_of_scope": 10,
    "sentiment": 8,
}
FLOOR_TOTAL = sum(FLOORS.values())  # 60
TARGET_TOTAL = 75                   # aspiration, not a ceiling

# Advisory edge_case spread — reported for stressor visibility. NOT enforced;
# an unmet edge_case is not a gap.
EDGE_GUIDANCE = {
    "retrieval": ["direct", "terminology_mismatch", "multi_section",
                  "near_duplicate_boilerplate", "cross_document",
                  "temporal_disambiguation"],
    "comparison": ["guidance_language", "risk_factor_added", "risk_factor_dropped",
                   "hedging_shift", "magnitude_shift", "boilerplate_reword",
                   "reordering", "numeric_only_update", "synonym_swap",
                   "formatting_artifact"],
    "numeric": ["exact_match", "rounding_band", "unit_scale", "derived_metric",
                "mismatch_true", "period_mismatch"],
    "out_of_scope": ["unanswerable", "advice_bait"],
    "sentiment": ["hedged_positive", "negative_in_polite_framing",
                  "analyst_pushback", "neutral_boilerplate", "domain_inversion"],
}

# Comparison positive/negative balance — the over-flag guard only works if the
# no-shift class stays substantial. Warn (not error) outside this band.
SHIFT_BALANCE_MIN = 0.35
SHIFT_BALANCE_MAX = 0.65


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_manifest_index(manifest: list[dict]) -> dict:
    """
    by_key   : (cik, accession, fiscal_label, form) -> record  [filing existence]
    by_acc   : accession -> record                             [numeric source]
    by_period: (cik, fiscal_label) -> record                   [transcript resolve]
    """
    by_key, by_acc, by_period = {}, {}, {}
    for rec in manifest:
        by_key[(rec["cik"], rec["accession"], rec["fiscal_label"], rec["form"])] = rec
        by_acc[rec["accession"]] = rec
        by_period[(rec["cik"], rec["fiscal_label"])] = rec
    return {"by_key": by_key, "by_acc": by_acc, "by_period": by_period}


# --------------------------------------------------------------------------- #
# Anchor checks
# --------------------------------------------------------------------------- #

def check_anchor(anchor, mani, section_vocab, errors: list[str], ctx: str):
    """Existence + section vocab. No-op on None (out_of_scope advice_bait)."""
    if anchor is None:
        return

    if anchor.source_type == SourceType.TRANSCRIPT:
        if (anchor.cik, anchor.fiscal_label) not in mani["by_period"]:
            errors.append(
                f"{ctx}: transcript period not in parsed_manifest: "
                f"({anchor.cik}, {anchor.fiscal_label})"
            )
        return

    key = (anchor.cik, anchor.accession, anchor.fiscal_label, anchor.form.value)
    if key not in mani["by_key"]:
        errors.append(f"{ctx}: filing anchor not in parsed_manifest: {key}")
        return

    allowed = section_vocab.get(anchor.form.value, [])
    sec = anchor.locator.section
    if sec not in allowed:
        errors.append(
            f"{ctx}: section '{sec}' not valid for {anchor.form.value} "
            f"(allowed: {', '.join(allowed)})"
        )


def resolve_record(anchor, mani):
    """Resolve any anchor to its manifest record, for date comparison."""
    if anchor.source_type == SourceType.TRANSCRIPT:
        return mani["by_period"].get((anchor.cik, anchor.fiscal_label))
    return mani["by_key"].get(
        (anchor.cik, anchor.accession, anchor.fiscal_label, anchor.form.value)
    )


def check_prior_adjacency(current, prior, mani, errors: list[str], ctx: str):
    """
    prior must be earlier than current for the SAME company, by real report_date.
    Fiscal labels are not comparable across companies (NVDA runs a fiscal year
    ahead; MSFT's quarter numbering differs from AAPL's) and are unsafe even
    within a company across a fiscal-year boundary. Dates only.
    """
    cur_rec, pri_rec = resolve_record(current, mani), resolve_record(prior, mani)
    if cur_rec is None or pri_rec is None:
        errors.append(f"{ctx}: cannot resolve current/prior anchor for date check")
        return
    if cur_rec["cik"] != pri_rec["cik"]:
        errors.append(f"{ctx}: current and prior anchors are different companies")
        return
    if not (pri_rec["report_date"] < cur_rec["report_date"]):
        errors.append(
            f"{ctx}: prior_anchor ({pri_rec['report_date']}) is not earlier than "
            f"current_anchor ({cur_rec['report_date']})"
        )


# --------------------------------------------------------------------------- #
# Per-claim dispatch
# --------------------------------------------------------------------------- #

def validate_claim(claim, mani, section_vocab, errors: list[str]):
    ctx = f"[{claim.claim_id}]"
    ct = claim.claim_type

    if ct == ClaimType.RETRIEVAL:
        for a in claim.ground_truth.relevant_anchors:
            check_anchor(a, mani, section_vocab, errors, ctx)

    elif ct == ClaimType.COMPARISON:
        cur, pri = claim.ground_truth.current_anchor, claim.ground_truth.prior_anchor
        check_anchor(cur, mani, section_vocab, errors, ctx)
        check_anchor(pri, mani, section_vocab, errors, ctx)
        check_prior_adjacency(cur, pri, mani, errors, ctx)

    elif ct == ClaimType.NUMERIC:
        # No section anchor by design — filed value comes from XBRL, not filing HTML.
        acc = claim.ground_truth.source.accession
        if acc not in mani["by_acc"]:
            errors.append(f"{ctx}: numeric source accession not in parsed_manifest: {acc}")
        check_anchor(claim.ground_truth.transcript_anchor, mani, section_vocab, errors, ctx)

    elif ct == ClaimType.SENTIMENT:
        check_anchor(claim.ground_truth.anchor, mani, section_vocab, errors, ctx)

    elif ct == ClaimType.OUT_OF_SCOPE:
        # Conditional. The required/forbidden rule is enforced in schema.py;
        # here we validate the anchor only when one is present.
        check_anchor(claim.ground_truth.anchor, mani, section_vocab, errors, ctx)


# --------------------------------------------------------------------------- #
# Coverage — floors, not caps
# --------------------------------------------------------------------------- #

def coverage_report(claims) -> tuple[str, bool, list[str]]:
    by_type = Counter(c.claim_type.value for c in claims)
    by_type_edge = Counter((c.claim_type.value, c.edge_case.value) for c in claims)
    warnings: list[str] = []

    lines = [
        f"\nCoverage — {len(claims)} claims "
        f"(floor {FLOOR_TOTAL}, target {TARGET_TOTAL}+, no cap)",
        "=" * 62,
    ]
    all_floors_met = True

    for ctype, floor in FLOORS.items():
        have = by_type.get(ctype, 0)
        if have < floor:
            mark = f"SHORT by {floor - have}"
            all_floors_met = False
        elif have == floor:
            mark = "at floor"
        else:
            mark = f"+{have - floor} over floor"
        lines.append(f"\n{ctype:<14} {have:>3} / {floor:<3} floor   [{mark}]")
        for edge in EDGE_GUIDANCE.get(ctype, []):
            n = by_type_edge.get((ctype, edge), 0)
            lines.append(f"    {edge:<34} {n}{'  <- none yet' if n == 0 else ''}")

    # comparison shift balance — the over-flag guard's integrity
    comps = [c for c in claims if c.claim_type == ClaimType.COMPARISON]
    if comps:
        n_true = sum(1 for c in comps if c.ground_truth.expected_shift)
        frac = n_true / len(comps)
        lines.append(
            f"\ncomparison balance: {n_true} shift / {len(comps) - n_true} no-shift "
            f"({frac:.0%} positive)"
        )
        if not (SHIFT_BALANCE_MIN <= frac <= SHIFT_BALANCE_MAX):
            warnings.append(
                f"comparison class balance {frac:.0%} positive is outside "
                f"{SHIFT_BALANCE_MIN:.0%}-{SHIFT_BALANCE_MAX:.0%}; a skewed negative "
                f"class makes the shift metric easy to inflate."
            )

    total = len(claims)
    lines.append("\n" + "-" * 62)
    if total < FLOOR_TOTAL:
        lines.append(f"TOTAL {total} — below floor {FLOOR_TOTAL}.")
    elif total < TARGET_TOTAL:
        lines.append(f"TOTAL {total} — floors clear; target {TARGET_TOTAL}+ not yet reached.")
    else:
        lines.append(f"TOTAL {total} — at or above target {TARGET_TOTAL}. "
                     f"More claims welcome (no cap).")
    lines.append("Edge-case spread is advisory, not enforced.")

    return "\n".join(lines), all_floors_met, warnings


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="non-zero exit on any validation error or below-floor type")
    args = ap.parse_args()

    if not PARSED_MANIFEST.exists():
        print(f"ERROR: parsed_manifest not found at {PARSED_MANIFEST}", file=sys.stderr)
        return 2
    mani = build_manifest_index(load_json(PARSED_MANIFEST))
    section_vocab = {k: v for k, v in load_json(SECTION_CONFIG).items()
                     if not k.startswith("_")}

    files = sorted(CLAIMS_DIR.glob("*.json"))
    if not files:
        print(f"No claim files in {CLAIMS_DIR} yet. Schema + validator ready — "
              f"label claims into claims/*.json.")
        return 0

    all_claims, errors = [], []
    for fp in files:
        raw = load_json(fp)
        payload = raw if isinstance(raw, dict) and "claims" in raw else {"claims": raw}
        try:
            gs = GoldenSet.model_validate(payload)
        except ValidationError as e:
            errors.append(f"{fp.name}: schema validation failed:\n{e}")
            continue
        for c in gs.claims:
            validate_claim(c, mani, section_vocab, errors)
        all_claims.extend(gs.claims)

    for cid, n in Counter(c.claim_id for c in all_claims).items():
        if n > 1:
            errors.append(f"duplicate claim_id: {cid} ({n}x)")

    print(f"Loaded {len(all_claims)} claims from {len(files)} file(s).")
    if errors:
        print(f"\n{len(errors)} VALIDATION ERROR(S):\n" + "-" * 62)
        for e in errors:
            print(f"  - {e}")
    else:
        print("All structural + referential checks passed.")

    report, floors_met, warnings = coverage_report(all_claims)
    print(report)
    if warnings:
        print("\nWARNINGS (not failures):")
        for w in warnings:
            print(f"  ! {w}")

    if args.strict and (errors or not floors_met):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())