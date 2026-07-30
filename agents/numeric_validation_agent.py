"""

Two-step process:
  1. LLM extracts numeric claims from transcript chunks (what did the CEO say?).
  2. calculate_metric() computes the same metric from SQL financial_facts deterministically.
  3. Agent compares the two — match/mismatch, delta %.

The LLM is ONLY used in step 1 (claim extraction from natural language).
All arithmetic is done by calculate_metric() — never by the LLM.
Zero-tolerance accuracy target per ARCHITECTURE.md §7.

Tools: calculate_metric(statement_data, formula) — deterministic SQL/Python calculation
"""

import json
import time
from graph.state import GraphState, DecisionLogEntry, NumericValidation
from agents._common import ms, skipped
from tools.calculate_metric import calculate_metric
from tools.fetch_prior_quarter import _parse_fiscal_label
from azure_clients.openai_client import openai_client


_CLAIM_EXTRACTION_PROMPT = """\
You are a financial data extraction assistant. Your output feeds a deterministic
verifier (calculate_metric) that can ONLY check claims against standard GAAP
line items filed in SEC XBRL. It cannot verify segment/product-level figures,
operational KPIs, or qualitative statements — there is no point extracting
those, since every one of them will fail to verify for a reason that has
nothing to do with whether the claim is true.

Extract a claim ONLY if it states a specific number for one of these GAAP
concepts (or its YoY/QoQ/constant-currency growth):
  total revenue, cost of revenue, gross profit/margin, operating income/margin,
  operating expenses, net income/margin, EPS (diluted or basic), R&D expense,
  SG&A expense, cash & equivalents, total assets, total liabilities,
  stockholders' equity, shares outstanding.

Do NOT extract claims about:
  - Segment or product-level figures of ANY kind — revenue, margin, or growth.
    "Azure grew 30%", "iPhone revenue was $46B", "Data Center revenue",
    "Products gross margin was 35.9%", "the Q2 Services margin" — none of these
    are consolidated GAAP line items and calculate_metric cannot resolve them.
    Only the COMPANY-WIDE consolidated figure qualifies.
  - Changes, deltas, and growth rates — "up 340 basis points sequentially",
    "grew 5% year over year", "margin improved 110 bps". The verifier checks a
    single filed value for ONE period and is given no prior period to difference
    against, so every change claim fails for a reason unrelated to its truth.
    Extract the LEVEL if one is also stated ("gross margin was 47.1%, up 20 bps"
    -> extract 47.1 as gross_margin); otherwise skip the sentence.
  - Operational KPIs (bookings, ARR, RPO, DAU/MAU, seats, users) — not filed
    in XBRL at all.
  - Statements where the NUMBER ITSELF is not one of the GAAP concepts above
    (inventory posture, a count of product models, a tariff rate, "13%
    Services growth" — Services is a segment, not the consolidated figure)
    — nothing to verify against a structured fact, regardless of topic.
  - Forward-looking guidance for a FUTURE period (e.g. "next quarter we
    expect...").

IMPORTANT — do not confuse "has qualitative color attached" with "not
extractable": a sentence stating an actual GAAP figure for the CURRENT/PAST
period is extractable even if it also explains WHY, or compares it to prior
guidance. Extract the figure; ignore the surrounding color.
  - Extractable: "Gross margin was 46.5%, at the high end of our guidance
    range, driven by favorable costs." -> gross_margin = 0.465. The
    explanation doesn't disqualify the actual reported figure.
  - NOT extractable: "We had some build-ahead inventory within our supply
    chain." -> no GAAP concept is stated at all, regardless of phrasing.

If nothing in the excerpt meets the bar above, return an empty array — that
is a correct, useful answer, not a failure.

For each claim that DOES qualify, identify:
  - "claim": exact quoted phrase containing the number
  - "metric": short snake_case identifier (e.g. revenue_growth_yoy, gross_margin, eps_diluted)
  - "claimed_value": the number EXACTLY as written, as a float. Do NOT convert,
    rescale, or do any arithmetic on it:
      "49.3%"           -> 49.3
      "340 basis points"-> 340
      "$95.4 billion"   -> 95.4
      "$44,867 million" -> 44867
    The verifier rescales deterministically using "claimed_unit"; converting
    here corrupts that and produces false mismatches.
  - "claimed_unit": the unit the number is STATED in. Exactly one of:
      "percent"       — "49.3%", "up 62 percent"
      "basis_points"  — "340 basis points", "70 bps"
      "usd"           — "$1.65", "EPS of $1.65"
      "usd_thousands" — "$4,300 thousand"
      "usd_millions"  — "$44,867 million"
      "usd_billions"  — "$95.4 billion"
      "ratio"         — a bare multiplier with no unit, e.g. "1.4x"
  - "value_type": "percentage" | "absolute" | "ratio"

Respond ONLY with a JSON array. No preamble, no markdown fences."""


def numeric_validation_agent(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    t0 = time.time()
    company = state["company"]
    quarter = state["quarter"]
    retrieval_results = state.get("retrieval_results") or []

    # Step 1: pull transcript chunks to extract claims from.
    #
    # Source is transcript_retrieval_results (the dedicated transcript pool),
    # not retrieval_results. Scavenging the merged top-5 for doc_type=transcript
    # is unreliable by construction: top-5 is the globally reranked filing+
    # transcript mix, so it frequently contains ZERO transcript chunks (observed
    # on NVDA_FY2026-Q3_cmp_001 and _cmp_004), in which case claim extraction
    # got an empty string and this agent returned nothing at all. Same bug class
    # as sentiment_agent's — graph/state.py defines
    # transcript_retrieval_results as the transcript pool for exactly this.
    #
    # Falls back to the old path so behavior degrades rather than breaks if the
    # field is ever absent (e.g. state built by an older caller).
    transcript_pool = state.get("transcript_retrieval_results") or retrieval_results
    transcript_text = _concat_transcript(transcript_pool)
    if not transcript_text.strip():
        return _empty("no transcript content for claim extraction", t0)

    # Step 2: extract claims via LLM
    raw_claims = _extract_claims(transcript_text)
    if not raw_claims:
        return _empty("no numeric claims extracted from transcript", t0)

    # Step 3: validate each claim against SQL financial_facts
    validations: list[NumericValidation] = []
    tokens_used = 0
    period = _normalize_period(quarter)

    skipped_unverifiable = 0

    for claim_obj in raw_claims:
        claimed_metric = claim_obj.get("metric", "")
        claimed_value = claim_obj.get("claimed_value")
        claimed_unit = (claim_obj.get("claimed_unit") or "").strip().lower()

        # Mirror of the calculated_value=None guard below, other side of the
        # same bug class: a claim the model extracted with no actual number
        # ("you can see that in the OpEx numbers") comes back with
        # claimed_value=None. calculate_metric can still resolve a real filed
        # figure for the metric name, so without this guard the entry was
        # appended as calculated=<real value>, claimed=None, match=False — a
        # ✗ next to a quote that never stated a number to check in the first
        # place. Skip before even calling calculate_metric; there is nothing
        # to compare regardless of what it returns.
        if claimed_value is None:
            print(f"[numeric_validation_agent] no claimed value extracted, skipping '{claimed_metric}'")
            skipped_unverifiable += 1
            continue

        try:
            calc_result = calculate_metric(
                company=company,
                fiscal_label=period,
                metric=claimed_metric,
                prior_fiscal_label=None,  # growth metrics need prior period — not available from transcript extraction
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[numeric_validation_agent] calculate_metric failed for {claimed_metric}: {exc}")
            skipped_unverifiable += 1
            continue

        calculated_value = calc_result.get("value")
        calc_unit = calc_result.get("unit") or ""

        # ── Emit ONLY claims that were actually checked ───────────────────
        # "Couldn't check it" is not "the executive was wrong". These used to
        # be appended with calculated_value=None / match=False, which the UI
        # renders as a ✗ next to the quote — an accusation against a figure
        # nobody verified, and the reason a run could show a 0% pass rate with
        # an empty Filed column on every row. Segment metrics, growth metrics
        # (no prior period is passed), and unaliased metrics all land here.
        if calculated_value is None:
            print(f"[numeric_validation_agent] unverifiable, skipping '{claimed_metric}': {calc_result.get('error')}")
            skipped_unverifiable += 1
            continue

        # A basis-point figure is always a CHANGE. If the metric resolved to a
        # LEVEL (unit "%") rather than a change (unit "pp"), the two describe
        # different quantities and the comparison is a category error that
        # always reports a ~100% delta — e.g. "margin improved 110 bps"
        # (1.1 pp) scored against a filed 49.27% level. The prompt asks the
        # model to skip change claims; it does not reliably comply, so this is
        # the deterministic backstop.
        if claimed_unit == "basis_points" and not calc_unit.lower().startswith("pp"):
            print(f"[numeric_validation_agent] delta-vs-level mismatch, skipping '{claimed_metric}'")
            skipped_unverifiable += 1
            continue

        # Rescale onto calculate_metric's own unit before comparing —
        # the two sides use different conventions by design.
        claimed_value = _rescale_claimed(claimed_value, claimed_unit, calc_unit)
        match, delta_pct = _compare(claimed_value, calculated_value, claim_obj.get("value_type"))

        validations.append(NumericValidation(
            claim=str(claim_obj.get("claim", "")),
            metric=claimed_metric,
            claimed_value=claimed_value,
            calculated_value=calculated_value,
            match=match,
            delta_pct=delta_pct,
            source_fiscal_label=period,
        ))

    mismatches = sum(1 for v in validations if not v["match"])
    entry: DecisionLogEntry = {
        "agent": "numeric_validation_agent",
        "tool_called": "calculate_metric",
        "input_summary": f"company={company} quarter={quarter} claims={len(raw_claims)}",
        "output_summary": (
            f"{len(validations)} validated, {mismatches} mismatches, "
            f"{skipped_unverifiable} unverifiable"
        ),
        "confidence": 1.0 if mismatches == 0 else round(1 - mismatches / max(len(validations), 1), 2),
        "tokens_used": tokens_used,
        "latency_ms": ms(t0),
    }

    return {
        "numeric_validations": validations,
        "decision_log_entries": [entry],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

# Factors that convert a claimed value into the scale calculate_metric returns.
#
# calculate_metric emits percentage-family values in PERCENTAGE POINTS
# (gross_margin -> 47.0506, unit "%"; margin deltas -> unit "pp") and
# currency-family values in RAW USD (revenue -> 95359000000.0, unit "USD";
# eps -> 1.65, unit "USD/shares"). The extraction prompt asks for the number as
# literally written. These bridge the two.
_PERCENT_POINT_SCALE = {
    "percent":      1.0,     # "49.3%"            -> 49.3 pp
    "basis_points": 0.01,    # "340 basis points" ->  3.4 pp
    "ratio":        100.0,   # 0.493              -> 49.3 pp
}
_USD_SCALE = {
    "usd":           1.0,
    "usd_thousands": 1e3,
    "usd_millions":  1e6,    # "$44,867 million"  -> 4.4867e10
    "usd_billions":  1e9,    # "$95.4 billion"    -> 9.54e10
}


def _rescale_claimed(
    claimed: float | None,
    claimed_unit: str | None,
    calc_unit: str | None,
) -> float | None:
    """
    Put the claimed value on the same scale as calculate_metric's value.

    Root cause of a confirmed false mismatch: the agent read calculate_metric's
    `value` and threw away its `unit`, so a claim of "the gross margin of 49.3%"
    (extracted as 0.493 under the old decimal-fraction instruction) was compared
    against a filed 47.0506 — a 98.95% "delta" reported as the executive
    misstating a figure that was actually right to within 2.3 points. Verified
    against the only stored run in blob history that ever produced both values.

    Unknown/absent unit falls through at 1.0 — with the current prompt that is
    the correct identity for `percent`, the dominant case, and never invents a
    conversion the model did not state.
    """
    if claimed is None:
        return None
    unit = (claimed_unit or "").strip().lower()
    calc = (calc_unit or "").strip().lower()
    if calc.startswith("%") or calc.startswith("pp"):
        return claimed * _PERCENT_POINT_SCALE.get(unit, 1.0)
    if calc.startswith("usd"):
        return claimed * _USD_SCALE.get(unit, 1.0)
    return claimed


def _normalize_period(quarter: str) -> str:
    """
    Canonicalize the STATE's quarter into the "FY2025-Q2" format
    calculate_metric/financial_facts key on.

    The period deliberately does NOT come from the LLM any more. The transcript
    pool this agent extracts from is retrieved with an OData filter of
    `fiscal_label eq '<quarter>'`, so every chunk is from that quarter by
    construction — the model has strictly less information than the state does,
    and was measurably getting it wrong: on AAPL FY2026-Q2 all 12 retrieved
    chunks carried fiscal_label "FY2026-Q2" and the model still reported
    "FY2025-Q2". The claim "the gross margin of 49.3%" then got checked against
    FY2025-Q2's filed 47.0506 instead of FY2026-Q2's 49.2706 and was reported as
    a mismatch — a false accusation against a figure that was accurate to 0.06%.

    (An earlier bug in the same spot: the prompt suggested "Q2_FY2025", the model
    echoed that format, and every lookup silently found nothing. Canonicalizing
    here guards the format regardless of who supplies the label.)
    """
    try:
        q_idx, year = _parse_fiscal_label(quarter)
        return f"FY{year}-Q{q_idx + 1}"
    except ValueError:
        return quarter


def _concat_transcript(retrieval_results: list, max_chars: int = 6000) -> str:
    """
    Concatenate transcript chunk text up to a character budget.

    Skips (does not stop at) a chunk that would overflow the budget. This used
    to `break`, which meant a single oversized chunk at the front discarded
    every chunk behind it — and the restored transcript chunking has chunks of
    6,974-16,256 chars, so the FIRST chunk routinely blew a 6,000-char budget
    and this returned "". Claim extraction then had no text and the agent
    produced zero validations. Measured on the seed-42 sample: 3 of 3 numeric
    claims got an empty string despite each having 12 transcript chunks
    available, most of them comfortably small (771-2,615 chars).
    """
    parts: list[str] = []
    total = 0
    for r in retrieval_results:
        if r.get("doc_type", "").lower() not in ("transcript", "earnings_call"):
            continue
        text = r.get("content", "")
        if not text or total + len(text) > max_chars:
            continue
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def _extract_claims(transcript_text: str) -> list[dict]:
    try:
        response = openai_client.chat(
            messages=[
                {"role": "system", "content": _CLAIM_EXTRACTION_PROMPT},
                {"role": "user", "content": transcript_text},
            ],
            max_completion_tokens=1024,
        )
        raw = response.choices[0].message.content or "[]"
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[numeric_validation_agent] claim JSON parse failed: {exc}")
        return []
    except Exception as exc:
        print(f"[numeric_validation_agent] claim extraction LLM failed: {exc}")
        return []


def _compare(
    claimed: float | None,
    calculated: float | None,
    value_type: str | None,
) -> tuple[bool, float | None]:
    """Returns (match, delta_pct). Tolerance: 0.5% for percentages, 1% for absolutes."""
    if claimed is None or calculated is None:
        return False, None
    if calculated == 0:
        return claimed == 0, None
    delta_pct = abs(claimed - calculated) / abs(calculated) * 100
    tolerance = 0.5 if value_type == "percentage" else 1.0
    return delta_pct <= tolerance, round(delta_pct, 4)


def _empty(reason: str, t0: float) -> dict:
    return skipped("numeric_validation_agent", "numeric_validations", [], reason, t0)
