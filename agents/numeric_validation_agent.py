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
You are a financial data extraction assistant.
Extract every specific numeric claim made by management from the transcript excerpts below.
For each claim, identify:
  - "claim": exact quoted phrase containing the number
  - "metric": short snake_case identifier (e.g. revenue_growth_yoy, gross_margin, eps_diluted)
  - "claimed_value": the numeric value as a float (percentages as decimals if stated as %, else raw)
  - "value_type": "percentage" | "absolute" | "ratio"
  - "period": fiscal quarter the claim refers to, in the exact format "FY2025-Q2"
    (four-digit year, dash, Q + quarter number — this must match calculate_metric's
    fiscal_label format exactly, or the lookup will silently find nothing)

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

    for claim_obj in raw_claims:
        claimed_metric = claim_obj.get("metric", "")
        claimed_value = claim_obj.get("claimed_value")
        period = _normalize_period(claim_obj.get("period"), quarter)

        try:
            calc_result = calculate_metric(
                company=company,
                fiscal_label=period,
                metric=claimed_metric,
                prior_fiscal_label=None,  # growth metrics need prior period — not available from transcript extraction
            )
            calculated_value = calc_result.get("value")  # fixed: was "value" key mismatch
            match, delta_pct = _compare(claimed_value, calculated_value, claim_obj.get("value_type"))
        except Exception as exc:  # noqa: BLE001
            print(f"[numeric_validation_agent] calculate_metric failed for {claimed_metric}: {exc}")
            calculated_value = None
            match = False
            delta_pct = None

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
        "output_summary": f"{len(validations)} validated, {mismatches} mismatches",
        "confidence": 1.0 if mismatches == 0 else round(1 - mismatches / max(len(validations), 1), 2),
        "tokens_used": tokens_used,
        "latency_ms": ms(t0),
    }

    return {
        "numeric_validations": validations,
        "decision_log_entries": [entry],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_period(period: str | None, fallback: str) -> str:
    """
    Canonicalize the LLM-extracted period into the "FY2025-Q2" format
    calculate_metric/financial_facts actually key on.

    The extraction prompt now asks for this format directly, but LLMs don't
    reliably follow format instructions — this is a defensive second layer.
    Confirmed as the root cause of a 100% numeric-validation failure rate: the
    prompt previously suggested "Q2_FY2025" as the example, the LLM followed
    it literally, and calculate_metric's fiscal_label lookup silently found
    nothing for every single claim (verified directly: calculate_metric(...,
    fiscal_label="Q4_FY2025") -> value=None, "No fact found"; the identical
    call with fiscal_label="FY2025-Q4" -> the real filed value). Reuses
    tools.fetch_prior_quarter._parse_fiscal_label, which already accepts both
    the canonical and legacy formats (comparison_agent imports it the same way).

    Falls back to `fallback` — the state's own known-correct quarter — when the
    extracted string can't be parsed as a quarter at all (missing, or the LLM
    described a period in free text instead of a Q#/year pair).
    """
    if period:
        try:
            q_idx, year = _parse_fiscal_label(period)
            return f"FY{year}-Q{q_idx + 1}"
        except ValueError:
            pass
    return fallback


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
