"""
Two-step process:
  1. LLM extracts numeric claims from transcript chunks (what did the CEO say?).
  2. calculate_metric() computes the same metric from SQL financial_facts deterministically.
  3. Agent compares the two — match/mismatch, delta %.

The LLM is ONLY used in step 1 (claim extraction from natural language).
All arithmetic is done by calculate_metric() — never by the LLM.
Zero-tolerance accuracy target per ARCHITECTURE.md §7.

Tools: calculate_metric(statement_data, formula) — deterministic SQL/Python calculation
LLM: gpt-5-mini via openai_client.achat() (async, Phase 2).
"""

import asyncio
import json
import time
from graph.state import GraphState, DecisionLogEntry, NumericValidation
from tools.calculate_metric import calculate_metric
from azure_clients.openai_client import openai_client


_CLAIM_EXTRACTION_PROMPT = """\
You are a financial data extraction assistant.
Extract every specific numeric claim made by management from the transcript excerpts below.
For each claim, identify:
  - "claim": exact quoted phrase containing the number
  - "metric": short snake_case identifier (e.g. revenue_growth_yoy, gross_margin, eps_diluted)
  - "claimed_value": the numeric value as a float (percentages as decimals if stated as %, else raw)
  - "value_type": "percentage" | "absolute" | "ratio"
  - "period": fiscal quarter or period the claim refers to (e.g. "Q2_FY2025")

Respond ONLY with a JSON array. No preamble, no markdown fences."""


async def numeric_validation_agent(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    t0 = time.time()
    company = state["company"]
    quarter = state["quarter"]
    retrieval_results = state.get("retrieval_results") or []

    # Step 1: pull transcript chunks to extract claims from
    transcript_text = _concat_transcript(retrieval_results)
    if not transcript_text.strip():
        return _empty("no transcript content for claim extraction", t0)

    # Step 2: extract claims via LLM (async)
    raw_claims = await _extract_claims(transcript_text)
    if not raw_claims:
        return _empty("no numeric claims extracted from transcript", t0)

    # Step 3: validate each claim against SQL financial_facts (concurrent)
    async def _validate_one(claim_obj: dict) -> NumericValidation:
        claimed_metric = claim_obj.get("metric", "")
        claimed_value = claim_obj.get("claimed_value")
        period = claim_obj.get("period", quarter)

        try:
            calc_result = await asyncio.to_thread(
                calculate_metric,
                statement_data={
                    "company": company,
                    "fiscal_label": period,
                    "metric": claimed_metric,
                },
                formula=claimed_metric,
            )
            calculated_value = calc_result.get("value")
            match, delta_pct = _compare(claimed_value, calculated_value, claim_obj.get("value_type"))
        except Exception as exc:  # noqa: BLE001
            print(f"[numeric_validation_agent] calculate_metric failed for {claimed_metric}: {exc}")
            calculated_value = None
            match = False
            delta_pct = None

        return NumericValidation(
            claim=str(claim_obj.get("claim", "")),
            metric=claimed_metric,
            claimed_value=claimed_value,
            calculated_value=calculated_value,
            match=match,
            delta_pct=delta_pct,
            source_fiscal_label=period,
        )

    validation_tasks = [_validate_one(c) for c in raw_claims]
    validations: list[NumericValidation] = list(await asyncio.gather(*validation_tasks))

    mismatches = sum(1 for v in validations if not v["match"])
    entry: DecisionLogEntry = {
        "agent": "numeric_validation_agent",
        "tool_called": "calculate_metric",
        "input_summary": f"company={company} quarter={quarter} claims={len(raw_claims)}",
        "output_summary": f"{len(validations)} validated, {mismatches} mismatches",
        "confidence": 1.0 if mismatches == 0 else round(1 - mismatches / max(len(validations), 1), 2),
        "tokens_used": None,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }

    return {
        "numeric_validations": validations,
        "decision_log_entries": [entry],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _concat_transcript(retrieval_results: list, max_chars: int = 6000) -> str:
    parts: list[str] = []
    total = 0
    for r in retrieval_results:
        if r.get("doc_type", "").lower() not in ("transcript", "earnings_call"):
            continue
        text = r.get("content", "")
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


async def _extract_claims(transcript_text: str) -> list[dict]:
    try:
        response = await openai_client.achat(
            messages=[
                {"role": "system", "content": _CLAIM_EXTRACTION_PROMPT},
                {"role": "user", "content": transcript_text},
            ],
        )
        raw = response.choices[0].message.content or "[]"
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[numeric_validation_agent] claim JSON parse failed: {exc}")
        return []
    except Exception as exc:  # noqa: BLE001
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
    entry: DecisionLogEntry = {
        "agent": "numeric_validation_agent",
        "tool_called": None,
        "input_summary": reason,
        "output_summary": "skipped",
        "confidence": None,
        "tokens_used": None,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
    return {
        "numeric_validations": [],
        "decision_log_entries": [entry],
    }