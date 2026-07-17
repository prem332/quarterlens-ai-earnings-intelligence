"""
agents/report_agent.py

Two-step:
  1. Draft: LLM synthesises all agent outputs into an analyst-tone briefing.
  2. Verify: LLM checks every factual claim in the draft traces back to a
     retrieved chunk or a validated numeric fact. Claims that can't be
     grounded are flagged or removed.

Both LLM calls are async (achat) — Phase 2 async execution.
gpt-5-mini note: reasoning model — max_completion_tokens must be >= 4096
or the model produces empty output (reasoning tokens exhaust the budget).
The openai_client wrapper enforces this minimum automatically.
"""

import asyncio
import time
from graph.state import (
    GraphState, DecisionLogEntry,
    ComparisonFinding, SentimentScore, NumericValidation,
)
from azure_clients.openai_client import openai_client


_DRAFT_SYSTEM = """\
You are a senior equity research analyst writing a concise earnings intelligence briefing.
Write in a professional, direct analyst tone — no financial advice, no buy/sell recommendations.
Structure the briefing as:

## Executive Summary
## Key Financial Metrics (verified)
## Guidance & Language Shifts
## Risk Factor Changes
## Sentiment Overview
## Source Citations

Every factual claim must reference specific evidence. Use [FILING] or [TRANSCRIPT] as inline tags.
Keep the total briefing under 800 words."""

_VERIFY_SYSTEM = """\
You are a fact-checker for financial analyst reports.
You will be given a DRAFT REPORT and the SOURCE EVIDENCE it was drawn from.
Your task:
1. Identify every factual claim in the draft.
2. Check whether each claim is supported by the provided evidence.
3. Remove or flag (with [UNVERIFIED]) any claim not traceable to the evidence.
4. Return the corrected report text ONLY — no commentary, no JSON."""


async def report_agent(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    t0 = time.time()
    total_tokens = 0

    # ── Step 1: Draft ─────────────────────────────────────────────────────
    draft_prompt = _build_draft_prompt(state)
    draft, tokens = await _llm_call(_DRAFT_SYSTEM, draft_prompt)
    total_tokens += tokens

    if not draft:
        return _empty("draft generation failed", t0)

    # ── Step 2: Verify ────────────────────────────────────────────────────
    evidence_summary = _build_evidence_summary(state)
    verify_prompt = (
        f"DRAFT REPORT:\n{draft}\n\n"
        f"SOURCE EVIDENCE:\n{evidence_summary}"
    )
    verified_report, tokens = await _llm_call(_VERIFY_SYSTEM, verify_prompt)
    total_tokens += tokens

    final_report = verified_report or draft  # fall back to draft if verify fails

    entry: DecisionLogEntry = {
        "agent": "report_agent",
        "tool_called": None,
        "input_summary": (
            f"chunks={len(state.get('retrieval_results', []))} "
            f"comparisons={len(state.get('comparison_findings', []))} "
            f"sentiments={len(state.get('sentiment_scores', []))} "
            f"validations={len(state.get('numeric_validations', []))}"
        ),
        "output_summary": f"report drafted and verified, len={len(final_report)}",
        "confidence": None,
        "tokens_used": total_tokens,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }

    return {
        "report": final_report,
        "decision_log_entries": [entry],
    }


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_draft_prompt(state: GraphState) -> str:
    company = state["company"]
    quarter = state["quarter"]

    chunks = state.get("retrieval_results") or []
    chunk_text = "\n\n".join(
        f"[{r['doc_type'].upper()}] {r['content']}" for r in chunks[:8]
    )

    findings: list[ComparisonFinding] = state.get("comparison_findings") or []
    findings_text = "\n".join(
        f"- {f['topic']}: shift={'YES' if f['shift_detected'] else 'no'} — {f.get('shift_description') or 'no change'}"
        for f in findings
    ) or "None detected."

    scores: list[SentimentScore] = state.get("sentiment_scores") or []
    if scores:
        pos = sum(1 for s in scores if s["label"] == "positive")
        neg = sum(1 for s in scores if s["label"] == "negative")
        neu = len(scores) - pos - neg
        sentiment_summary = f"positive={pos} negative={neg} neutral={neu} across {len(scores)} passages"
    else:
        sentiment_summary = "Not available."

    validations: list[NumericValidation] = state.get("numeric_validations") or []
    val_lines = []
    for v in validations:
        status = "✓" if v["match"] else "✗ MISMATCH"
        val_lines.append(
            f"- {v['metric']}: claimed={v['claimed_value']} "
            f"calculated={v['calculated_value']} {status}"
            + (f" (Δ{v['delta_pct']:.2f}%)" if v["delta_pct"] is not None else "")
        )
    validations_text = "\n".join(val_lines) or "No validations performed."

    return f"""COMPANY: {company}
QUARTER: {quarter}

=== RETRIEVED EVIDENCE ===
{chunk_text}

=== LANGUAGE SHIFT ANALYSIS ===
{findings_text}

=== SENTIMENT ANALYSIS (FinBERT) ===
{sentiment_summary}

=== NUMERIC VALIDATION ===
{validations_text}

Draft the analyst earnings intelligence briefing based on the above."""


def _build_evidence_summary(state: GraphState) -> str:
    """Compact evidence summary for the verification step."""
    lines: list[str] = []
    for r in (state.get("retrieval_results") or [])[:10]:
        lines.append(f"[{r['doc_type'].upper()}] {r['content'][:300]}")
    for v in (state.get("numeric_validations") or []):
        lines.append(
            f"[VALIDATED] {v['metric']}: claimed={v['claimed_value']} "
            f"calc={v['calculated_value']} match={v['match']}"
        )
    return "\n\n".join(lines)


# ── Async LLM wrapper ─────────────────────────────────────────────────────────

async def _llm_call(system: str, user: str) -> tuple[str, int]:
    """
    Async LLM call via openai_client.achat(). Returns (text, token_count).
    Wrapper default max_completion_tokens (16384) applies — safe for
    gpt-5-mini's reasoning token budget.
    """
    try:
        response = await openai_client.achat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens
    except Exception as exc:
        print(f"[report_agent] LLM call failed: {exc}")
        return "", 0


def _empty(reason: str, t0: float) -> dict:
    entry: DecisionLogEntry = {
        "agent": "report_agent",
        "tool_called": None,
        "input_summary": reason,
        "output_summary": "skipped",
        "confidence": None,
        "tokens_used": None,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
    return {
        "report": "",
        "decision_log_entries": [entry],
    }