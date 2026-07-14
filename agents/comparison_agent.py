"""
Runs in parallel with sentiment_agent (see build_graph.py).

For each comparison quarter, calls fetch_prior_quarter() to retrieve prior
chunks, then uses the LLM to identify language shifts between current and
prior quarter text. LLM role: linguistic comparison and shift detection only
— not arithmetic, not sentiment scoring.

Tools: fetch_prior_quarter(company, quarters_back) → list[dict]
LLM: gpt-5-mini via openai_client (function-calling not used here; structured
     output prompt used instead for reliability at Phase 1 scope).
"""

import json
import time
from graph.state import GraphState, DecisionLogEntry, ComparisonFinding
from tools.fetch_prior_quarter import fetch_prior_quarter
from azure_clients.openai_client import openai_client


_SYSTEM_PROMPT = """\
You are a financial analyst assistant specialising in earnings disclosure analysis.
You will be given excerpts from a company's CURRENT quarter filing/transcript and
one or more PRIOR quarter excerpts on the same topic.

Identify meaningful language shifts: changes in tone, dropped or added phrases,
hedging language that appeared or disappeared, forward guidance changes.

Respond ONLY with a JSON array. Each element must have:
  "topic": string,
  "current_language": string (verbatim excerpt),
  "prior_language": object {fiscal_label: verbatim excerpt},
  "shift_detected": boolean,
  "shift_description": string or null

No preamble, no markdown fences — raw JSON array only."""


def comparison_agent(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    t0 = time.time()
    company = state["company"]
    quarter = state["quarter"]
    comparison_quarters = state.get("comparison_quarters") or []
    retrieval_results = state.get("retrieval_results") or []

    if not retrieval_results:
        return _empty("no retrieval results to compare against", t0)

    if not comparison_quarters:
        return _empty("no comparison_quarters specified", t0)

    # Build current-quarter context (filing chunks preferred)
    current_text = _build_context(retrieval_results, prefer_doc_type="filing")

    # Fetch prior quarter chunks and build context per quarter
    prior_contexts: dict[str, str] = {}
    quarters_back_map = _resolve_quarters_back(quarter, comparison_quarters)

    for fiscal_label, quarters_back in quarters_back_map.items():
        try:
            prior_hits = fetch_prior_quarter(company=company, quarters_back=quarters_back)
            prior_contexts[fiscal_label] = _build_context(prior_hits, prefer_doc_type="filing")
        except Exception as exc:  # noqa: BLE001
            print(f"[comparison_agent] fetch_prior_quarter failed for {fiscal_label}: {exc}")

    if not prior_contexts:
        return _empty("all prior quarter fetches failed", t0)

    # Build LLM user message
    prior_section = "\n\n".join(
        f"--- PRIOR QUARTER: {label} ---\n{ctx}"
        for label, ctx in prior_contexts.items()
    )
    user_msg = (
        f"COMPANY: {company}\n"
        f"CURRENT QUARTER: {quarter}\n\n"
        f"--- CURRENT QUARTER EXCERPT ---\n{current_text}\n\n"
        f"{prior_section}"
    )

    findings: list[ComparisonFinding] = []
    tokens_used = None

    try:
        response = openai_client.chat.completions.create(
            model=openai_client._deployment,  # gpt-5-mini
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        tokens_used = response.usage.total_tokens if response.usage else None
        raw = response.choices[0].message.content or "[]"
        parsed = json.loads(raw)
        findings = [_to_finding(f) for f in parsed if isinstance(f, dict)]
    except json.JSONDecodeError as exc:
        print(f"[comparison_agent] JSON parse failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"[comparison_agent] LLM call failed: {exc}")

    entry: DecisionLogEntry = {
        "agent": "comparison_agent",
        "tool_called": "fetch_prior_quarter",
        "input_summary": f"company={company} quarter={quarter} prior={list(prior_contexts.keys())}",
        "output_summary": f"{len(findings)} findings, {sum(f['shift_detected'] for f in findings)} shifts detected",
        "confidence": None,
        "tokens_used": tokens_used,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }

    return {
        "comparison_findings": findings,
        "decision_log_entries": [entry],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context(chunks: list[dict], prefer_doc_type: str, max_chars: int = 4000) -> str:
    preferred = [c for c in chunks if c.get("doc_type") == prefer_doc_type]
    ordered = preferred + [c for c in chunks if c.get("doc_type") != prefer_doc_type]
    parts: list[str] = []
    total = 0
    for chunk in ordered:
        text = chunk.get("content", "")
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def _resolve_quarters_back(current_quarter: str, comparison_quarters: list[str]) -> dict[str, int]:
    """
    Maps each comparison quarter label to a quarters_back integer.
    Assumes quarters are in order (most recent = index 0 in sorted list).
    Falls back to sequential numbering if fiscal label parsing is unreliable.
    """
    result: dict[str, int] = {}
    for i, label in enumerate(comparison_quarters, start=1):
        result[label] = i
    return result


def _to_finding(raw: dict) -> ComparisonFinding:
    return ComparisonFinding(
        topic=str(raw.get("topic", "")),
        current_language=str(raw.get("current_language", "")),
        prior_language=raw.get("prior_language") or {},
        shift_detected=bool(raw.get("shift_detected", False)),
        shift_description=raw.get("shift_description"),
    )


def _empty(reason: str, t0: float) -> dict:
    entry: DecisionLogEntry = {
        "agent": "comparison_agent",
        "tool_called": None,
        "input_summary": reason,
        "output_summary": "skipped",
        "confidence": None,
        "tokens_used": None,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
    return {
        "comparison_findings": [],
        "decision_log_entries": [entry],
    }