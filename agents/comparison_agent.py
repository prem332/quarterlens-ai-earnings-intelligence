"""
Runs in parallel with sentiment_agent (see build_graph.py).

For each comparison quarter, calls fetch_prior_quarter() to retrieve prior
chunks, then uses the LLM to identify language shifts between current and
prior quarter text. LLM role: linguistic comparison and shift detection only
— not arithmetic, not sentiment scoring.

Fix (Phase 3): current-quarter context now uses retrieval_results directly
in their globally reranked order from retrieval_agent, instead of rebuilding
and reordering by doc_type. This ensures comparison_agent and report_agent
operate on the same evidence set with the same ranking.

Tools: fetch_prior_quarter(company, quarters_back) → list[dict]
LLM: gpt-5-mini via openai_client.achat() (async, Phase 2).
"""

import asyncio
import json
import time
from graph.state import GraphState, DecisionLogEntry, ComparisonFinding
from tools.fetch_prior_quarter import fetch_prior_quarter
from azure_clients.openai_client import openai_client


_SYSTEM_PROMPT = """\
You are a financial analyst assistant specialising in earnings disclosure analysis.
You will be given excerpts from a company's CURRENT quarter filing/transcript and
one or more PRIOR quarter excerpts on the same topic.

You will be given a QUERY naming the specific topic to analyze. Your task is to
answer that query, not to survey the excerpts broadly for anything noteworthy:

0. FIND THE QUERY'S TOPIC FIRST. Locate the current-quarter statement most
   directly relevant to QUERY, and the prior-quarter statement it corresponds
   to. Your FIRST array element must be this comparison. You may add further
   elements for other genuine shifts you notice, but the first one must
   address QUERY — a correct, on-topic "no shift" (shift_detected: false) is a
   better first element than an off-topic "shift" elsewhere in the excerpts.

Identify meaningful language shifts: changes in tone, dropped or added phrases,
hedging language that appeared or disappeared, forward guidance changes.

Apply these rules when deciding shift_detected — get these right, they are the
cases analysts most often get wrong:

1. NEW TOPIC, NO PRIOR COUNTERPART → shift_detected: true. If a topic appears in
   the current excerpts with nothing corresponding in the prior excerpts (e.g. a
   new investment, new risk disclosure, new commitment), that absence-then-presence
   IS itself a meaningful shift — an addition. Do not skip it just because there is
   nothing to diff against.

2. VERBATIM-IDENTICAL SENTENCE → shift_detected: false, even if nearby or
   surrounding text in the same excerpt changed. Judge each specific sentence you
   are comparing in isolation. Do not let unrelated changes elsewhere in the
   passage cause you to flag a sentence that itself did not change.

3. ATTRIBUTION/EXPLANATORY ADDITIONS THAT DON'T CHANGE THE CORE CLAIM →
   shift_detected: false. Adding a secondary explanatory factor (e.g. "and a
   different mix" alongside an already-stated driver) is standard disclosure
   elaboration, not a characterization change. Only flag a shift when the
   DIRECTION, MAGNITUDE, or SUBSTANTIVE characterization actually changes
   (e.g. "decreased" → "increased", a driver being dropped and replaced, new
   quantified risk exposure).

4. REORDERING OR REFORMATTING OF THE SAME CONTENT → shift_detected: false. The
   same items appearing in a different order, or as bullets vs. prose, is not a
   language shift.

5. GROUNDING — shift_description must state only what is explicitly present in
   the given excerpts. Do not infer intent, motivation, or characterize what
   management is "now emphasizing" unless that characterization is directly
   stated in the text. When in doubt, describe less rather than more.

Respond ONLY with a JSON array. Each element must have:
  "topic": string,
  "current_language": string (verbatim excerpt),
  "prior_language": object {fiscal_label: verbatim excerpt},
  "shift_detected": boolean,
  "shift_description": string or null

No preamble, no markdown fences — raw JSON array only."""


async def comparison_agent(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    t0 = time.time()
    company = state["company"]
    quarter = state["quarter"]
    query = state["query"]
    comparison_quarters = state.get("comparison_quarters") or []
    retrieval_results = state.get("retrieval_results") or []

    if not retrieval_results:
        return _empty("no retrieval results to compare against", t0)

    if not comparison_quarters:
        return _empty("no comparison_quarters specified", t0)

    # Use retrieval_results directly in globally reranked order — do not
    # rebuild or reorder by doc_type. This keeps current-quarter evidence
    # identical to what report_agent receives.
    current_text = _ranked_context(retrieval_results, max_chars=4000)

    # Fetch prior quarter chunks concurrently
    quarters_back_map = _resolve_quarters_back(quarter, comparison_quarters)

    async def _fetch_one(fiscal_label: str, quarters_back: int) -> tuple[str, str]:
        try:
            prior_result = await asyncio.to_thread(
                fetch_prior_quarter,
                company=company,
                current_quarter=quarter,
                quarters_back=quarters_back,
                query=query,
            )
            # Prior quarter context: preserve fetch order (no global ranking available)
            return fiscal_label, _ranked_context(prior_result["results"], max_chars=2000)
        except Exception as exc:
            print(f"[comparison_agent] fetch_prior_quarter failed for {fiscal_label}: {exc}")
            return fiscal_label, ""

    fetch_tasks = [
        _fetch_one(label, qb) for label, qb in quarters_back_map.items()
    ]
    fetch_results = await asyncio.gather(*fetch_tasks)
    prior_contexts = {label: ctx for label, ctx in fetch_results if ctx}

    if not prior_contexts:
        return _empty("all prior quarter fetches failed", t0)

    # Build LLM user message
    prior_section = "\n\n".join(
        f"--- PRIOR QUARTER: {label} ---\n{ctx}"
        for label, ctx in prior_contexts.items()
    )
    user_msg = (
        f"COMPANY: {company}\n"
        f"CURRENT QUARTER: {quarter}\n"
        f"QUERY: {query}\n\n"
        f"--- CURRENT QUARTER EXCERPT ---\n{current_text}\n\n"
        f"{prior_section}"
    )

    findings: list[ComparisonFinding] = []
    tokens_used = None

    try:
        response = await openai_client.achat_tiered(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model_tier=state.get("model_tier", "primary"),
        )
        tokens_used = response.usage.total_tokens if response.usage else None
        raw = response.choices[0].message.content or "[]"
        parsed = json.loads(raw)
        findings = [_to_finding(f) for f in parsed if isinstance(f, dict)]
    except json.JSONDecodeError as exc:
        print(f"[comparison_agent] JSON parse failed: {exc}")
    except Exception as exc:
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

def _ranked_context(chunks: list[dict], max_chars: int = 4000) -> str:
    """
    Build context string from chunks preserving their input order.
    For retrieval_results this is the global rerank order from retrieval_agent.
    For prior-quarter hits this is the fetch order from fetch_prior_quarter.
    No reordering by doc_type — the reranker already determined the best order.
    """
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        text = chunk.get("parent_content") or chunk.get("content", "")
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def _resolve_quarters_back(current_quarter: str, comparison_quarters: list[str]) -> dict[str, int]:
    """
    Map each comparison label to how many quarters back it is from the current
    quarter, computed from the labels themselves (QoQ=1, YoY=4, ...). Falls back
    to list position only if a label can't be parsed.
    """
    from tools.fetch_prior_quarter import _parse_fiscal_label

    try:
        c_idx, c_fy = _parse_fiscal_label(current_quarter)
        current_abs = c_fy * 4 + c_idx
    except ValueError:
        current_abs = None

    result: dict[str, int] = {}
    for i, label in enumerate(comparison_quarters, start=1):
        quarters_back = i
        if current_abs is not None:
            try:
                t_idx, t_fy = _parse_fiscal_label(label)
                dist = current_abs - (t_fy * 4 + t_idx)
                if dist >= 1:
                    quarters_back = dist
            except ValueError:
                pass
        result[label] = quarters_back
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