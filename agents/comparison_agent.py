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

Two-step extract-then-compare (this session, replaces a single one-shot call):
a single call asking the LLM to survey the whole current+prior context and
return an array of "whatever shifts it notices" repeatedly picked the wrong
topic (a prompt-rules-only rewrite did not fix this — the task itself, not the
judging criteria, was the problem) and got the shift_detected verdict wrong on
the majority of sampled claims even after query-anchoring was added. Splitting
into (1) EXTRACT the specific current+prior statements on QUERY's topic, then
(2) COMPARE only those two narrow statements, removes the topic-selection
ambiguity entirely (one targeted finding, not an array to pick from) and gives
the verdict step a much narrower, less distracting input to reason over.

Tools: fetch_prior_quarter(company, quarters_back) → list[dict]
LLM: gpt-5-mini via openai_client.achat_tiered() (async, Phase 2).

_COMPARE_SYSTEM rule 6 (2026-08-02, Known Issue #4): fixed a real, reproduced
instability in how small magnitude-only differences are judged. Rules 1-5
alone handled clear cases fine (large deceleration, pure paraphrase, sign
flips, verbatim) with 100% consistency across repeated isolated calls, but a
trivial difference with the same driver/tone (e.g. "up 15%" vs "up 17%")
flip-flopped 3-true/1-false across 4 repeats — the model had no materiality
threshold to anchor on. Added a rule giving one explicitly: a small
same-driver magnitude difference is not a shift; only flag magnitude changes
an analyst would actually call out. Re-verified: the same case now returns
false consistently (5/5), with no regression on the 4 previously-passing
cases. Zero added LLM calls (still the same 2-call extract+compare flow) —
prompt got longer, not more calls, so no production latency impact.
"""

import asyncio
import json
import time
from graph.state import GraphState, DecisionLogEntry, ComparisonFinding
from agents._common import ms, skipped
from tools.fetch_prior_quarter import fetch_prior_quarter
from azure_clients.openai_client import openai_client


_EXTRACT_SYSTEM = """\
You are a financial analyst assistant. You will be given a QUERY naming a specific
topic, a CURRENT quarter excerpt, and one or more PRIOR quarter excerpts.

Your ONLY job here is extraction, not judgment. Do not decide whether anything
shifted — just locate text.

1. Find the verbatim sentence(s) in the CURRENT excerpt that most directly address
   QUERY. If nothing in CURRENT addresses QUERY at all, say so.
2. For EACH prior quarter excerpt, find the verbatim sentence(s) addressing that
   SAME specific topic — the corresponding statement, if one exists. If nothing in
   a given prior excerpt corresponds to the same specific topic, say so explicitly
   rather than picking the closest-sounding unrelated sentence — a genuine "no
   counterpart" is a correct and useful answer, not a failure.

Respond ONLY with JSON, no markdown fences:
{
  "topic": "<short topic label>",
  "current_statement": "<verbatim sentence(s) from CURRENT, or null if none addresses QUERY>",
  "prior_statements": {"<fiscal_label>": "<verbatim sentence(s), or null if no corresponding statement>", ...}
}"""

_COMPARE_SYSTEM = """\
You are comparing exactly ONE current-quarter statement against its prior-quarter
counterpart(s) to decide if the language meaningfully shifted. You are given only
the extracted statements below — nothing else. Apply these rules when deciding
shift_detected — get these right, they are the cases analysts most often get wrong:

1. NEW TOPIC, NO PRIOR COUNTERPART → shift_detected: true. If a prior_statement is
   null (nothing corresponding was found), that absence-then-presence IS itself a
   meaningful shift — an addition. Do not skip it just because there is nothing to
   diff against.

2. VERBATIM-IDENTICAL SENTENCE → shift_detected: false. Judge the specific
   statements given in isolation — you have no surrounding text to be misled by.

3. ATTRIBUTION/EXPLANATORY ADDITIONS THAT DON'T CHANGE THE CORE CLAIM →
   shift_detected: false. Adding a secondary explanatory factor (e.g. "and a
   different mix" alongside an already-stated driver) is standard disclosure
   elaboration, not a characterization change. Only flag a shift when the
   DIRECTION, MAGNITUDE, or SUBSTANTIVE characterization actually changes
   (e.g. "decreased" → "increased", a driver being dropped and replaced, new
   quantified risk exposure).

4. REORDERING OR REFORMATTING OF THE SAME CONTENT → shift_detected: false. The
   same items in a different order, or as bullets vs. prose, is not a language shift.

5. GROUNDING — shift_description must state only what is explicitly present in the
   given statements. Do not infer intent, motivation, or characterize what
   management is "now emphasizing" unless directly stated. When in doubt, describe
   less rather than more.

6. A SMALL, MARGINAL DIFFERENCE IN A REPORTED NUMBER (a few percentage points,
   e.g. 15% vs 17%) is NOT by itself a meaningful shift when the driver,
   direction, and tone are otherwise the same — treat it like rule 3's
   "explanatory addition" case: shift_detected: false. Reserve shift_detected:
   true for a magnitude change an analyst would actually call out: a clear
   deceleration/acceleration (e.g. 32% -> 15%), a change from growth to decline
   or vice versa, or a genuinely different driver/tone. If the current figure
   could fairly be described as "roughly in line with" the prior one, that is
   false, not true — this is the rule most often gotten wrong, so apply it even
   when the exact wording differs slightly (e.g. "continued growth" vs
   "increased").

Respond ONLY with JSON, no markdown fences:
{"shift_detected": boolean, "shift_description": string or null}"""


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

    # Build LLM user message for the extract step
    prior_section = "\n\n".join(
        f"--- PRIOR QUARTER: {label} ---\n{ctx}"
        for label, ctx in prior_contexts.items()
    )
    extract_user_msg = (
        f"COMPANY: {company}\n"
        f"CURRENT QUARTER: {quarter}\n"
        f"QUERY: {query}\n\n"
        f"--- CURRENT QUARTER EXCERPT ---\n{current_text}\n\n"
        f"{prior_section}"
    )

    findings: list[ComparisonFinding] = []
    tokens_used = 0
    model_tier = state.get("model_tier", "primary")

    try:
        # ── Step 1: EXTRACT — locate the specific statements, no judgment yet ──
        extract_resp = await openai_client.achat_tiered(
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": extract_user_msg},
            ],
            model_tier=model_tier,
        )
        tokens_used += extract_resp.usage.total_tokens if extract_resp.usage else 0
        extracted = json.loads(extract_resp.choices[0].message.content or "{}")

        topic = str(extracted.get("topic", ""))
        current_statement = extracted.get("current_statement")
        prior_statements = {
            label: text for label, text in (extracted.get("prior_statements") or {}).items()
            if text
        }

        if not current_statement:
            return _empty("query's topic not found in current-quarter excerpt", t0)

        # ── Step 2: COMPARE — judge only the narrow extracted pair ──────────────
        compare_user_msg = (
            f"CURRENT STATEMENT: {current_statement}\n\n"
            f"PRIOR STATEMENT(S): "
            f"{json.dumps(prior_statements) if prior_statements else 'null (no corresponding statement found)'}"
        )
        compare_resp = await openai_client.achat_tiered(
            messages=[
                {"role": "system", "content": _COMPARE_SYSTEM},
                {"role": "user", "content": compare_user_msg},
            ],
            model_tier=model_tier,
        )
        tokens_used += compare_resp.usage.total_tokens if compare_resp.usage else 0
        verdict = json.loads(compare_resp.choices[0].message.content or "{}")

        findings = [ComparisonFinding(
            topic=topic,
            current_language=str(current_statement),
            prior_language=prior_statements,
            shift_detected=bool(verdict.get("shift_detected", False)),
            shift_description=verdict.get("shift_description"),
        )]
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
        "latency_ms": ms(t0),
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

    Skips (does not stop at) a chunk that would overflow the budget. This used
    to `break`, so one oversized chunk at the front discarded everything behind
    it and returned "" — and these are parent-expanded blocks, which are large
    by construction. Measured on the seed-42 sample: 5 of 10 claims got an empty
    current-quarter excerpt, including comparison claims, meaning the LLM was
    asked to compare against nothing. Same bug as numeric_validation_agent's
    _concat_transcript.
    """
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        text = chunk.get("parent_content") or chunk.get("content", "")
        if not text or total + len(text) > max_chars:
            continue
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


def _empty(reason: str, t0: float) -> dict:
    return skipped("comparison_agent", "comparison_findings", [], reason, t0)
