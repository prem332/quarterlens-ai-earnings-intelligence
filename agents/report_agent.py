"""
agents/report_agent.py

Three-step:
  1. Bull/Bear debate: CrewAI two-agent debate over the retrieved evidence
     (bull analyst vs bear analyst) — surfaces competing interpretations
     before the draft is written.
  2. Draft: LLM synthesises all agent outputs + debate into analyst-tone briefing.
  3. Verify: LLM checks every factual claim in the draft traces back to a
     retrieved chunk or a validated numeric fact.

Fix (Phase 3): draft and verify now use the identical chunk_text payload.
Previously verify used _build_evidence_summary() which truncated each chunk
to 300 chars — the verifier could not confirm claims from the full chunk text
used by the drafter. Now the same chunk_text string is passed to both steps.

gpt-5.4-mini note: reasoning model — max_completion_tokens must be >= 4096.
The openai_client wrapper enforces this minimum automatically.
"""

import os
import re
import asyncio
import time
import openai
from crewai import Agent, Task, Crew, LLM
from azure_clients.key_vault_client import kv
from graph.state import (
    GraphState, DecisionLogEntry,
    ComparisonFinding, SentimentScore, NumericValidation,
)
from agents._common import ms, skipped
from azure_clients.openai_client import openai_client

# Retry/backoff for draft + verify calls — same curve as
# data_pipeline/embedding.py's _embed_batch. Added after the parallelized
# bull/bear debate (two simultaneous calls instead of sequential) made 429s
# from the gpt-5-mini dev deployment (10K TPM) more likely to land right as
# draft was about to run: draft had zero retry, so one 429 collapsed the
# entire report to empty instead of just costing a few extra seconds.
_MAX_RETRIES = 4
_RETRY_BACKOFF = 4.0   # seconds, doubled per retry (4, 8, 16)
_MAX_RETRY_AFTER = 60  # cap on an Azure-supplied Retry-After, so a long
                       # server hint can't stall a request indefinitely

# ── Fix 1: report_agent always uses the primary deployment ───────────────────
# router.py routes simple-looking questions ("what is the revenue growth...")
# to the "standard" tier on the assumption that a smaller model is cheaper and
# faster. For this model pair that is measurably backwards. Same 5,018-token
# verify prompt, measured 2026-07-31:
#
#   gpt-5-mini   (standard)  8.33s   512 hidden reasoning tokens
#                            -> left "No data available." (missed the rule)
#   gpt-5.4-mini (primary)   1.54s     0 hidden reasoning tokens
#                            -> wrote "No verified data available." (correct)
#
# 5.4x faster AND more correct. gpt-5-mini is also the 10K-TPM dev deployment,
# so it is the one that actually 429s -- a real run today produced a blank
# report for exactly that reason. Report generation is the most expensive and
# most user-visible step in the pipeline; it should not run on the dev
# deployment. Other agents keep using router.py's tier.
#
# Ablate with REPORT_FORCE_PRIMARY=0 to restore router-chosen routing.
_FORCE_PRIMARY_TIER = os.environ.get("REPORT_FORCE_PRIMARY", "1") != "0"

# ── Fix 2: skip the verify pass when it provably has nothing to delete ───────
# Verify re-sends the entire evidence set to re-read a draft that was just
# generated from that same evidence. In a real traced call it spent 17.15s and
# 8,537 tokens to change exactly one line ("No data available." ->
# "No verified data available.").
#
# Its primary job is deleting numeric claims that aren't in the evidence, so
# when every number in the draft is present in the evidence there is nothing
# for it to delete. That check is deterministic and exact -- no model needed.
#
# Deliberately conservative: any doubt runs verify. The skip requires all of
#   * every numeric token in the draft appears in the evidence
#   * the draft actually contains numbers (a purely qualitative draft still
#     gets checked, since claims there can be unsupported without a number)
#   * the question isn't asking for a recommendation (verify rule 0b refuses
#     those, and that refusal must not be bypassed)
#
# Ablate with REPORT_SKIP_VERIFY=0. NOTE: this changes what reaches the user
# on skipped runs, so it needs an eval run against the locked faithfulness
# baseline before being treated as settled.
_SKIP_VERIFY_WHEN_CLEAN = os.environ.get("REPORT_SKIP_VERIFY", "1") != "0"

_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")
_RECOMMENDATION_RE = re.compile(
    r"\b(should i|recommend|buy|sell|invest|overweight|underweight|good stock|worth buying)\b",
    re.IGNORECASE,
)


def _numeric_tokens(text: str) -> list[str]:
    """Digit sequences with separators stripped, so '$62,578' in a draft still
    matches '62,578' / '62578' however the filing table happened to format it."""
    return [m.group(0).replace(",", "").rstrip(".") for m in _NUM_RE.finditer(text or "")]


def _verify_needed(draft: str, evidence: str, query: str) -> tuple[bool, str]:
    """Return (needs_verify, reason). See _SKIP_VERIFY_WHEN_CLEAN above."""
    if not _SKIP_VERIFY_WHEN_CLEAN:
        return True, "skip disabled"
    if _RECOMMENDATION_RE.search(query or ""):
        return True, "recommendation-style question"

    draft_nums = _numeric_tokens(draft)
    if not draft_nums:
        return True, "no numeric claims to check deterministically"

    evidence_nums = set(_numeric_tokens(evidence))
    unsupported = [n for n in draft_nums if n not in evidence_nums]
    if unsupported:
        return True, f"{len(unsupported)} unsupported number(s), e.g. {unsupported[:3]}"
    return False, f"all {len(draft_nums)} draft numbers found in evidence"


def _retry_after_seconds(exc) -> float | None:
    """Azure returns a Retry-After header on 429 saying when quota frees up.

    The previous policy ignored it and used a fixed 2s/4s backoff, which is
    far shorter than the deployment actually needs — a real GOOGL run burned
    all 3 attempts in ~6s and produced an empty report. Honouring the header
    is what makes the retry meaningful rather than decorative.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        raw = headers.get(key)
        if not raw:
            continue
        try:
            return min(float(str(raw).rstrip("s")), _MAX_RETRY_AFTER)
        except (TypeError, ValueError):
            continue
    return None


def _fallback_tier(model_tier: str) -> str | None:
    """gpt-5-mini (standard) is the low-quota dev deployment and is what
    actually gets rate-limited; gpt-5.4-mini (primary) is Global Standard with
    far more headroom. When standard is exhausted, finishing the report on
    primary beats returning nothing."""
    return "primary" if model_tier == "standard" else None


_DRAFT_SYSTEM = """\
You are a senior equity research analyst writing earnings intelligence briefings for institutional investors.

You will always be given a specific QUESTION. Answering that question is the primary job:
- The Executive Summary's first 1-2 sentences must directly answer the QUESTION — not a general
  overview of the quarter. A reader who reads only those sentences should have their question answered.
- Scope the rest of the report to the QUESTION. If it asks about one specific metric, topic, or
  segment, keep the report focused there — do not pad with unrelated sections just to fill space.
  A section with nothing relevant to the QUESTION should be written as "Not applicable to this
  question." (one line), not expanded with tangential content.
- Broad questions ("how did the quarter go overall") still warrant the fuller structure below.
- If QUESTION asks for a SPECIFIC figure, breakdown, or exact disclosure (an exact sub-segment
  number, an adjusted/non-GAAP figure, a precise dollar impact) and the retrieved evidence does not
  contain that exact disclosure — only a rounded mention, a different metric, transcript commentary
  standing in for a filing figure, or an approximation — say so explicitly: state plainly that the
  filing/transcript does not disclose the specific figure requested, rather than substituting the
  closest available number as if it satisfies the question. A correct "this isn't disclosed at the
  precision requested" is a better answer than a confident approximation.
- If QUESTION asks for a recommendation, judgment call, or opinion outside objective reporting of
  disclosed facts — "should I buy/sell/invest", "is this a good stock", "should I overweight this
  position", or similar — DECLINE. Write only a brief Executive Summary explaining this system
  reports on disclosed earnings data and does not give investment recommendations; do not fill out
  the rest of the report structure, and do not answer the recommendation question indirectly by
  presenting a bull/bear synthesis as if it were an implied answer. Having plenty of retrieved
  evidence available does NOT make an investment recommendation in scope — decline regardless of
  how much relevant financial data was retrieved.

TONE:
- Professional, direct, assertive — not hedged or generic
- Use active voice: "Revenue grew 10%" not "Revenue was reported to have grown"
- No filler phrases: never use "it is worth noting", "importantly", "it should be mentioned"
- No financial advice, no buy/sell/hold recommendations

STRUCTURE (always follow this exact order; collapse non-applicable sections to one line as above):
## Executive Summary (2-3 sentences max — directly answers the QUESTION first)
## Key Financial Metrics (bullet points, one metric per line)
## Guidance & Language Shifts (what changed vs prior quarter, be specific)
## Risk Factor Changes (what's new or dropped)

Write NO other sections. Every sentence you emit must be traceable to a specific statement in the
retrieved evidence. Do not add a sentiment overview, a bull/bear section, a citations list, an
outlook, or any closing synthesis — analysis inputs given to you below (the bull/bear debate,
FinBERT sentiment counts) are BACKGROUND to help you decide what matters and what to lead with.
They are not themselves evidence and must never be restated as findings: a derived sentiment
label, a debate talking point, or your own synthesis is not something the filing says, and stating
it as though it were makes the report unverifiable.

FORMATTING RULES:
- Never write paragraphs longer than 3 sentences
- Use bullet points for lists of 3+ items
- Every number must have a unit: "$94.0B" not "94036"
- Every factual claim tagged: [FILING] or [TRANSCRIPT]
- No hallucinated facts — ONLY state what is explicitly present in the retrieved evidence
- Dates always in format: Q3 FY2025, not "third quarter of fiscal year 2025"
- If a section has no evidence, write "No data available." — do not invent content

LENGTH: for a narrow, single-topic QUESTION, 150-350 words focused on that topic is correct — do
not stretch to fill 600-800 words. For a broad QUESTION covering the whole quarter, 600-800 words.

EXAMPLE OF IDEAL OUTPUT FORMAT:
## Executive Summary
Apple delivered $94.0B in revenue for Q3 FY2025, up 5% year-over-year, driven by Services growth. [FILING] Gross margin expanded to 46.5%, reflecting favorable product mix. [FILING]

## Key Financial Metrics
- Revenue: $94.0B [FILING]
- Gross Profit: $43.7B (46.5% margin) [FILING]
- Operating Income: $28.2B [FILING]
- EPS (diluted): $1.57 [FILING]

## Guidance & Language Shifts
Management maintained confidence in Services momentum but removed prior references to "strong iPhone demand." [FILING] The phrase "we remain cautious" appeared for the first time in guidance language. [TRANSCRIPT]"""

_VERIFY_SYSTEM = """\
You are a strict fact-checker for financial analyst reports.
You will be given the QUESTION that was asked, a DRAFT REPORT, and the SOURCE EVIDENCE
it was drawn from.

YOUR TASK — apply these rules in order:

0. REFUSAL CHECK — a second, independent check on whether this draft should have
   declined to answer. The draft step has this same instruction but doesn't always
   follow it; treat that as unreliable and re-check here regardless of what the draft
   already did:
   a. If QUESTION asks for a specific figure/breakdown/exact disclosure and the
      SOURCE EVIDENCE only contains an approximation, a rounded mention, a different
      metric, or transcript commentary standing in for a filing-only figure — and the
      draft presents that substitute as if it satisfies QUESTION — REWRITE the
      Executive Summary to state plainly that the exact figure requested is not
      disclosed at that precision in the evidence, and remove the substituted figure
      from the rest of the report. Two substitutions that specifically count as
      failures here, because they manufacture precision that the source does not have:
        - UNIT CONVERSION: QUESTION asks for a figure "in millions"; evidence says
          "$4.3 billion"; draft writes "$4,300 million". The evidence disclosed two
          significant figures — rewriting the unit does not make it an exact
          disclosure. This is a refusal, not an answer.
        - SOURCE SUBSTITUTION: QUESTION asks what a specific document (e.g. "the
          10-Q") discloses; the figure appears only in the earnings-call transcript.
          A transcript number is not a filing disclosure. Say the filing does not
          disclose it — naming where it did appear is fine, presenting it as the
          filing's answer is not.
   b. If QUESTION asks for a recommendation, judgment call, or opinion (should I buy/
      sell/invest/overweight this, is this a good stock) — REWRITE the entire report
      to a brief decline: this system reports on disclosed earnings data and does not
      give investment recommendations. Do this regardless of how much relevant
      financial data the draft cites — having evidence available does not put a
      recommendation in scope.
   If neither applies, continue to the draft as-is.

1. IDENTIFY every factual claim in the (possibly rewritten) draft (numbers, percentages, growth rates, quotes, guidance statements).

2. CHECK each claim against the SOURCE EVIDENCE:
   - SUPPORTED: claim is explicitly stated in the evidence with matching figures
   - UNSUPPORTED: claim is not present in the evidence, inferred, or calculated by the model

3. ACT on unsupported claims:
   - Numbers not in evidence → DELETE the entire sentence containing them
   - Qualitative claims not in evidence → DELETE the entire sentence
   - Growth rates (YoY, QoQ) not explicitly stated in evidence → DELETE (never calculate)
   - Comparisons to prior quarters not in evidence → DELETE

4. RETURN the corrected report text ONLY.
   - No commentary, no JSON, no explanations
   - Preserve all section headers (##)
   - PRESERVE SENTENCES YOU KEEP VERBATIM, including their [FILING] / [TRANSCRIPT]
     citation tags. You are a deleter, not an editor: a retained sentence must come
     through character-for-character. Do not paraphrase, re-word, compress, or drop
     the tags — an untagged sentence is unattributable and defeats the point of
     verifying it.
   - If a section becomes empty after removing unsupported claims, write "No verified data available."
     EXCEPT the Executive Summary: it must always say something useful. If its claims
     don't survive, replace it with one sentence naming what the evidence does not
     establish (rule 0a's phrasing), not the bare "No verified data available."
   - Do NOT add new content beyond what rule 0's rewrite requires — only remove unsupported claims

CRITICAL: When in doubt, DELETE. A shorter grounded report scores higher than a longer hallucinated one."""


# ── CrewAI LLM factory ────────────────────────────────────────────────────────

def _make_crewai_llm(model_tier: str) -> LLM:
    endpoint = kv.get_secret("AZURE-OPENAI-ENDPOINT")
    os.environ["AZURE_ENDPOINT"] = endpoint

    deployment = (
        kv.get_secret("AZURE-OPENAI-DEPLOYMENT-NAME-STANDARD")
        if model_tier == "standard"
        else kv.get_secret("AZURE-OPENAI-DEPLOYMENT-NAME")
    )
    return LLM(
        model=f"azure/{deployment}",
        api_key=kv.get_secret("AZURE-OPENAI-KEY"),
        api_base=endpoint,
        api_version="2024-12-01-preview",
    )


# ── CrewAI bull/bear debate ───────────────────────────────────────────────────
#
# Bull and bear cases are independent — neither reads the other's output — but
# CrewAI's default Process.sequential ran them as two LLM calls back to back
# inside one Crew, serializing two calls that have nothing to wait on each
# other for. Split into two single-agent, single-task crews so report_agent
# can run them concurrently via asyncio.gather(to_thread(...), to_thread(...))
# instead of one crew.kickoff() blocking on both in turn. Same prompts, same
# model, same evidence — only the execution schedule changes.

def _run_bull_sync(evidence_summary: str, model_tier: str) -> str:
    try:
        llm = _make_crewai_llm(model_tier)
        bull_analyst = Agent(
            role="Bull Analyst",
            goal="Make the strongest positive case for this company's earnings results",
            backstory=(
                "You are an optimistic equity research analyst who focuses on growth "
                "drivers, positive surprises, and upside catalysts in earnings disclosures. "
                "You are rigorous — you only cite evidence that actually exists in the filing."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
        bull_task = Task(
            description=(
                f"Based on the following earnings evidence, present the strongest "
                f"positive investment case in 150-200 words. Focus on growth drivers, "
                f"beats vs expectations, positive guidance, and operational strengths.\n\n"
                f"EVIDENCE:\n{evidence_summary}"
            ),
            expected_output="A 150-200 word bull case summary citing specific evidence.",
            agent=bull_analyst,
        )
        crew = Crew(agents=[bull_analyst], tasks=[bull_task], verbose=False)
        return crew.kickoff().raw
    except Exception as exc:
        print(f"[report_agent] CrewAI bull analysis failed (non-fatal): {exc}")
        return ""


def _run_bear_sync(evidence_summary: str, model_tier: str) -> str:
    try:
        llm = _make_crewai_llm(model_tier)
        bear_analyst = Agent(
            role="Bear Analyst",
            goal="Identify the key risks, weaknesses, and concerns in this company's earnings results",
            backstory=(
                "You are a skeptical equity research analyst who focuses on risks, "
                "missed targets, deteriorating metrics, and cautionary language in earnings "
                "disclosures. You are rigorous — you only cite evidence that actually exists."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
        bear_task = Task(
            description=(
                f"Based on the following earnings evidence, present the strongest "
                f"critical case in 150-200 words. Focus on risks, misses, deteriorating "
                f"trends, hedged guidance, and concerns raised by analysts.\n\n"
                f"EVIDENCE:\n{evidence_summary}"
            ),
            expected_output="A 150-200 word bear case summary citing specific evidence.",
            agent=bear_analyst,
        )
        crew = Crew(agents=[bear_analyst], tasks=[bear_task], verbose=False)
        return crew.kickoff().raw
    except Exception as exc:
        print(f"[report_agent] CrewAI bear analysis failed (non-fatal): {exc}")
        return ""


async def _run_debate(evidence_summary: str, model_tier: str) -> str:
    bull_text, bear_text = await asyncio.gather(
        asyncio.to_thread(_run_bull_sync, evidence_summary, model_tier),
        asyncio.to_thread(_run_bear_sync, evidence_summary, model_tier),
    )
    if not bull_text and not bear_text:
        return ""
    return f"=== BULL CASE ===\n{bull_text}\n\n=== BEAR CASE ===\n{bear_text}"


# ── Main agent node ───────────────────────────────────────────────────────────

async def report_agent(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    t0 = time.time()
    total_tokens = 0
    routed_tier = state.get("model_tier", "primary")
    model_tier = "primary" if _FORCE_PRIMARY_TIER else routed_tier
    if model_tier != routed_tier:
        print(f"[report_agent] overriding routed tier '{routed_tier}' -> 'primary' (see _FORCE_PRIMARY_TIER)")
    # Not part of GraphState's TypedDict (would need Python 3.11+ NotRequired;
    # this repo targets 3.10) — set only by api/routes/analysis.py when a
    # browser is listening on the SSE stream endpoint. None for every eval-path
    # invocation (run_baseline_eval.py calls compiled_graph.ainvoke() directly
    # with no such key), which is exactly when streaming should be skipped.
    stream_queue = state.get("stream_queue")

    # ── Step 1: Bull/Bear debate (CrewAI) ─────────────────────────────────
    evidence_summary = _build_evidence_summary(state)
    debate_summary = await _run_debate(evidence_summary, model_tier)

    # ── Step 2: Draft ─────────────────────────────────────────────────────
    # chunk_text and extra_evidence are built once and reused in verify —
    # ensures draft and verifier operate on identical evidence, not a subset.
    chunk_text = _build_chunk_text(state)
    extra_evidence = _build_extra_evidence(state)
    draft_prompt = _build_draft_prompt(state, chunk_text, extra_evidence, debate_summary)
    if stream_queue is not None:
        draft, tokens = await _llm_call_streaming(_DRAFT_SYSTEM, draft_prompt, model_tier, stream_queue)
    else:
        draft, tokens = await _llm_call(_DRAFT_SYSTEM, draft_prompt, model_tier)
    total_tokens += tokens

    if not draft:
        # Surface this as a real pipeline error rather than an empty-but-
        # "completed" run. A rate-limited draft used to return report="" with
        # error=None, so the API stored status=completed and the UI rendered a
        # blank page with no explanation — a failure disguised as success.
        out = _empty("draft generation failed", t0)
        out["error"] = (
            "Report generation failed: the language model was rate-limited and "
            "did not return a draft. Retry in a minute."
        )
        return out

    # ── Step 3: Verify ────────────────────────────────────────────────────
    # Use chunk_text + extra_evidence — the exact same evidence the draft was
    # given, not chunk_text alone. Verify previously only saw chunk_text, so a
    # correctly-grounded fact the draft pulled from findings_text/sentiment_summary/
    # validations_text (e.g. a validated growth rate) had nothing for verify to
    # confirm it against and was deleted by rule 3's "not explicitly stated in
    # evidence" — a real bug, not just an eval-measurement gap: it could strip
    # correct content from the live report. QUESTION included so the refusal
    # check (rule 0) has something to check against — verify previously never
    # saw it at all.
    findings_text, sentiment_summary, validations_text = extra_evidence
    verify_prompt = (
        f"QUESTION: {state.get('query', '')}\n\n"
        f"DRAFT REPORT:\n{draft}\n\n"
        f"SOURCE EVIDENCE:\n{chunk_text}\n\n"
        f"=== LANGUAGE SHIFT ANALYSIS ===\n{findings_text}\n\n"
        f"=== SENTIMENT ANALYSIS (FinBERT) ===\n{sentiment_summary}\n\n"
        f"=== NUMERIC VALIDATION ===\n{validations_text}"
    )

    # Deterministic pre-check — cheaper than asking the model to discover it
    # has nothing to do. Evidence scanned is the same text verify would read.
    needs_verify, reason = _verify_needed(
        draft,
        f"{chunk_text}\n{findings_text}\n{sentiment_summary}\n{validations_text}",
        state.get("query", ""),
    )

    verify_skipped = not needs_verify
    if needs_verify:
        if stream_queue is not None:
            await stream_queue.put({"type": "verifying"})
        verified_report, tokens = await _llm_call(_VERIFY_SYSTEM, verify_prompt, model_tier)
        total_tokens += tokens
        final_report = verified_report or draft
    else:
        print(f"[report_agent] verify skipped — {reason}")
        final_report = draft

    if stream_queue is not None:
        await stream_queue.put({"type": "final", "report": final_report})

    entry: DecisionLogEntry = {
        "agent": "report_agent",
        "tool_called": "crewai_bull_bear_debate",
        "input_summary": (
            f"chunks={len(state.get('retrieval_results', []))} "
            f"comparisons={len(state.get('comparison_findings', []))} "
            f"sentiments={len(state.get('sentiment_scores', []))} "
            f"validations={len(state.get('numeric_validations', []))} "
            f"debate={'yes' if debate_summary else 'failed'}"
        ),
        "output_summary": (
            f"report drafted, verify={'skipped' if verify_skipped else 'run'} "
            f"({reason}), tier={model_tier}, len={len(final_report)}"
        ),
        "confidence": None,
        "tokens_used": total_tokens,
        "latency_ms": ms(t0),
    }

    return {
        "report": final_report,
        "decision_log_entries": [entry],
    }


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_chunk_text(state: GraphState, max_chunks: int = 8) -> str:
    """
    Build the full chunk text payload used by both draft and verify steps.
    Preserves the global reranking order from retrieval_agent.
    Each chunk tagged with source type for citation tracking.
    """
    chunks = state.get("retrieval_results") or []
    return "\n\n".join(
        f"[{r['doc_type'].upper()}] {r.get('parent_content') or r['content']}"
        for r in chunks[:max_chunks]
    )


def _build_extra_evidence(state: GraphState) -> tuple[str, str, str]:
    """
    Language-shift / sentiment / numeric-validation text blocks the draft is given
    alongside chunk_text. Returns (findings_text, sentiment_summary, validations_text)
    so the caller can also pass them to verify — verify previously only saw
    chunk_text, so any correctly-grounded fact the draft pulled from one of these
    three (e.g. a validated growth rate) had no evidence to be confirmed against
    and was deleted by verify's own "not explicitly stated in evidence" rule.
    """
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

    return findings_text, sentiment_summary, validations_text


def _build_draft_prompt(
    state: GraphState, chunk_text: str, extra_evidence: tuple[str, str, str],
    debate_summary: str = "",
) -> str:
    company = state["company"]
    quarter = state["quarter"]
    query = state.get("query", "")
    findings_text, sentiment_summary, validations_text = extra_evidence

    debate_section = (
        f"\n=== BULL/BEAR DEBATE (CrewAI) ===\n{debate_summary}\n"
        if debate_summary else ""
    )

    return f"""COMPANY: {company}
QUARTER: {quarter}
QUESTION: {query}

=== RETRIEVED EVIDENCE ===
{chunk_text}

=== LANGUAGE SHIFT ANALYSIS ===
{findings_text}

=== SENTIMENT ANALYSIS (FinBERT) ===
{sentiment_summary}

=== NUMERIC VALIDATION ===
{validations_text}
{debate_section}
Draft the analyst earnings intelligence briefing. Answer QUESTION directly first, then support it."""


def _build_evidence_summary(state: GraphState) -> str:
    """
    Compact evidence summary for the CrewAI debate input only.
    Debate agents need a shorter context — 300 chars per chunk is sufficient
    for bull/bear framing. Full chunk_text is used for draft + verify.
    """
    lines: list[str] = []
    for r in (state.get("retrieval_results") or [])[:10]:
        lines.append(f"[{r['doc_type'].upper()}] {(r.get('parent_content') or r['content'])[:300]}")
    for v in (state.get("numeric_validations") or []):
        lines.append(
            f"[VALIDATED] {v['metric']}: claimed={v['claimed_value']} "
            f"calc={v['calculated_value']} match={v['match']}"
        )
    return "\n\n".join(lines)


# ── Async LLM wrapper ─────────────────────────────────────────────────────────

async def _llm_call(system: str, user: str, model_tier: str = "primary") -> tuple[str, int]:
    tier = model_tier
    tried_fallback = False
    attempt = 0
    while attempt < _MAX_RETRIES:
        attempt += 1
        try:
            response = await openai_client.achat_tiered(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                model_tier=tier,
            )
            text = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            return text, tokens
        except openai.RateLimitError as exc:
            fallback = _fallback_tier(tier) if not tried_fallback else None
            if fallback:
                tried_fallback = True
                tier = fallback
                attempt = 0  # fresh budget on the higher-headroom deployment
                print(f"[report_agent] {model_tier} rate-limited — switching to {fallback}")
                continue
            if attempt >= _MAX_RETRIES:
                print(f"[report_agent] LLM call rate-limited, giving up after {_MAX_RETRIES} attempts: {exc}")
                return "", 0
            wait = _retry_after_seconds(exc) or _RETRY_BACKOFF * (2 ** (attempt - 1))
            print(f"[report_agent] LLM call rate-limited (attempt {attempt}/{_MAX_RETRIES}) — retrying in {wait:.0f}s")
            await asyncio.sleep(wait)
        except Exception as exc:
            print(f"[report_agent] LLM call failed: {exc}")
            return "", 0
    return "", 0


async def _llm_call_streaming(
    system: str, user: str, model_tier: str, stream_queue,
) -> tuple[str, int]:
    """
    Same contract and retry policy as _llm_call, but pushes each text delta
    to stream_queue as {"type": "draft_token", "text": delta} while the
    response is still being generated, instead of returning only once the
    full text is back. Only used for the draft step, and only when a
    consumer is actually listening (api/routes/analysis.py's SSE endpoint).

    No token-usage figure is available in streaming mode (the SDK only
    returns usage on the final non-streamed response); returns 0, same as
    every other failure path here — total_tokens is already best-effort.
    """
    tier = model_tier
    tried_fallback = False
    attempt = 0
    while attempt < _MAX_RETRIES:
        attempt += 1
        chunks: list[str] = []
        try:
            async for delta in openai_client.achat_tiered_stream(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                model_tier=tier,
            ):
                chunks.append(delta)
                await stream_queue.put({"type": "draft_token", "text": delta})
            return "".join(chunks), 0
        except openai.RateLimitError as exc:
            # A 429 mid-stream (rare — the request was already accepted, but
            # possible) leaves partial text already pushed to the queue. Any
            # retry re-generates from scratch, so the consumer must discard
            # what it has rather than append the retry on top of it.
            if chunks:
                await stream_queue.put({"type": "draft_reset"})

            fallback = _fallback_tier(tier) if not tried_fallback else None
            if fallback:
                tried_fallback = True
                tier = fallback
                attempt = 0
                print(f"[report_agent] {model_tier} rate-limited — switching to {fallback}")
                continue
            if attempt >= _MAX_RETRIES:
                print(f"[report_agent] streaming LLM call rate-limited, giving up after {_MAX_RETRIES} attempts: {exc}")
                return "", 0
            wait = _retry_after_seconds(exc) or _RETRY_BACKOFF * (2 ** (attempt - 1))
            print(f"[report_agent] streaming LLM call rate-limited (attempt {attempt}/{_MAX_RETRIES}) — retrying in {wait:.0f}s")
            await asyncio.sleep(wait)
        except Exception as exc:
            print(f"[report_agent] streaming LLM call failed: {exc}")
            return "", 0
    return "", 0  # unreachable — loop always returns or retries


def _empty(reason: str, t0: float) -> dict:
    return skipped("report_agent", "report", "", reason, t0)
