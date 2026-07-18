"""
agents/report_agent.py

Three-step:
  1. Bull/Bear debate: CrewAI two-agent debate over the retrieved evidence
     (bull analyst vs bear analyst) — surfaces competing interpretations
     before the draft is written.
  2. Draft: LLM synthesises all agent outputs + debate into analyst-tone briefing.
  3. Verify: LLM checks every factual claim in the draft traces back to a
     retrieved chunk or a validated numeric fact.

gpt-5.4-mini note: reasoning model — max_completion_tokens must be >= 4096.
The openai_client wrapper enforces this minimum automatically.
"""

import os
import asyncio
import time
from crewai import Agent, Task, Crew, LLM
from azure_clients.key_vault_client import kv
from graph.state import (
    GraphState, DecisionLogEntry,
    ComparisonFinding, SentimentScore, NumericValidation,
)
from azure_clients.openai_client import openai_client


_DRAFT_SYSTEM = """\
You are a senior equity research analyst writing earnings intelligence briefings for institutional investors.

TONE:
- Professional, direct, assertive — not hedged or generic
- Use active voice: "Revenue grew 10%" not "Revenue was reported to have grown"
- No filler phrases: never use "it is worth noting", "importantly", "it should be mentioned"
- No financial advice, no buy/sell/hold recommendations

STRUCTURE (always follow this exact order):
## Executive Summary (2-3 sentences max — the single most important takeaway)
## Key Financial Metrics (bullet points, one metric per line)
## Guidance & Language Shifts (what changed vs prior quarter, be specific)
## Risk Factor Changes (what's new or dropped)
## Sentiment Overview (FinBERT-based, cite specific passages)
## Bull/Bear Perspectives
## Source Citations (list every [FILING] and [TRANSCRIPT] reference used)

FORMATTING RULES:
- Never write paragraphs longer than 3 sentences
- Use bullet points for lists of 3+ items
- Every number must have a unit: "$94.0B" not "94036"
- Every factual claim tagged: [FILING] or [TRANSCRIPT]
- No hallucinated facts — ONLY state what is explicitly present in the retrieved evidence
- Dates always in format: Q3 FY2025, not "third quarter of fiscal year 2025"
- If a section has no evidence, write "No data available." — do not invent content

LENGTH: 600-800 words total. No more, no less.

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
You will be given a DRAFT REPORT and the SOURCE EVIDENCE it was drawn from.

YOUR TASK — apply these rules in order:

1. IDENTIFY every factual claim in the draft (numbers, percentages, growth rates, quotes, guidance statements).

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
   - If a section becomes empty after removing unsupported claims, write "No verified data available."
   - Do NOT add new content — only remove unsupported claims

CRITICAL: When in doubt, DELETE. A shorter grounded report scores higher than a longer hallucinated one."""


# ── CrewAI LLM factory ────────────────────────────────────────────────────────

def _make_crewai_llm(model_tier: str) -> LLM:
    """
    Build a CrewAI LLM pointed at Azure OpenAI.
    Uses the same tier routing as the rest of the pipeline.
    Sets AZURE_ENDPOINT env var so CrewAI's Azure provider can find it —
    sourced from Key Vault, works identically in local dev and Container Apps.
    """
    endpoint = kv.get_secret("AZURE-OPENAI-ENDPOINT")
    os.environ["AZURE_ENDPOINT"] = endpoint  # CrewAI Azure provider requires this

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

def _build_debate_crew(evidence_summary: str, model_tier: str) -> Crew:
    """
    Build a two-agent CrewAI crew for bull/bear debate.
    Bull agent argues the positive case; bear agent argues the critical/risk case.
    Sequential execution: bull first, bear responds.
    """
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

    return Crew(
        agents=[bull_analyst, bear_analyst],
        tasks=[bull_task, bear_task],
        verbose=False,
    )


def _run_debate_sync(evidence_summary: str, model_tier: str) -> str:
    """
    Synchronous CrewAI kickoff — wrapped in asyncio.to_thread by the caller.
    Returns formatted bull/bear debate summary string.
    """
    try:
        crew = _build_debate_crew(evidence_summary, model_tier)
        result = crew.kickoff()
        # result.tasks_output is a list of TaskOutput objects
        outputs = result.tasks_output if hasattr(result, "tasks_output") else []
        if len(outputs) >= 2:
            bull_text = outputs[0].raw if hasattr(outputs[0], "raw") else str(outputs[0])
            bear_text = outputs[1].raw if hasattr(outputs[1], "raw") else str(outputs[1])
            return f"=== BULL CASE ===\n{bull_text}\n\n=== BEAR CASE ===\n{bear_text}"
        return str(result)
    except Exception as exc:
        print(f"[report_agent] CrewAI debate failed (non-fatal): {exc}")
        return ""


# ── Main agent node ───────────────────────────────────────────────────────────

async def report_agent(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    t0 = time.time()
    total_tokens = 0
    # Use report_model_tier if set — isolates finetuned model to report_agent only.
    # Falls back to model_tier for backward compatibility.
    model_tier = state.get("report_model_tier") or state.get("model_tier", "primary")

    # ── Step 1: Bull/Bear debate (CrewAI) ─────────────────────────────────
    evidence_summary = _build_evidence_summary(state)
    debate_summary = await asyncio.to_thread(
        _run_debate_sync, evidence_summary, model_tier
    )

    # ── Step 2: Draft ─────────────────────────────────────────────────────
    draft_prompt = _build_draft_prompt(state, debate_summary)
    draft, tokens = await _llm_call(_DRAFT_SYSTEM, draft_prompt, model_tier)
    total_tokens += tokens

    if not draft:
        return _empty("draft generation failed", t0)

    # ── Step 3: Verify ────────────────────────────────────────────────────
    verify_prompt = (
        f"DRAFT REPORT:\n{draft}\n\n"
        f"SOURCE EVIDENCE:\n{evidence_summary}"
    )
    verified_report, tokens = await _llm_call(_VERIFY_SYSTEM, verify_prompt, model_tier)
    total_tokens += tokens

    final_report = verified_report or draft

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

def _build_draft_prompt(state: GraphState, debate_summary: str = "") -> str:
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

    debate_section = (
        f"\n=== BULL/BEAR DEBATE (CrewAI) ===\n{debate_summary}\n"
        if debate_summary else ""
    )

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
{debate_section}
Draft the analyst earnings intelligence briefing based on the above."""


def _build_evidence_summary(state: GraphState) -> str:
    """Compact evidence summary for debate input and verification step."""
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

async def _llm_call(system: str, user: str, model_tier: str = "primary") -> tuple[str, int]:
    """
    Async tiered LLM call via openai_client.achat_tiered(). Returns (text, token_count).
    """
    try:
        response = await openai_client.achat_tiered(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model_tier=model_tier,
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