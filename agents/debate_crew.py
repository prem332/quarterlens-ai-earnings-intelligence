"""
agents/debate_crew.py — CrewAI multi-agent bull/bear debate.

This is the project's CrewAI component. It used to run inside report_agent on
every analysis, adding roughly 11s and two LLM calls to the critical path for
output that the draft step only used as background framing. It now runs
ON DEMAND against an already-completed run (POST /api/analysis/{run_id}/debate),
so users who want the investment-debate view get it, and users who just want
the verified report don't pay for it.

Architecture note (matches CLAUDE.md's "LangGraph is sole orchestrator"):
LangGraph still owns the analysis pipeline end to end. CrewAI is used only
here, for a genuinely multi-agent task -- two independent analyst personas
arguing opposite sides of the same evidence -- which is the shape of problem
CrewAI is actually for.

Bull and bear are independent (neither reads the other's output), so they run
as two single-agent crews concurrently rather than one sequential Crew. That
is deliberate: CrewAI's default Process.sequential would run them back to
back, serializing two calls that have nothing to wait on each other for.
"""

from __future__ import annotations

import asyncio
import os

from crewai import Agent, Task, Crew, LLM

from azure_clients.key_vault_client import kv
from graph.state import GraphState


def make_crewai_llm(model_tier: str = "primary") -> LLM:
    """
    get_secret_cached, not get_secret. kv.get_secret hits Key Vault over the
    network on EVERY call -- measured at ~575ms each -- so this factory was
    costing ~1.5s per invocation purely in secret lookups, and it runs once
    per analyst crew (bull and bear both call it). get_secret_cached is
    ~0ms after the first read.

    These four values are deployment configuration, not rotating credentials,
    so process-lifetime caching is appropriate here (that caveat is why
    get_secret_cached exists as a separate method rather than being the
    default).
    """
    endpoint = kv.get_secret_cached("AZURE-OPENAI-ENDPOINT")
    os.environ["AZURE_ENDPOINT"] = endpoint

    deployment = (
        kv.get_secret_cached("AZURE-OPENAI-DEPLOYMENT-NAME-STANDARD")
        if model_tier == "standard"
        else kv.get_secret_cached("AZURE-OPENAI-DEPLOYMENT-NAME")
    )
    return LLM(
        model=f"azure/{deployment}",
        api_key=kv.get_secret_cached("AZURE-OPENAI-KEY"),
        api_base=endpoint,
        api_version="2024-12-01-preview",
    )


def _run_bull_sync(evidence_summary: str, model_tier: str) -> str:
    try:
        llm = make_crewai_llm(model_tier)
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
        print(f"[debate_crew] CrewAI bull analysis failed (non-fatal): {exc}")
        return ""


def _run_bear_sync(evidence_summary: str, model_tier: str) -> str:
    try:
        llm = make_crewai_llm(model_tier)
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
        print(f"[debate_crew] CrewAI bear analysis failed (non-fatal): {exc}")
        return ""


def build_evidence_summary(state: GraphState | dict) -> str:
    """
    Compact evidence summary for the debate agents. 300 chars per chunk is
    enough for bull/bear framing; the full chunk text is what report_agent's
    draft and verify steps use.

    Accepts either a live GraphState or a stored run document, since the
    on-demand endpoint reads a completed run back out of blob storage.
    """
    lines: list[str] = []
    for r in (state.get("retrieval_results") or [])[:10]:
        body = r.get("parent_content") or r.get("content") or ""
        lines.append(f"[{(r.get('doc_type') or '').upper()}] {body[:300]}")
    for v in (state.get("numeric_validations") or []):
        lines.append(
            f"[VALIDATED] {v.get('metric')}: claimed={v.get('claimed_value')} "
            f"calc={v.get('calculated_value')} match={v.get('match')}"
        )
    return "\n\n".join(lines)


async def run_debate(evidence_summary: str, model_tier: str = "primary") -> dict:
    """
    Run both analyst crews concurrently.

    Returns {"bull": str, "bear": str}. Either side can be "" if that crew
    failed — the debate is supplementary, so a partial result is still worth
    showing rather than failing the whole request.
    """
    bull_text, bear_text = await asyncio.gather(
        asyncio.to_thread(_run_bull_sync, evidence_summary, model_tier),
        asyncio.to_thread(_run_bear_sync, evidence_summary, model_tier),
    )
    return {"bull": bull_text, "bear": bear_text}
