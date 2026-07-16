"""
evaluation/llm_as_judge.py
LLM-as-judge scoring for QuarterLens AI.

Scores pipeline outputs on a 1–5 scale across three dimensions:
  - accuracy:    Does the answer correctly reflect the filing/transcript?
  - grounding:   Is every claim traceable to the retrieved context?
  - refusal:     Did the system correctly refuse out-of-scope / advice-bait?

Judge model: gpt-5-mini via Azure OpenAI (same as the pipeline — Phase 1
constraint; a stronger judge is a Phase 2 ablation entry).

Usage:
    from evaluation.llm_as_judge import judge_sample, judge_batch

    score = judge_sample(
        question="What was Apple's revenue growth?",
        answer="Apple's revenue grew 5% YoY...",
        contexts=["...retrieved chunk text..."],
        ground_truth="Revenue grew 5% year-over-year...",
        claim_type="numeric",
    )
    # score: {"accuracy": 4, "grounding": 5, "overall": 4.5, "reasoning": "..."}
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are an expert evaluator for an earnings intelligence system.
Score the system's answer on the dimensions below. Return ONLY valid JSON.

## Question
{question}

## Retrieved context (what the system had access to)
{contexts}

## Ground truth answer
{ground_truth}

## System answer
{answer}

## Claim type
{claim_type}

## Scoring dimensions (1=very poor, 5=excellent)

accuracy:  Does the answer correctly reflect the ground truth?
           For numeric claims, is the figure exact (or within stated tolerance)?
           For out_of_scope claims, did the system correctly refuse?
grounding: Is every claim in the answer traceable to the retrieved context?
           Penalise hallucinated facts not present in the context.

## Output format (JSON only, no markdown, no preamble)
{{
  "accuracy": <int 1-5>,
  "grounding": <int 1-5>,
  "reasoning": "<one sentence explaining the scores>"
}}
"""


def _get_client():
    from azure_clients.key_vault_client import kv
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=kv.get_secret("AZURE-OPENAI-ENDPOINT"),
        api_key=kv.get_secret("AZURE-OPENAI-KEY"),
        api_version="2024-10-21",
    ), kv.get_secret("AZURE-OPENAI-DEPLOYMENT-NAME")


def judge_sample(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    claim_type: str = "retrieval",
) -> dict[str, Any]:
    """
    Score a single pipeline output with LLM-as-judge.

    Args:
        question:     The original query.
        answer:       The pipeline's generated answer.
        contexts:     Retrieved chunk texts the pipeline had access to.
        ground_truth: Expected answer from the golden dataset.
        claim_type:   Claim type from the golden schema (affects scoring criteria).

    Returns:
        Dict with keys: accuracy (int), grounding (int),
        overall (float), reasoning (str), error (str|None).
    """
    client, deployment = _get_client()

    prompt = _JUDGE_PROMPT.format(
        question=question,
        contexts="\n\n---\n\n".join(contexts) if contexts else "(none)",
        ground_truth=ground_truth,
        answer=answer,
        claim_type=claim_type,
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        accuracy = int(parsed.get("accuracy", 1))
        grounding = int(parsed.get("grounding", 1))
        overall = round((accuracy + grounding) / 2, 2)

        return {
            "accuracy": accuracy,
            "grounding": grounding,
            "overall": overall,
            "reasoning": parsed.get("reasoning", ""),
            "error": None,
        }

    except json.JSONDecodeError as e:
        log.warning("Judge returned non-JSON: %s — raw: %s", e, raw[:200])
        return {"accuracy": 0, "grounding": 0, "overall": 0.0,
                "reasoning": "", "error": f"json_parse_error: {e}"}
    except Exception as e:
        log.warning("Judge call failed: %s", e)
        return {"accuracy": 0, "grounding": 0, "overall": 0.0,
                "reasoning": "", "error": str(e)}


def judge_batch(
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """
    Score a batch of samples and return per-sample scores + mean overall.

    Args:
        samples: List of dicts, each with keys:
                 question, answer, contexts, ground_truth, claim_type.

    Returns:
        (per_sample_scores, mean_overall_score)
    """
    results = []
    for s in samples:
        score = judge_sample(
            question=s["question"],
            answer=s["answer"],
            contexts=s.get("contexts", []),
            ground_truth=s.get("ground_truth", ""),
            claim_type=s.get("claim_type", "retrieval"),
        )
        score["claim_id"] = s.get("claim_id", "unknown")
        results.append(score)

    valid = [r["overall"] for r in results if r["error"] is None]
    mean_overall = round(sum(valid) / len(valid), 4) if valid else 0.0

    log.info(
        "LLM-as-judge: %d samples scored, mean_overall=%.4f (%d errors)",
        len(results), mean_overall, sum(1 for r in results if r["error"]),
    )
    return results, mean_overall