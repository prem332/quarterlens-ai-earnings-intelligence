"""
evaluation/llm_as_judge.py
LLM-as-judge scoring for QuarterLens AI.

Scores pipeline outputs on a 1–5 scale across three dimensions:
  - accuracy:    Does the answer correctly reflect the filing/transcript?
  - grounding:   Is every claim traceable to the retrieved context?
  - relevancy:   Does the answer directly address the question asked?

Claim-type-aware scoring:
  - numeric:      accuracy weighted heavily (exact figures matter)
  - out_of_scope: accuracy = did it correctly refuse?
  - comparison:   accuracy = did it detect/miss the shift correctly?
  - retrieval:    balanced across all three dimensions
  - sentiment:    accuracy = correct FinBERT label?

Judge model: gpt-5.4-mini (primary tier, reasoning model).
api_version: 2024-12-01-preview (required for reasoning models).
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
You are an expert evaluator for a financial earnings intelligence system.
Score the system's answer on the three dimensions below.
Be strict but fair — a score of 3 means acceptable, 4 means good, 5 means excellent.

## Question
{question}

## Claim type
{claim_type}

## Retrieved context (what the system had access to)
{contexts}

## Ground truth answer
{ground_truth}

## System answer
{answer}

## Scoring dimensions (1=very poor, 3=acceptable, 5=excellent)

accuracy:
  - numeric: Is the figure exact or within reasonable tolerance? Wrong number = 1.
  - out_of_scope: Did the system correctly refuse to answer? Refusal = 5, hallucinated answer = 1.
  - comparison: Did it correctly identify/miss the language shift?
  - retrieval/sentiment: Does the answer match the ground truth?

grounding:
  - Is every factual claim in the answer traceable to the retrieved context?
  - Penalise hallucinated facts not present in context. Empty answer = 1.
  - Well-cited, context-supported answer = 5.

relevancy:
  - Does the answer directly address what was asked?
  - Off-topic or generic answer = 1. Precise, query-focused answer = 5.
  - For out_of_scope: did it explain WHY it can't answer? Explanation = 5.

## Output format (JSON only, no markdown, no preamble)
{{
  "accuracy": <int 1-5>,
  "grounding": <int 1-5>,
  "relevancy": <int 1-5>,
  "reasoning": "<two sentences: what the answer got right and what it missed>"
}}
"""

# Per-claim-type dimension weights for overall score
_WEIGHTS: dict[str, dict[str, float]] = {
    "numeric":      {"accuracy": 0.6, "grounding": 0.3, "relevancy": 0.1},
    "out_of_scope": {"accuracy": 0.7, "grounding": 0.1, "relevancy": 0.2},
    "comparison":   {"accuracy": 0.4, "grounding": 0.3, "relevancy": 0.3},
    "retrieval":    {"accuracy": 0.3, "grounding": 0.4, "relevancy": 0.3},
    "sentiment":    {"accuracy": 0.5, "grounding": 0.3, "relevancy": 0.2},
}
_DEFAULT_WEIGHTS = {"accuracy": 0.4, "grounding": 0.3, "relevancy": 0.3}


def _get_client():
    from azure_clients.key_vault_client import kv
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=kv.get_secret("AZURE-OPENAI-ENDPOINT"),
        api_key=kv.get_secret("AZURE-OPENAI-KEY"),
        api_version="2024-12-01-preview",  # required for reasoning models
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

    Returns:
        Dict with keys: accuracy, grounding, relevancy (int 1-5),
        overall (float), reasoning (str), error (str|None).
    """
    client, deployment = _get_client()

    context_text = "\n\n---\n\n".join(contexts[:5]) if contexts else "(none retrieved)"

    prompt = _JUDGE_PROMPT.format(
        question=question,
        contexts=context_text,
        ground_truth=ground_truth,
        answer=answer if answer.strip() else "(empty — pipeline returned no answer)",
        claim_type=claim_type,
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,  # reasoning model needs headroom
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        accuracy  = max(1, min(5, int(parsed.get("accuracy",  1))))
        grounding = max(1, min(5, int(parsed.get("grounding", 1))))
        relevancy = max(1, min(5, int(parsed.get("relevancy", 1))))

        weights = _WEIGHTS.get(claim_type, _DEFAULT_WEIGHTS)
        overall = round(
            accuracy  * weights["accuracy"] +
            grounding * weights["grounding"] +
            relevancy * weights["relevancy"],
            2
        )

        return {
            "accuracy":  accuracy,
            "grounding": grounding,
            "relevancy": relevancy,
            "overall":   overall,
            "reasoning": parsed.get("reasoning", ""),
            "error":     None,
        }

    except json.JSONDecodeError as e:
        log.warning("Judge returned non-JSON: %s", e)
        return {"accuracy": 0, "grounding": 0, "relevancy": 0,
                "overall": 0.0, "reasoning": "", "error": f"json_parse_error: {e}"}
    except Exception as e:
        log.warning("Judge call failed: %s", e)
        return {"accuracy": 0, "grounding": 0, "relevancy": 0,
                "overall": 0.0, "reasoning": "", "error": str(e)}


def judge_batch(
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """
    Score a batch of samples and return per-sample scores + mean overall.

    Args:
        samples: List of dicts, each with keys:
                 question, answer, contexts, ground_truth, claim_type, claim_id.

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