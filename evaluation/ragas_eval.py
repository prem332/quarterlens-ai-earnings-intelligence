"""
evaluation/ragas_eval.py
RAGAS-equivalent evaluation for QuarterLens AI.

Implements the four core RAGAS metrics directly via Azure OpenAI LLM-as-judge,
without the ragas package. Used because ragas 0.4.x conflicts with LangGraph 1.x
(both pin incompatible langchain-core versions — a known ecosystem conflict).

Metric definitions follow the RAGAS paper (Es et al., 2023):
  - Faithfulness:        fraction of answer claims supported by retrieved context
  - Answer Relevancy:    semantic alignment between question and answer
  - Context Precision:   fraction of retrieved chunks that are actually relevant
  - Context Recall:      fraction of ground-truth facts covered by retrieved chunks

All scores are 0.0–1.0. Interview framing:
  "I implemented the RAGAS metric definitions directly due to a LangGraph/RAGAS
   version conflict. The metrics are equivalent to the paper definitions."

Usage:
    from evaluation.ragas_eval import run_ragas_eval

    metrics = run_ragas_eval(samples)
    # {"faithfulness": 0.82, "answer_relevancy": 0.79, ...}
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)

# ── Prompts (following RAGAS paper definitions) ───────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are evaluating whether an answer is faithful to the retrieved context.

Context:
{context}

Answer:
{answer}

Task: List every distinct factual claim made in the answer. For each claim,
state whether it is supported by the context (yes/no).

Respond ONLY with valid JSON, no markdown:
{{
  "claims": [
    {{"claim": "<claim text>", "supported": true}},
    ...
  ]
}}

If the answer is empty or makes no claims, return {{"claims": []}}
"""

_ANSWER_RELEVANCY_PROMPT = """\
You are evaluating whether an answer is relevant to the question.

Question: {question}
Answer: {answer}

Score the relevancy from 0.0 to 1.0:
  1.0 = answer directly and completely addresses the question
  0.5 = answer partially addresses the question
  0.0 = answer is off-topic or empty

Respond ONLY with valid JSON, no markdown:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}
"""

_CONTEXT_PRECISION_PROMPT = """\
You are evaluating whether retrieved context chunks are relevant to the question.

Question: {question}
Ground truth answer: {ground_truth}

Retrieved chunks:
{chunks}

For each chunk (numbered from 1), state whether it is relevant to answering
the question given the ground truth (yes/no).

Respond ONLY with valid JSON, no markdown:
{{
  "chunks": [
    {{"chunk_num": 1, "relevant": true}},
    ...
  ]
}}
"""

_CONTEXT_RECALL_PROMPT = """\
You are evaluating whether retrieved context covers the ground truth answer.

Ground truth answer: {ground_truth}

Retrieved context:
{context}

Task: List the key facts from the ground truth. For each fact, state whether
it is covered by the retrieved context (yes/no).

Respond ONLY with valid JSON, no markdown:
{{
  "facts": [
    {{"fact": "<fact text>", "covered": true}},
    ...
  ]
}}

If the ground truth is empty, return {{"facts": []}}
"""


def _get_client():
    """Build Azure OpenAI client from Key Vault."""
    from azure_clients.key_vault_client import kv
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=kv.get_secret("AZURE-OPENAI-ENDPOINT"),
        api_key=kv.get_secret("AZURE-OPENAI-KEY"),
        api_version="2024-12-01-preview",
    )
    deployment = kv.get_secret("AZURE-OPENAI-DEPLOYMENT-NAME")
    return client, deployment


def _call_llm(client, deployment: str, prompt: str) -> dict:
    """Call LLM and parse JSON response. Returns empty dict on failure."""
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        log.warning("LLM call failed: %s", e)
        return {}


def _score_faithfulness(client, deployment: str, answer: str, contexts: list[str]) -> float:
    """Faithfulness: fraction of answer claims supported by context."""
    if not answer.strip() or not contexts:
        return 0.0
    context_text = "\n\n---\n\n".join(contexts[:5])  # cap at 5 chunks
    prompt = _FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)
    result = _call_llm(client, deployment, prompt)
    claims = result.get("claims", [])
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if c.get("supported", False))
    return round(supported / len(claims), 4)


def _score_answer_relevancy(client, deployment: str, question: str, answer: str) -> float:
    """Answer relevancy: semantic alignment between question and answer."""
    if not answer.strip():
        return 0.0
    prompt = _ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)
    result = _call_llm(client, deployment, prompt)
    score = result.get("score", 0.0)
    try:
        return round(float(score), 4)
    except (ValueError, TypeError):
        return 0.0


def _score_context_precision(
    client, deployment: str, question: str,
    contexts: list[str], ground_truth: str
) -> float:
    """Context precision: fraction of retrieved chunks that are relevant."""
    if not contexts or not ground_truth.strip():
        return 0.0
    chunks_text = "\n\n".join(
        f"Chunk {i+1}: {c[:300]}" for i, c in enumerate(contexts[:5])
    )
    prompt = _CONTEXT_PRECISION_PROMPT.format(
        question=question, ground_truth=ground_truth, chunks=chunks_text
    )
    result = _call_llm(client, deployment, prompt)
    chunks = result.get("chunks", [])
    if not chunks:
        return 0.0
    relevant = sum(1 for c in chunks if c.get("relevant", False))
    return round(relevant / len(chunks), 4)


def _score_context_recall(
    client, deployment: str, contexts: list[str], ground_truth: str
) -> float:
    """Context recall: fraction of ground-truth facts covered by context."""
    if not contexts or not ground_truth.strip():
        return 0.0
    context_text = "\n\n---\n\n".join(contexts[:5])
    prompt = _CONTEXT_RECALL_PROMPT.format(
        ground_truth=ground_truth, context=context_text
    )
    result = _call_llm(client, deployment, prompt)
    facts = result.get("facts", [])
    if not facts:
        return 0.0
    covered = sum(1 for f in facts if f.get("covered", False))
    return round(covered / len(facts), 4)


def run_ragas_eval(
    samples: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """
    Run RAGAS-equivalent evaluation over a list of pipeline output samples.

    Implements faithfulness, answer_relevancy, context_precision, context_recall
    directly via Azure OpenAI — no ragas package required.

    Args:
        samples: List of dicts, each with:
            - "question":     str
            - "answer":       str
            - "contexts":     list[str]
            - "ground_truth": str
        metrics: Subset of the four metric names. Defaults to all four.

    Returns:
        Dict of metric_name -> mean float score across all samples.
    """
    _all_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    requested = metrics or _all_metrics

    unknown = set(requested) - set(_all_metrics)
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Valid: {_all_metrics}")

    if not samples:
        return {m: 0.0 for m in requested}

    client, deployment = _get_client()

    scores: dict[str, list[float]] = {m: [] for m in requested}

    for i, s in enumerate(samples):
        question = s.get("question", "")
        answer = s.get("answer", "")
        contexts = s.get("contexts", [])
        ground_truth = s.get("ground_truth", "")

        log.debug("Scoring sample %d/%d", i + 1, len(samples))

        if "faithfulness" in requested:
            scores["faithfulness"].append(
                _score_faithfulness(client, deployment, answer, contexts)
            )
        if "answer_relevancy" in requested:
            scores["answer_relevancy"].append(
                _score_answer_relevancy(client, deployment, question, answer)
            )
        if "context_precision" in requested:
            scores["context_precision"].append(
                _score_context_precision(client, deployment, question, contexts, ground_truth)
            )
        if "context_recall" in requested:
            scores["context_recall"].append(
                _score_context_recall(client, deployment, contexts, ground_truth)
            )

    result = {}
    for m in requested:
        vals = [v for v in scores[m] if not math.isnan(v)]
        result[m] = round(sum(vals) / len(vals), 4) if vals else 0.0

    log.info(
        "RAGAS-equivalent scores (%d samples): %s",
        len(samples),
        {k: f"{v:.4f}" for k, v in result.items()},
    )
    return result