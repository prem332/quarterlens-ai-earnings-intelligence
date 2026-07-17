"""
agents/router.py — Model routing logic.

Classifies incoming queries into model tiers before the pipeline runs.
The tier is written to GraphState.model_tier by supervisor_init and read
by each agent's LLM call via openai_client.achat_tiered().

Tiers:
  "standard" → gpt-5-mini   (simple fact lookups, single-metric questions)
  "primary"  → gpt-5.4-mini (comparison, contradiction, report drafting — default)

Routing is keyword-based — fast, zero LLM cost, zero latency.
Validated as an explicit MLflow ablation entry:
  baseline (all-primary) vs. routed — measured on fixed golden dataset.

Agents unaffected by routing (always use their own model/tool):
  - sentiment_agent  → FinBERT always
  - numeric_validation_agent → deterministic SQL/Python always (LLM only for
    claim extraction, which routes normally)
"""

from __future__ import annotations

import re

# Compiled patterns for simple fact-lookup queries → standard tier
_STANDARD_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bwhat was\b",
        r"\bwhat is\b",
        r"\bhow much\b",
        r"\bhow many\b",
        r"\bwhen did\b",
        r"\bwhen was\b",
        r"\bwho is\b",
        r"\bwho was\b",
        r"\blist\b",
        r"\bstate the\b",
        r"\btell me the\b",
        r"\bwhat were\b",
        r"\bwhat did\b",
    ]
]

# Patterns that override to primary regardless of above matches
_PRIMARY_OVERRIDE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bcompare\b",
        r"\bshift\b",
        r"\bchange[d]?\b",
        r"\bcontradict\b",
        r"\bsummar\w+\b",       # summarize, summary
        r"\banalys\w+\b",       # analyse, analysis
        r"\banalyze\b",
        r"\bexplain\b",
        r"\bwhy\b",
        r"\bhow did\b",
        r"\bwhat caused\b",
        r"\bguidance\b",
        r"\boutlook\b",
        r"\brisk\b",
    ]
]


def classify_query(query: str) -> str:
    """
    Classify a query into a model tier.

    Args:
        query: The analyst's question or pipeline trigger string.

    Returns:
        "standard" — route to gpt-5-mini (simple fact lookup)
        "primary"  — route to gpt-5.4-mini (reasoning, comparison, report)
    """
    if not query or not query.strip():
        return "primary"

    # Primary override takes precedence — complex reasoning patterns
    for pattern in _PRIMARY_OVERRIDE_PATTERNS:
        if pattern.search(query):
            return "primary"

    # Standard tier — simple fact lookups
    for pattern in _STANDARD_PATTERNS:
        if pattern.search(query):
            return "standard"

    # Default: primary for anything ambiguous
    return "primary"