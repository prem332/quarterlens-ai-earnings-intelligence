"""
api/guardrails.py

Input guardrails for the free-text `query` field on AnalysisRequest — the
only user-supplied text in this app (company/quarter are enum-constrained,
see api/schemas/shared.py). Runs BEFORE the request is ever accepted into
the pipeline, i.e. before any Azure OpenAI call — a rejected query spends
zero tokens and is never sent to the LLM gateway at all.

Local regex/keyword heuristics only: no LLM call, no new Azure resource, a
few milliseconds of cost. Chosen over NeMo Guardrails / Azure AI Content
Safety deliberately — this app has exactly one free-text field with no
conversation history, and both of those alternatives either call an LLM
per check (reintroducing the latency/cost this is meant to avoid) or need
a new Azure resource. Revisit if the input surface grows beyond one field
or gains multi-turn state.

Checks, in order (first match wins, cheapest/most-certain first):
  1. PII              — email, phone, SSN, credit card patterns
  2. Prompt injection  — instruction-override / role-hijack / prompt-leak phrasing
  3. Harmful content   — violence, self-harm, hate, illegal-activity keyword sets
  4. Off-topic/out-of-scope — query must contain at least one earnings/financial
     term; this tool only ever answers questions about a filed 10-Q/10-K + call

Not exhaustive — regex/keyword matching is evadable by design (paraphrase,
encoding tricks, typos). This is a first line of defense, not the only one;
report_agent's own prompts are also written defensively (see _VERIFY_SYSTEM).
"""

import re
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    allowed: bool
    category: str | None = None   # "pii" | "prompt_injection" | "harmful_content" | "off_topic"
    reason: str | None = None


_PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
}

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all|any|the)?\s*(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(all|any|the)?\s*(previous|prior|above)", re.I),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"reveal\s+(your|the)\s+(system\s+)?prompt", re.I),
    re.compile(r"\bDAN\b"),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"</?(system|assistant|user)>", re.I),  # fake chat-role tags
    re.compile(r"\bnew\s+instructions\b", re.I),
    re.compile(r"act\s+as\s+(an?|the)\b.{0,40}\b(unfiltered|uncensored|no\s+restrictions)", re.I),
]

_HARMFUL_KEYWORDS = [
    "bomb", "weapon", "murder", "kill someone",
    "suicide", "self-harm", "self harm",
    "hate speech", "racial slur",
    "how to hack", "credit card fraud", "launder money",
]

_FINANCIAL_DOMAIN_TERMS = [
    "revenue", "margin", "earnings", "guidance", "growth", "quarter", "fiscal",
    "eps", "profit", "income", "segment", "forecast", "outlook", "sentiment",
    "claim", "filing", "10-q", "10-k", "10q", "10k", "transcript", "call",
    "cost", "expense", "cash flow", "operating", "gross", "net", "yoy", "qoq",
    "compare", "verify", "summar", "risk", "capex", "spend", "backlog", "demand",
    "management", "executive", "management's discussion", "mda", "balance sheet",
    # Broadened after a real false positive on a legitimate earnings-call
    "leadership", "ceo", "cfo", "investor", "shareholder", "stock", "share",
    "investment", "invest", "ai ", " ai", "artificial intelligence", "infrastructure",
    "data center", "datacenter", "chip", "semiconductor", "supply chain",
    "customer", "subscriber", "competition", "competitor", "market share",
    "buyback", "dividend", "headcount", "hiring", "layoff", "product", "launch",
    "pricing", "price", "tariff", "regulation", "macro", "cloud", "platform",
    "engagement", "advertising", "monetiz", "retention", "capacity", "utilization",
    "r&d", "research and development", "partnership", "acquisition", "strategy",
    "strategic", "priorit", "business", "company", "plan",
    # Broadened again after a second real false positive: "3.5 billion people
    # using at least one of our apps every day" (a real Meta DAP metric quote)
    # — a legitimate operational-KPI claim with no obvious finance keyword.
    # Engagement/usage metrics are routinely discussed on earnings calls
    # without ever using words like "revenue" or "quarter".
    "user", "users", "app", "apps", "daily active", "monthly active",
    "dau", "mau", "dap", "active people", "install", "downloads",
]


def _check_pii(query: str) -> GuardrailResult | None:
    for name, pattern in _PII_PATTERNS.items():
        if pattern.search(query):
            return GuardrailResult(False, "pii", f"query appears to contain a {name.replace('_', ' ')}")
    return None


def _check_injection(query: str) -> GuardrailResult | None:
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(False, "prompt_injection", "query matches a known instruction-override pattern")
    return None


def _check_harmful(query: str) -> GuardrailResult | None:
    lowered = query.lower()
    for kw in _HARMFUL_KEYWORDS:
        if kw in lowered:
            return GuardrailResult(False, "harmful_content", f"query contains a disallowed term ('{kw}')")
    return None


def _check_off_topic(query: str) -> GuardrailResult | None:
    lowered = query.lower()
    if not any(term in lowered for term in _FINANCIAL_DOMAIN_TERMS):
        return GuardrailResult(
            False, "off_topic",
            "query doesn't reference any earnings/financial-analysis terms — out of scope for this tool",
        )
    return None


# Order matters: cheapest/most-certain checks first, so an obviously-bad
# query short-circuits before the (slightly) fuzzier off-topic check runs.
_CHECKS = [_check_pii, _check_injection, _check_harmful, _check_off_topic]


def check_query(query: str) -> GuardrailResult:
    """Run all guardrail checks against a user-supplied query.

    Returns the first violation found, or an allowed=True result if the
    query clears every check.
    """
    for check in _CHECKS:
        result = check(query)
        if result is not None:
            return result
    return GuardrailResult(True)
