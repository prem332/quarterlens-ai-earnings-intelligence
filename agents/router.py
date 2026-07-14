"""
agents/router.py — Model routing logic (Phase 2).

Phase 1: not implemented. Routing decisions are made statically in
supervisor.py (route_after_init always returns "retrieval_agent").

Phase 2 plan (per ARCHITECTURE.md §3 — Model Routing):
  - Lightweight classifier inspects the incoming query before the Supervisor
    decides which model tier to use.
  - Simple lookups → gpt-5-mini, may bypass full 5-agent pipeline.
  - Comparison/contradiction reasoning, final report drafting → larger model (TBD).
  - Sentiment → FinBERT (no LLM regardless of routing).
  - Numeric validation → deterministic tool (no LLM regardless of routing).
  - Must be validated as an explicit MLflow ablation entry:
    baseline (all-large-model) vs. routed — no silent quality degradation.
"""

# Phase 2 implementation goes here.