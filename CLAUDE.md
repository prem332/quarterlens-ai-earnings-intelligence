# QuarterLens AI — Claude Code Instructions

## Claude Code Session Instructions

When starting a new session, follow this exact order:

### Step 1 — Read all code files
Read every file in these folders in order:
- `graph/state.py` — GraphState schema (start here, defines all data contracts)
- `agents/` — all 5 agent files (supervisor, retrieval, comparison, sentiment, numeric_validation, report)
- `tools/` — search_documents.py, rerank_documents.py, calculate_metric.py, run_finbert.py, fetch_prior_quarter.py
- `azure_clients/` — ai_search_client.py, openai_client.py, redis_client.py, key_vault_client.py
- `graph/build_graph.py` — LangGraph pipeline wiring
- `data_pipeline/chunking.py` — chunking strategy (Phase 4 structure-aware)
- `data_pipeline/embedding.py` — embedding pipeline
- `data_pipeline/indexer.py` — AI Search index schema
- `evaluation/run_baseline_eval.py` — eval runner with retrieval error analysis
- `evaluation/precision_recall_at_k.py` — retrieval metrics

### Step 2 — Read data samples to understand actual data structure
- `golden_dataset/claims/` — read 3-4 claim JSON files, one of each type:
  retrieval, comparison, numeric, sentiment
- `data/parsed/MSFT/FY2026-Q3_10Q.json` — parsed filing: section structure and raw text format
- `data/raw/transcripts/` — read one transcript JSON: speaker-turn structure
- `data/chunks/MSFT/` — read one chunk JSON: actual chunk sizes, boundaries, subsection metadata
- `evaluation/FINAL_REPORT.md` — the 7 locked metrics, their source runs, and the
  cache hit-rate breakdown

### Step 3 — Produce diagnosis before writing any code
After reading all files and data, diagnose:
1. Why context_precision is stuck at 0.25-0.30 despite precision@5=0.82
2. What the gap between precision@5 (0.82) and context_precision (0.25) means —
   are right chunks retrieved but ranked wrong, or is RAGAS measuring something else?
3. Whether the sentence-boundary chunking is splitting financial facts across chunk
   boundaries (compare parsed vs chunked data to verify)
4. Whether the duplicate_density=0.60 is true AI Search duplicates or semantically
   similar chunks from the same section (check chunk_id vs content similarity)
5. Ranked list of fixes by expected impact on context_precision with no recall@5
   regression risk

**Do not write any code until diagnosis is complete and confirmed.**

---

## Project Overview
Azure-native multi-agent earnings intelligence platform. Cross-verifies executive
earnings call claims against SEC-filed documents for 5 companies (AAPL, MSFT, NVDA,
GOOGL, META) across 5 fiscal quarters using a LangGraph 5-agent pipeline.

## Current Phase
**Phase 3/4 — Active optimization.** Retrieval and generation pipeline tuning.
Do not propose architectural changes without flagging as deviation from locked spec.

---

## Architecture (read before touching any file)

### Agent Pipeline (LangGraph)
```
supervisor → retrieval_agent → [comparison_agent || sentiment_agent] → numeric_validation_agent → report_agent
```
- `agents/supervisor.py` — pipeline entry, state initialization
- `agents/retrieval_agent.py` — global MMR + cross-encoder reranking
- `agents/comparison_agent.py` — language shift detection (LLM)
- `agents/sentiment_agent.py` — FinBERT sentiment (no LLM)
- `agents/numeric_validation_agent.py` — deterministic numeric verification
- `agents/report_agent.py` — CrewAI bull/bear debate + draft + verify

### Graph State
`graph/state.py` — single source of truth for all TypedDicts.
Key fields: `retrieval_results`, `transcript_retrieval_results`, `comparison_findings`,
`sentiment_scores`, `numeric_validations`, `report`, `decision_log_entries`

### Retrieval Pipeline
```
AI Search hybrid (BM25+vector) → global MMR (λ=0.5) → cross-encoder → top-5
```
- Filing pass (top-10) + transcript pass (top-10) → merge → global MMR → rerank
- `tools/search_documents.py` — raw hybrid search, no reranking
- `tools/rerank_documents.py` — cross-encoder/ms-marco-MiniLM-L-6-v2
- `tools/mmr_rerank` — in search_documents.py, public function

### Data Pipeline
```
chunking.py → embedding.py → indexer.py
```
- Structure-aware chunking (Phase 4): sentence-boundary, zero overlap, 400 tokens
- Index: `quarterlens-filings`, 2,240 chunks (25 filings + 25 transcripts)
- Embedding: text-embedding-3-small (1536-dim)

---

## Critical Constraints (never violate)

1. **Never name any folder `azure/`** — shadows Azure SDK namespace. Use `azure_clients/`
2. **`gpt-4o-mini` and `gpt-4.1-mini` are retired** (March 31, 2026) — do not reference
3. **`gpt-5.4-mini` requires** `api_version="2024-12-01-preview"`, `max_completion_tokens`
   (not `max_tokens`), minimum 4096 tokens
4. **LangGraph is sole orchestrator** — CrewAI only in report_agent bull/bear debate
5. **No architectural changes without explicit confirmation**
6. **Single-variable ablations only** — never compound changes before measuring
7. **Always flush Redis before eval runs** — cache key does not include pipeline config

---

## Azure Infrastructure

- **Resource group:** `quarterlens-phase1-rg`
- **AI Search:** `quarterlens-search`, Free F0, East US — index: `quarterlens-filings`
- **Azure OpenAI:** `quarterlens-openai`, East US
  - `gpt-5-mini` (dev, 10K TPM)
  - `gpt-5.4-mini` (production, Global Standard)
  - `text-embedding-3-small` (1536-dim)
- **Cosmos DB:** `quarterlens-cosmos`, NoSQL, West US 2 (decision log)
- **Key Vault:** `quarterlens-kv`, East US, RBAC — all secrets stored here (hyphen-named)
- **Azure SQL:** `quarterlens-sqlserver`, Central US, Serverless Free (financial_facts)
- **Redis:** Azure Cache Basic C0 (L2/L3 retrieval cache)
- **Blob Storage:** `quarterlensstorage`, container `raw-documents`

---

## Evaluation

### Locked Baselines

**`production-full-eval-75-fixes`** (current, 2026-08-02 — HEADLINE baseline, supersedes the
2026-08-01 run below). Full 75-claim golden dataset, real HTTP requests against the deployed
Container App, `no_cache=true` on every request, `REPORT_SKIP_VERIFY` at its production default.
Same script/methodology as the prior run (`evaluation/production_full_eval.py` pattern), rerun
after this session's fixes landed: the comparison_agent magnitude-instability fix (Known Issue
#4, prompt half), the faithfulness judge free-form-refusal fix (Known Issue #5), and the
guardrail domain-term broadening. Redis flushed immediately before running (mandatory — see
"Running Evaluations").
- faithfulness=0.9646 (▲ +0.029 vs 2026-08-01), answer_relevancy=0.9361 (▼ −0.026)
- context_precision=0.8176 (k=2/full-chunk window, ≈flat) | context_recall=0.8218 (▼ −0.017) /
  context_recall_excl_oos=0.8682 (▼ −0.024) — expected: nothing touched retrieval this session,
  see Known Issues #2/#3
- precision@5=0.7222 (exactly identical to the prior run — real internal-consistency signal,
  retrieval-only and unaffected by anything changed this session) | recall@5=1.0000
- **llm_judge=4.0878/5 (81.8%)** — ▲ +0.10 vs 2026-08-01's 3.9863, consistent with the
  comparison_agent fix; still short of the 4.5 target because the MMR-concentration half of that
  gap (#3) is untouched
- **Latency (n=75 traces, real end-to-end HTTP wall time)**: p50=5.99s (▲ faster than 7.42s),
  mean=7.06s, p90=9.27s, p95=9.41s, p99=21.44s (▼ much worse tail vs 10.88s), min=4.05s,
  max=22.04s. The two slow outliers (`AAPL_FY2025-Q3_num_002`=21.22s, `_num_004`=22.04s) are both
  `numeric` claims, same company/quarter, early in the run — the signature of Azure SQL
  Serverless resuming from auto-pause (documented elsewhere as up to ~49s cold), not a code
  regression; every other numeric claim later in the same run was back to 5-7s. Plausible, not
  independently proven via SQL-side logs.
- Error rate 1/75 (1.3%, ▼ improved from 2/75) — but **not the same claim**: the two original
  false positives are fixed, and this run surfaced a **third, different** off-topic false
  positive the guardrail fix didn't cover — `"iPhone, which grew a strong 13% year-over-year"`
  matches none of `_FINANCIAL_DOMAIN_TERMS` ("grew" ≠ "growth", spelled-out "year-over-year" ≠
  "yoy"). Not yet fixed — a real, still-open residual gap in the same heuristic.

**Independent cross-validation via Langfuse** (same n=75 run, window 21:18–21:31 IST
2026-08-02 — confirmed via Langfuse's own "Observations by time" graph, which ramps up and drops
off at exactly that window):
- **Trace latency (backend-only, no network/polling)**: p50=4.74s, p90=6.93s, p95=7.62s,
  p99=20.46s — lower than the HTTP numbers at every percentile, same expected pattern as the
  2026-08-01 cross-validation (trace latency is a strict subset of HTTP latency). Notably, **both
  independent measurement methods caught the same p99 spike** (20.46s here vs 21.44s HTTP) — real
  evidence it's a genuine backend-side event, not an artifact of HTTP polling.
- **Real cost: $0.947491 total** — input $0.725825, output $0.218435, input cached-tokens
  $0.003149. Essentially flat vs the 2026-08-01 run's $0.973803 — no cost regression.
- 382 observations tracked (not the same as trace count — each of the 74 completed traces fans
  out to multiple observations, one per embedding/generation call; ~5.2 observations/trace is
  expected for this multi-agent pipeline, not a discrepancy).

**`production-full-eval-75`** (2026-08-01 — prior baseline, pre this-session's fixes, kept for
trend comparison only). Full 75-claim golden dataset, same methodology as above.
- faithfulness=0.9353, answer_relevancy=0.9616
- context_precision=0.8219 (k=2/full-chunk window, same as `final-precision-check-75`)
- context_recall=0.8386 (all claim types) / context_recall_excl_oos=0.8918
- precision@5=0.7222, recall@5=1.0000 — **identical to the local run to 4 decimal places**, a real
  internal-consistency signal (precision@5 is retrieval-only, unaffected by the one deliberate
  config difference below)
- llm_judge=3.9863/5 (79.7%)
- **Latency (n=75 traces, real end-to-end HTTP wall time)**: p50=7.42s, p90=9.19s, p95=9.33s,
  p99=10.88s, mean=7.15s, min=4.05s, max=11.01s. Error rate 2/75 (2.7%), both input-guardrail
  rejections (off-topic phrasing on a legitimate metric claim — see Known Issues), not infra
  failures — 73/75 completed cleanly.

**`final-precision-check-75`** (2026-08-01, local — methodology/config reference, not the headline
number. Full 75-claim golden dataset, `REPORT_SKIP_VERIFY=0` (verify forced on, not the default),
`CONTEXT_PRECISION_K=2`/`CONTEXT_PRECISION_CHUNK_CHARS=0`). Kept because it's what explains the
small faithfulness/llm_judge gap above — `REPORT_SKIP_VERIFY=0` was validated this session to give
a real (if modest) faithfulness/llm_judge edge over the default skip-when-safe path; production
intentionally measures the default, so a small gap here is expected, not a regression. Full
methodology in `evaluation/FINAL_REPORT.md`:
- faithfulness=0.9590, answer_relevancy=0.9421
- context_precision=0.8267 (k=2/full-chunk window — ~0.51-0.56 at the wider default k=5/300-char
  window; both real, disclose whichever is quoted — see FINAL_REPORT.md)
- context_recall=0.8134 (all claim types) / context_recall_excl_oos=0.8714 (fair comparison —
  out_of_scope ground truth is a policy statement, not a filing fact — see Session Fixes below)
- precision@5=0.7222, recall@5=1.0000
- llm_judge=4.0400/5 (80.8%) — not yet at target, see Known Issues

**Historical baselines (pre-2026-08-01, trend context only — different corpus/code states, not
directly comparable):**

**`baseline-recall-fix-25`**: faithfulness=0.9260, answer_relevancy=0.7344, context_precision=0.2640,
context_recall=0.7673, precision@5=0.7333, recall@5=1.0000, llm_judge=2.9720

**`baseline-evidence-consistency-25`** (pre-structure-aware): faithfulness=0.9274,
answer_relevancy=0.8264, context_precision=0.2960, precision@5=0.6500, recall@5=1.0000, llm_judge=3.0560

**`baseline-structure-aware-25`** (Phase 4): faithfulness=0.9139, answer_relevancy=0.6228,
context_precision=0.2560, precision@5=0.8167, recall@5=1.0000, llm_judge=3.0240

### Session Fixes Applied (do not revert)

1. **Fix 5 — chunk_index/chunk_total plumbing.** Added to `RetrievalResult` (`graph/state.py`),
   mapped in `search_documents.py` normalization, passed through `retrieval_agent._to_retrieval_results`.
   Un-blinds `adjacent_chunk_rate` (was permanently 0.0) and lets genuine identical-chunk duplicates
   be told apart from same-section-but-different-chunk pairs. No retrieval behavior change.
2. **Fix 3 — RAGAS context_precision measurement correction.** `run_ragas_eval()` can now return
   per-sample scores (`return_per_sample=True`, backward-compatible). `run_baseline_eval.py` tags
   each sample with `claim_type` and logs `ragas_context_precision_<type>` +
   `ragas_context_precision_retrieval_subset` (retrieval/comparison/out_of_scope only) to MLflow.
   Pure measurement change — zero retrieval impact.
3. **Recall fix — comparison claim ground truth anchors.** `_extract_ground_truth_anchors()` in
   `run_baseline_eval.py` now only includes the anchor matching the claim's own `fiscal_label` for
   comparison claims. `retrieval_results` never contains prior-quarter chunks (comparison_agent
   fetches those separately and never merges them back), so including `prior_anchor` structurally
   capped recall@5 at 0.5 for every comparison claim. Filters by `fiscal_label` match, not by
   hardcoded anchor key name, so it survives future claim-file reordering.
4. **`faithfulness_contexts`** (`ragas_eval.py`, `run_baseline_eval.py`, 2026-08-01) — a typed
   quote-only answer (sentiment label, comparison verdict) is graded against only the evidence it
   was actually built from, not the full 5-chunk retrieval pool. Reproduced directly: the identical
   answer/context pair scored 0.0/0.67/1.0 across three back-to-back judge calls when mixed with
   unrelated chunks; stable 1.0 four times against its true source alone.
5. **Faithfulness judge robustness to refusal/absence content** (`ragas_eval.py`, 2026-08-01) —
   prompt excludes absence/refusal boilerplate from claim extraction; an empty claims list scores
   1.0 (nothing false asserted) instead of 0.0; `report_agent.py`'s two exact templated strings
   ("No data available.", "No verified data available.") plus bare markdown headers are stripped
   deterministically before scoring (the judge was extracting these exact lines as unsupported
   claims 4/5 times even with the prompt instruction). Free-form (non-templated) refusal prose
   remains judge-scored, imperfectly — not solved, see Known Issues.
6. **`context_recall_excl_oos`** (`run_baseline_eval.py`, 2026-08-01) — new metric excluding
   `out_of_scope` claims, whose `ground_truth` is a policy statement ("Expected behavior: refuse —
   QuarterLens is not an investment advisor..."), not a filing/transcript fact. No retrieved chunk
   can ever "cover" it — including these 10 claims structurally caps raw `context_recall`
   regardless of retrieval quality (measured: 0.44 for this category vs 0.86-0.93 everywhere else).
7. **Sentiment passage selection now topic-matches first** (`run_baseline_eval.py`, 2026-08-01) —
   mirrors `_select_comparison_finding`'s existing pattern instead of raw FinBERT-confidence argmax,
   which could (and did, `AAPL_FY2025-Q3_sent_002`) select a passage about a different topic than
   what was asked, purely because it scored more confidently.
8. **`CANDIDATE_K` wired to an env var** (`retrieval_agent.py`, 2026-08-01, was hardcoded 12) —
   enables raw-candidate-pool ablation; default behavior unchanged.
9. **Deploy quality gate + test suite** (2026-08-02) — see "MLOps / Deploy Gate" below. No
   retrieval/generation behavior change; CI/testing infrastructure only.

### Key Diagnostic Findings

- **`context_precision_retrieval_subset`/`context_recall_excl_oos` are the metrics to track, not the
  raw overall numbers.** Both are diluted by claim types whose ground_truth has no chunk-level
  relevance signal a retrieved filing/transcript chunk could ever satisfy — numeric/sentiment's
  terse categorical ground_truth for the former, out_of_scope's policy-statement ground_truth for
  the latter. RAGAS scores those near 0 regardless of retrieval quality.
- **This repo's `context_precision` (`evaluation/ragas_eval.py`) is NOT the RAGAS-paper rank-weighted
  Average Precision.** It's order-insensitive `relevant / len(top-k)`. `CONTEXT_PRECISION_K`/
  `CONTEXT_PRECISION_CHUNK_CHARS` are measurement-scope parameters, not retrieval changes — see
  `evaluation/FINAL_REPORT.md` for the two legitimate values this produces (0.83 at k=2/full-chunk,
  0.51-0.56 at the wider k=5/300-char default) and which to disclose when.
- **precision@5/context_precision's remaining gap (0.72/0.83) is a confirmed MMR
  concentration-vs-diversity architectural tradeoff, not topical chunk impurity.** Full root-cause
  trace (2026-08-01, `evaluation/FINAL_REPORT.md`): the correct chunk reliably survives raw search,
  cross-source dedup, and MMR, but only wins 1 of the final 5 slots — MMR's diversity objective
  (needed for multi-source comparison/sentiment claims) spends the other 4 on other sections that
  share surface vocabulary. Three pool-widening ablations (`MMR_TOP_K` 10→15, `CANDIDATE_K`
  12→20→50) left results byte-identical, ruling out pool size as the lever. This supersedes the
  prior "topical impurity in MDA chunks" hypothesis below the line — that fix (Fix 6) was attempted
  and failed (see Rolled-Back Experiments).

### Metric Targets — LOCKED, see `evaluation/FINAL_REPORT.md`
The reported metric set is exactly 7 (+ L1/L2/L3 cache hit rates). Headline values are from
`production-full-eval-75-fixes` (2026-08-02, real production traffic, post this-session's
fixes); sources, the cold-vs-warm cache explanation, and the local reference run live in
`evaluation/FINAL_REPORT.md`.

- faithfulness 0.9646 ✅ | answer_relevancy 0.9361 ✅ | recall@5 1.0000 ✅
- context_precision 0.8176 ✅ (k=2/full-chunk window) | context_recall 0.8218 (0.8682 excl_oos)
- llm_judge 4.09/5 (81.8%) — not cleared (target 4.5), improved from 3.99 this session, see
  Known Issue #4 | precision@5 0.7222 ✅ (target 0.60)

`numeric_pass_rate` is no longer computed — it is not one of the 7 reported metrics.
Its implementation was removed with `evaluate_finetuned_vs_baseline.py`; recover from
git history if ever needed.

### Running Evaluations
```bash
# Always flush Redis first
python -c "from azure_clients.redis_client import clear_all_caches; clear_all_caches(); print('done')"

# Current locked baseline config (verify always runs — see report_agent.py note below)
REPORT_SKIP_VERIFY=0 python evaluation/run_baseline_eval.py --max-claims 10 --run-name <name>
REPORT_SKIP_VERIFY=0 python evaluation/run_baseline_eval.py --max-claims 25 --run-name <name>
```
Note: `--detail-report` was removed along with the per-claim chunk-dump/error-analysis
code path. Per-claim RAGAS scores and judge reasoning are still logged to MLflow as a
`per_claim` JSON artifact.

**`REPORT_SKIP_VERIFY`** — `report_agent.py`'s verify-skip latency optimization defaults to `1`
(skip verify when provably safe) but was never validated against a faithfulness baseline before
merging. Validated 2026-08-01 via a clean n=25/n=75 A/B: `REPORT_SKIP_VERIFY=0` (verify always
runs) gives a consistent `llm_judge` win (+0.24 to +0.43 vs three `skip=1` baselines), a smaller
faithfulness win, and a wash on context_recall — at the cost of report_agent latency rising ~0.5-1s
per claim typically. The `final-precision-check-75` locked baseline uses `=0`. Always pass it
explicitly for baseline-comparable eval runs; omit only when deliberately measuring the fast path.

### Retrieval determinism (important when verifying refactors)
Retrieval is **not** perfectly reproducible across time: AI Search hybrid BM25+vector
RRF scoring drifts, so the same query can return a different rank-5 chunk days apart.
Verified 2026-07-29 by running identical pre-refactor code hours apart and getting a
different result for one of ten claims. When checking whether a change altered
retrieval, always A/B the old and new code **in the same session** — a fingerprint
captured earlier is not a valid baseline.

### Experiment Discipline
- One variable change per experiment
- Run 10 claims first → check → 25 claims → confirm
- Never run 50/75 without reviewing 25-claim results
- Log all experiments in MLflow with descriptive run names

---

## MLOps / Deploy Gate (2026-08-02)

Before this session, nothing in the deploy pipeline checked quality metrics at all —
`eval_gate.yml` ran only `smoke_test.py` (one claim, completion/non-empty/time-budget check,
explicitly by its own docstring "not the real eval suite"). A faithfulness or precision
regression would have deployed with zero automated detection. Two things now close that gap:

**1. Eval-metric threshold gate** — `evaluation/eval_gate_check.py`, wired into
`.github/workflows/eval_gate.yml` as a second job (`quality-gate`) that runs after the existing
`smoke-test` job. Runs 10 stratified claims (`GATE_MAX_CLAIMS` env var to override) through the
real pipeline via `run_baseline_eval.run_eval()` — the same function `run_baseline_eval.py`
itself uses — locked to `CONTEXT_PRECISION_K=2`/`CONTEXT_PRECISION_CHUNK_CHARS=0` to match the
headline `production-full-eval-75-fixes` measurement window. Compares against **regression-guard
floors**, not the aspirational targets in "Metric Targets — LOCKED" above — `llm_judge` and
`context_recall` aren't at target yet and this gate isn't meant to block every deploy until they
are; it exists to catch a broken prompt or a retrieval regression, not to freeze development.
Floors (see the script's own docstring for the full reasoning): faithfulness≥0.80,
answer_relevancy≥0.80, context_precision≥0.65, context_recall_excl_oos≥0.65, precision@5≥0.55,
recall@5≥0.90, llm_judge≥3.30. `deploy.yml`'s `build-and-push` job has `needs: eval-gate` — since
a called reusable workflow's overall status is success only if every job inside it succeeds,
this second job failing blocks the image build/push exactly like `smoke-test` failing already
did. **Validated live** against current `main` (2026-08-02, n=10): faithfulness=0.9778,
answer_relevancy=0.9830, context_precision=0.8500, context_recall_excl_oos=0.8593,
precision@5=0.7200, recall@5=1.0000, llm_judge=4.3400 — all comfortably above floor, gate
printed PASS.

**2. Real test coverage** — `tests/unit/test_agents.py`, `tests/unit/test_numeric_validation.py`,
`tests/unit/test_tools.py`, and `tests/integration/test_full_pipeline.py` were all previously
0-byte empty files (only `test_guardrails.py` had real tests). Now 95 tests total, all offline —
no live Azure calls, no real model loads. `tests/conftest.py` is why: every `azure_clients/*.py`
module builds its client as a module-level singleton constructed at import time (`kv =
KeyVaultClient()`, `openai_client = OpenAIClient()`, etc.), so importing any agent/tool module in
CI (which never runs `azure/login` for the plain `lint-and-test` job) would otherwise raise
`ValueError` before a single test body runs. `conftest.py` sets dummy env-var secrets so those
constructors succeed, and specifically patches `azure.cosmos.CosmosClient` — unlike every other
client, `CosmosDecisionLogClient.__init__` makes a **real** network call
(`create_database_if_not_exists`/`create_container_if_not_exists`) at construction time, pulled
in transitively by `agents/supervisor.py`. Test design split: unit tests mock at the tool/LLM
boundary (`openai_client.chat`, `calculate_metric`, `run_finbert`, `fetch_prior_quarter`) and
exercise each agent's real internal logic; the integration test replaces whole agent functions
with fakes matching their return-shape contract and asserts on the **graph wiring** itself (the
retrieval → fan-out → fan-in structure, the `decision_log_entries` `operator.add` reducer
accumulating across parallel branches, the `error_exit` routing path) — a different concern from
what the unit tests already cover, not a duplicate of them. Several tests pin down specific bugs
already documented elsewhere in this file (the skip-not-break chunk-budget fix in three
different agents, the delta-vs-level category-error guard, etc.) as real regression tests, not
just generic coverage.

---

## Production Latency (2026-08-01 investigation)

Prior latency work (see README's Engineering Notes) took the full pipeline from a first
measured 123s to ~18-20s warm. This session's investigation (average query at the time,
~10s) found and fixed two more real, verified bugs, ruled out two plausible-looking
heuristics with real data, and confirmed the remaining cost is architecturally external.

### Fixes applied (do not revert)

1. **L2 retrieval cache now stores the `embedding` field** (`tools/search_documents.py`).
   Previously stripped before writing to Redis on the theory that L1's embedding cache
   made this "no gain" — measured false: `embed_batch()`'s own docstring already recorded
   1.3-1.9s per call for ~24 chunks, paid on **every** retrieval including L2 cache hits,
   because a cache hit returning chunks without their embedding forced `mmr_rerank()`'s
   fallback re-embed path regardless of L1 state. Confirmed live: a cache-hit retrieval
   spent 2.34s in MMR alone before the fix, 0.01s after. Cost: cache entries grow to
   ~500-700KB (24 chunks × 1536 floats, JSON-serialized) against a 30min TTL on Basic C0
   (250MB) — comfortable margin.
2. **Azure SQL connection is now pooled** (`azure_clients/sql_client.py`). `calculate_metric()`
   (numeric_validation_agent's SQL read) opened a brand-new connection on every single call
   — paying a full connect handshake each time, on top of whatever the Serverless tier's
   compute-scaling state added underneath (the same tier `warm_up()`'s own docstring
   already measured at ~49s cold / ~0.95s warm). Fixed with a held-open, lock-protected,
   health-checked (`SELECT 1` probe, rebuilt via the existing `_connect()` retry/backoff
   path if dead) connection reused across calls, with an explicit commit-or-rollback on
   every checkout so no implicit transaction (pyodbc defaults `autocommit=False`) is left
   open between reuses. Verified in isolation: repeat `sql_client.count()` calls dropped
   from 3.88s (cold) to a stable ~0.75s. Live: `numeric_validation_agent` stabilized at
   ~0.7-0.8s from the second call onward (first call after any process restart still pays
   the one-time connect cost — expected, not a regression).
3. **Frontend completion detection is now push- not poll-driven**
   (`frontend/src/pages/NewAnalysis.jsx`). The UI only noticed a finished run via a 2s
   poll interval (`setInterval(poll, 2000)`), so it could sit for up to ~2s (average ~1s)
   after `report_agent`/`supervisor_finalize` actually finished before the report page
   ever loaded — a real, measured gap, not perceived lag. The SSE stream already emits a
   `"done"` event the instant the pipeline finishes (success or failure — it carries no
   outcome itself, `/status` is still the source of truth), but the handler only closed
   the stream. Now it also triggers an immediate status check. Verified working live.
4. **Production SQL connections were completely broken** (`Dockerfile`) — discovered
   *after* deploying fix #2 above, by checking production logs rather than trusting the
   green CI/deploy checkmark. `msodbcsql18` installs with `--no-install-recommends`, which
   silently drops `libgssapi-krb5-2` (Kerberos/GSSAPI) — Microsoft lists it as a
   Recommends, not a hard Depends, even though `libmsodbcsql-18.6.so.2.1` is unconditionally
   link-time-dependent on `libgssapi_krb5.so.2` regardless of whether a given connection
   actually uses Kerberos (this app only ever uses SQL-auth via UID/PWD). Every
   `calculate_metric()` call was silently failing and returning no data in production.
   Root-caused via `az containerapp exec` + `ldd` directly against the running container
   (the failing `.so` genuinely existed at its path — `ldd` was what revealed the *real*
   missing dependency instead of the misleading "file not found" pyodbc surfaced). Fixed
   by installing `libgssapi-krb5-2` explicitly; confirmed via a direct in-container
   `sql_client.count()` call and a live production request exercising
   `numeric_validation_agent` after redeploying.

### Production latency — real HTTP traffic, not in-process measurement

`evaluation/run_baseline_eval.py` calls `compiled_graph.ainvoke()` **in-process** — running it
measures "local machine → Azure," not production. A separate ad-hoc script (pattern:
`production_full_eval.py`, this session) instead makes real HTTP calls against the deployed
Container App (`POST /api/analysis/run` with `no_cache=true` → poll `/status` → `GET
/api/reports/{run_id}` for the full response) so the numbers reflect what a real user's browser
actually experiences, not this project's own dev-machine network path.

**`production-full-eval-75` latency** (n=75 traces, 73 completed): p50=7.42s, p90=9.19s,
p95=9.33s, p99=10.88s, mean=7.15s, min=4.05s, max=11.01s. Error rate 2.7% (2/75), both
input-guardrail rejections on a legitimate metric claim phrased without financial keywords
(see Known Issues) — not infra failures; 73/75 completed cleanly.

**Independent cross-validation via Langfuse** (same n=75 run, `us.cloud.langfuse.com`, project
`quarterlens-ai`) — a second, independently-instrumented measurement (OTEL trace spans emitted by
the pipeline itself, not measured by polling from outside) confirms the HTTP-based numbers above
rather than contradicting them:
- **Trace count: exactly 75** — matches the HTTP-measured request count precisely, confirming
  this project's "wrap the pipeline invocation in one OTEL span" instrumentation correctly groups
  one trace per analysis run. Phoenix, the other observability backend wired into this project,
  showed 471 "traces" for the same window — root-caused, not just noted: `phoenix_setup.py`'s
  `OpenAIInstrumentor().instrument(...)` patches the `openai` SDK process-globally, and
  `run_baseline_eval.py` used to call `setup_phoenix()`/`setup_langfuse()` as a **module-level
  import side effect**, so any script merely importing its helper functions (e.g. the standalone
  production-latency-test script that produced this exact run) silently wired up global tracing
  too. The 365 RAGAS/judge scoring calls that script made — real LLM calls, but with no parent
  pipeline span to group under — each became their own root-level trace in Phoenix's ingestion,
  inflating 75 real pipeline traces to 471. **Fixed**: observability setup moved from module
  import time into `run_eval()` itself, gated on `not dry_run` — only the code path that actually
  drives claims through the pipeline turns tracing on now. Verified: importing helpers no longer
  triggers Phoenix/Langfuse init. This does not retroactively clean up the already-ingested 471
  for that historical window — only Phoenix's own UI/API can do that, as a separate action.
- **Trace latency (pure backend pipeline execution, no network/polling)**: p50=4.96s, p90=7.21s,
  p95=8.13s, p99=9.16s — lower than the HTTP numbers at every percentile, exactly as expected: this
  measures only `compiled_graph.ainvoke()`'s own execution, a strict subset of the full HTTP
  round-trip (which also includes network time and up to ~1s of polling-detection lag from the
  1-second poll interval the measurement script uses). The two methods agreeing once you account
  for what each includes is the validation, not a contradiction to resolve.
- **Per-call latency** (real, not estimated from log timestamps): LLM completions
  (`OpenAI-generation`) p50=1.28s/p90=2.30s/p95=2.54s/p99=3.00s; embeddings p50=0.13s/p90=0.27s/
  p95=0.30s/p99=0.38s.
- **Real cost for this exact n=75 run: $0.973803 total** — input $0.734588, output $0.234054,
  input cached-tokens $0.005088. 1.1M tokens on `gpt-5.4-mini`, 3.64K tokens on
  `text-embedding-3-small`. A concrete, citable per-run cost figure, not an estimate.

**Local vs. production is not the same test, and production is faster** — confirmed via the same
5 manual queries run against both, same day: local averaged ~11.6s total, production ~4.8s, a
~2.4x gap driven almost entirely by `retrieval_agent` (local avg ~6.8s vs. production avg ~1.3s).
This is network locality, not a code difference — the Container App talks to Azure AI Search over
Azure's internal backbone; a local dev machine talks to the same F0-tier service over the public
internet. A meaningful share of what earlier local profiling attributed to "AI Search is slow" was
actually "local-machine-to-Azure distance." Do not use local retrieval timings as a production
estimate — measure production directly.

### Sub-stage retrieval profiling (local machine, real measured data — see network-locality note above)

| Step | Fresh query | Cached query |
|---|---:|---:|
| Azure AI Search (embed + hybrid search, one call) | 5.5–9.3s | 0.4–1.0s |
| MMR | 0.01s | 0.00s |
| Cross-encoder rerank | 1.05–1.2s | 1.05–1.2s (no cache possible) |
| Parent expansion | 0.2–0.4s | 0.2s |

Fresh and cached queries are different optimization problems: Azure Search dominates
fresh (F0 tier, shared/multi-tenant, no dedicated compute — external, no code fix
available short of a paid tier); the cross-encoder becomes the largest *remaining* cost
on a cache hit, since it re-scores every call regardless of cache state.

### Ideas tested and ruled out with data (do not re-attempt without new evidence)

- **Skip reranking heuristically for "simple" queries.** Tested directly: compared
  MMR-order top-5 against post-rerank top-5 across the stratified 25-claim sample.
  **0% were identical** (rerank never a no-op); **40% shared fewer than 3 of the same
  chunks**. No claim-type signal predicts stability — `numeric` (the closest proxy for
  "simple exact-lookup" queries) showed the same ~2.8/5 average overlap as
  `comparison`/`sentiment`/`retrieval`/`out_of_scope` (all 2.4-3.0/5). Reranking is doing
  substantial, non-redundant work uniformly across query types on this corpus; there is
  no safe target group for a skip heuristic. This is the same failure family as the
  diversity-cap and section-routing rollbacks below — a shortcut that looks reasonable
  and isn't supported once measured.
- **Adaptive candidate depth** (fewer candidates for "exact" queries like "NVDA FY2026 Q3
  revenue", more for broad analytical ones). Same data disproves the premise: the
  `numeric` category is not measurably more stable than broader query types, so there's
  no empirical basis for giving it a smaller pool.
- **`CANDIDATE_K`/`MMR_TOP_K` pool-widening** — see Rolled-Back Experiments table; already
  confirmed the bottleneck is final MMR/rerank selection, not pool size, so narrowing for
  pure speed remains untested but widening is a settled dead end.

### What's left is external, not something the retrieval pipeline's own logic can fix

Azure AI Search's network round-trip (F0 tier), real Azure OpenAI chat-completion time
(`report_agent`, `comparison_agent`'s extract+compare calls), and Cosmos DB not being
warmed at API boot (`api/main.py`'s lifespan warms cross-encoder/FinBERT/Redis/SQL but
not Cosmos — `supervisor_finalize` showed 1.1-1.7s on a fresh process's first request vs
0.2-0.3s after, consistent with SDK endpoint-discovery-on-first-call; a real, small,
not-yet-applied fix, same pattern as the other four warm-ups).

---

## Known Issues / Deferred Items

1. **AI Search duplicate chunks** — hybrid BM25+vector RRF returns same chunk_id twice
   in one search call. Dedup tested (baseline-dedup-k12-10) caused recall@5 regression.
   Root cause not yet proven: F0 tier artifact vs chunking vs RRF behavior.
   Next step: inspect actual duplicate chunk_ids and pairwise text similarity.

2. **Section-aware routing** — tested (baseline-section-routing-25), precision@5
   0.817→0.533. mda-only filter too restrictive for financial queries that span
   multiple sections. Needs redesigned intent→section mapping before re-enabling.

3. **precision@5 / context_precision ceiling (0.72 / 0.83)** — DIAGNOSED, see "Key Diagnostic
   Findings" under Evaluation above. Confirmed (2026-08-01) as an MMR concentration-vs-diversity
   architectural tradeoff, not a bug with an available quick fix — pool-widening ruled out with real
   data (3 ablations, byte-identical results), and every historical attempt to bias the tradeoff
   toward concentration (diversity cap, section routing, MDA rechunk, table demotion) has regressed
   something else. A real fix needs topic-aware MMR (diversify across different sub-topics,
   concentrate within one) — genuine design work, not a same-session ablation.

   **`context_recall`'s gap (0.8386, target 0.90) is very likely the same root cause, not a
   separate issue** — investigated 2026-08-02 (branch `fix10-known-issues-punch-list`, not
   merged into a code change, diagnosis only). Real per-claim data from that session's own eval
   runs showed `retrieval`-type claims specifically underperforming (0.4-0.5 vs ~1.0 for other
   claim types). Checked the two lowest-scoring claims directly against `golden_dataset/claims/`:
   both need coverage from two genuinely different filing sections in the same query —
   `AAPL_FY2025-Q3_ret_004` ("tariff impact") spans `mda` + a second section;
   `NVDA_FY2026-Q3_ret_001` ("China risks and financial exposure") needs both `risk_factors` and
   `mda`. That's the same shape as this issue: a query needing multi-section coverage only
   reliably gets one section into the final 5-chunk pool. No independently-actionable fix for
   `context_recall` distinct from the topic-aware MMR work above — whatever fixes #3 should fix
   this too. Not attempted this session given the cost/risk of #3's redesign (see above) and a
   tight eval budget; left for the same future dedicated session as #3.
4. **`llm_judge` gap (4.04/5, target 4.5/5)** — worst categories: `comparison` (~3.4),
   `sentiment` (~3.6). Split root cause: some comparison failures are the same MMR-concentration
   issue as #3 (confirmed on `GOOGL_FY2025-Q3_cmp_003` — the correct target sentence was never in
   `comparison_agent`'s retrieved context at all, still open, needs the topic-aware MMR work in #3);
   others were `_COMPARE_SYSTEM` prompt-rule ambiguity — **the prompt-rule half fixed 2026-08-02**
   (branch `fix10-known-issues-punch-list`). Root-caused properly before touching the prompt: 5
   scenarios covering the originally-suspected "value flip vs. characterization change" ambiguity
   (sign flips, same-number-different-tone, verbatim, new-topic) all scored perfectly and
   consistently against the *unmodified* prompt — that hypothesis didn't reproduce. The real,
   reproduced instability was narrower: a *small, same-driver* magnitude difference (e.g. "up 15%"
   vs "up 17%") flip-flopped 3-true/1-false across 4 repeated isolated calls, while a large,
   clearly-meaningful magnitude change (32%→15%) and a pure paraphrase (same number, different
   words) were both 100% stable and correct. Fixed by adding an explicit materiality-threshold rule
   (`_COMPARE_SYSTEM` rule 6, `agents/comparison_agent.py`) — re-verified the previously-unstable
   case now returns `false` consistently (5/5), no regression on 4 other cases. Zero added LLM
   calls (still the same 2-call extract+compare flow) — no production latency impact. The
   MMR-concentration half (#3) is unrelated and still open. **Confirmed at full production scale**
   (`production-full-eval-75-fixes`, 2026-08-02): aggregate `llm_judge` improved 3.9863→4.0878
   (+0.10) — a real, measured gain, not just an isolated-test artifact, though still short of the
   4.5 target since #3 remains untouched.
5. **~~Faithfulness judge unreliable on free-form (non-templated) refusal prose~~ — FIXED
   2026-08-02** (branch `fix10-known-issues-punch-list`). The deterministic boilerplate-stripping
   fix (Session Fixes #5) only ever covered `report_agent.py`'s two exact templated strings —
   free-form refusal sentences (e.g. "The evidence does not establish X") were still handled only
   by a single abstract prompt rule + enumerated example phrases, and regex-stripping this class is
   unsafe (these sentences can carry real embedded facts worth checking). Reproduced fresh, cheaply
   (isolated `_score_faithfulness` calls, not a pipeline run — zero production latency impact,
   `ragas_eval.py` never runs during a live request): 2 of 4 realistic free-form phrasings either
   flip-flopped (3x 1.0/2x 0.0 across 5 repeats) or were consistently wrong (5x 0.0 when correct is
   1.0). Fixed by replacing the phrase-matching approach with a general test the judge applies to
   every sentence regardless of wording ("is this describing what the DOCUMENTS say, or what
   happened at the company") plus one worked example — see `_score_faithfulness`'s own docstring
   in `evaluation/ragas_eval.py`. Re-verified: same 4 phrasings now score a consistent, correct 1.0
   across 5 repeats each, no regression on a plain-supported-claim control. One narrower residual
   case remains, not fixed: a sentence that both hedges one claim and states a real one in the same
   breath ("X wasn't established, though Y was 47.1%") now consistently scores 0.5 rather than
   1.0 — the real claim is correctly checked, but the explicitly-negated claim is still extracted
   and dinged even though the answer never asserted it. Narrower and rarer than the original bug.

6. **Alias map split** — `calculate_metric.py` `_CONCEPT_ALIASES` should split into
   `FINANCIAL_METRIC_ALIASES` + `SEGMENT_METRIC_ALIASES`. Deferred (numeric_pass=1.0).

7. **Metric extraction normalization** — `numeric_validation_agent.py` extracts compound
   strings. Should extract `metric=revenue_growth_cc` + `company_segment=Azure` separately.

8. **`run_baseline_eval.py` refactor** — 700+ lines. Split into scoring.py,
   error_analysis.py, report.py. Deferred until baseline locked.

9. **ARCHITECTURE.md update** — gpt-5-mini model change pending (documentation only).

10. **~~Input guardrails false-positive on some legitimate metric claims~~ — PARTIALLY FIXED
    2026-08-02, one new residual case found.** `api/guardrails.py`'s `_FINANCIAL_DOMAIN_TERMS`
    broadened (same pattern as the earlier "leadership"/"investment" fix) to cover engagement/
    usage-metric vocabulary: "user(s)", "app(s)", "daily active", "monthly active", "dau", "mau",
    "dap", "active people", "install", "downloads". Confirmed fixed both via unit tests and live
    against production (`tests/unit/test_guardrails.py`; manually verified through the deployed
    UI): the exact DAP-metric quote that failed before now passes, the off-topic check still
    correctly rejects a genuinely unrelated query. **Re-measured at full scale**
    (`production-full-eval-75-fixes`, 2026-08-02): error rate improved 2/75→1/75, confirming the
    two original false positives are gone — but that run surfaced a **third, different** false
    positive the fix doesn't cover: `"iPhone, which grew a strong 13% year-over-year"` matches no
    `_FINANCIAL_DOMAIN_TERMS` entry ("grew" ≠ "growth", spelled-out "year-over-year" ≠ "yoy").
    Same underlying heuristic gap (keyword-list matching doesn't generalize to paraphrases), not
    a new bug class. Not yet fixed — deferred, same reasoning as the rest of this list.

**Not attempted this session, deliberately: items #1, #3, #4, #5 above.** Each is explicitly
documented as needing real design work or careful multi-case validation, not a same-session
ablation — e.g. #4's own text says "further changes need careful validation against currently-
passing cases, not a quick tweak." Attempting several of these together right before a commit +
deploy would be exactly the kind of compound, unmeasured change this project's own "Single-
variable ablations only" rule (Experiment Discipline, above) exists to prevent — and is now also
what the new eval-metric gate (MLOps / Deploy Gate, above) is positioned to catch if violated
anyway. Left for a dedicated session per claim/experiment, same as before.

---

## Rolled-Back Experiments (do not re-implement without new evidence)

| Experiment | Result | Why Reverted |
|---|---|---|
| bge-reranker-base | answer_relevancy -0.077 at n=25 | Worse overall despite larger model |
| Diversity cap (max 2/section) | precision@5 0.76→0.44 | Comparison claims need multiple same-section chunks |
| Section routing (mda filter) | precision@5 0.817→0.533 | mda-only too restrictive |
| Chunk_id dedup | recall@5 1.0→0.9 | Removed needed evidence |
| Structured ComparisonFinding | answer_relevancy -0.050 at n=25 | More mechanical, less grounded |
| Draft grounding prompt | answer_relevancy -0.133 at n=25 | Too conservative |
| MDA chunk topical-purity rechunk (Fix 6, 2026-08-01) | context_precision moved wrong direction (0.50→0.46), precision@5 0.72→0.64 at n=10 | Full re-chunk/re-embed/re-index; mixed failure pattern (comparison-claim MMR dilution + transcript pollution), not the single predicted mechanism. Corpus + code rolled back and reconfirmed at 3,526/3,526 chunks. |
| Table-chunk soft demotion (2026-08-01) | precision@5 0.6833→0.6667, targeted case unchanged | Additive ranking-time penalty can't promote a chunk that was never in the raw candidate pool to begin with — root cause was upstream (MMR concentration-vs-diversity), not final ranking |
| `CANDIDATE_K`/`MMR_TOP_K` pool widening (2026-08-01) | precision@5 byte-identical at 15/20/50 vs default | Confirms the bottleneck is final MMR/rerank selection, not candidate pool size — see Known Issues #3 |

---

## Deviation Log (from locked spec)
- **#29** — `transcript_retrieval_results` added to GraphState
- **#30** — Structure-aware chunking (sentence-boundary, zero overlap) replaces recursive
- **#31** — `subsection` field added to AI Search index schema (filterable, not yet used)
- **#32** — Code optimization pass (branch `fix9-code-optimization`): ~2,000 lines removed.
  Deleted dead modules (`tool_registry`, `decision_log`, `evaluate_finetuned_vs_baseline`,
  `finetuning/*`) and the retired `gpt-4o-mini` "finetuned" tier — which also removed
  `report_model_tier` from `GraphState`. Added `agents/_common.py` (decision-log helpers)
  and `data_pipeline/manifest_io.py` (manifest read/project/write). No agent behavior change.

---

## Optimization pass — what was deliberately NOT consolidated
These look like duplication and are not. Merging any of them moves a locked metric:
- **`_topic_overlap` ×3** (`retrieval_agent`, `sentiment_agent`, `run_baseline_eval`) —
  **inverted semantics**: retrieval returns `1.0` on an empty query, the other two `0.0`;
  stopword filtering differs per copy. The divergence is intentional and documented in-file.
- **RAGAS `{}`-on-failure vs judge `error`-on-failure** — a RAGAS failure scores 0.0 *into*
  the mean; a judge failure is *excluded* from it. Same-looking try/except, opposite effect.
- **`ragas_eval` `[:8]` vs `llm_as_judge` `[:5]` context caps** — one shared constant would
  move `llm_judge_mean`.
- **`contexts` vs `gen_contexts`**, **`answer` vs `faithfulness_answer`** — each pair feeds
  different metrics on purpose.
- **5 retry loops** (sql/embedding/financials/indexer/transcript) — different counts, backoff
  curves, and exhaustion behavior.
- **2 sentence splitters** — both live, in the same call chain; only one guards abbreviations.
- **`math.isnan` filter in `ragas_eval.py`** — provably dead, kept anyway: one line in a
  metric-critical file isn't worth the risk.

**Retracted from this list:** the `break`-on-budget-overflow in
`numeric_validation_agent._concat_transcript` and `comparison_agent._ranked_context` was
initially left alone as "bug-shaped but metric-locked". That was wrong — it was measurably
returning `""` for 7/10 and 5/10 claims respectively (an oversized first chunk discarded
everything behind it), so numeric extraction and comparison both ran on empty input. Fixed
to `continue`. Lesson: "looks intentional" is not evidence; measure before classifying
something as load-bearing.

---

## Folder Structure (key paths)
```
agents/           — LangGraph agent nodes
azure_clients/    — Azure SDK wrappers (NEVER rename to azure/)
data_pipeline/    — chunking, embedding, indexer
data/
  parsed/         — section-split parsed filings (JSON)
  raw/            — downloaded SEC filings + transcripts
  chunks/         — structure-aware chunked output (JSON)
  embeddings/     — embedded chunks (JSON)
evaluation/       — run_baseline_eval.py, golden_dataset/, ragas_eval.py
golden_dataset/
  claims/         — 75 hand-verified claim JSONs
graph/            — state.py, build_graph.py
observability/    — MLflow, Langfuse, Phoenix setup
tools/            — search_documents, rerank_documents, calculate_metric, run_finbert
```