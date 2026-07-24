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
- `evaluation/detail_report_baseline-structure-aware-25b.json` — retrieval error analysis
  showing which specific claims fail, duplicate pairs, and classification breakdown

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

**`baseline-recall-fix-25`** (current — Fix 5 + Fix 3 + recall fix applied, this session):
- faithfulness=0.9260, answer_relevancy=0.7344, context_precision=0.2640 (all claim types — diluted by numeric/sentiment)
- context_precision_retrieval_subset=0.2833 (retrieval+comparison+out_of_scope only — fair comparison to precision@5)
- context_recall=0.7673
- precision@5=0.7333, recall@5=1.0000 (restored — prior-quarter anchors excluded from comparison claims)
- llm_judge=2.9720, numeric_pass=1.0000
- exact_match_rate=0.7333, duplicate_density=0.5667
- adjacent_chunk_rate=0.2417 (now meaningful — chunk_index plumbed through Fix 5)

**`baseline-evidence-consistency-25`** (pre-structure-aware, production reference):
- faithfulness=0.9274, answer_relevancy=0.8264, context_precision=0.2960
- precision@5=0.6500, recall@5=1.0000, llm_judge=3.0560, numeric_pass=1.0000

**`baseline-structure-aware-25`** (Phase 4, prior best precision@5):
- faithfulness=0.9139, answer_relevancy=0.6228, context_precision=0.2560
- precision@5=0.8167, recall@5=1.0000, llm_judge=3.0240, numeric_pass=1.0000

**Retrieval error analysis (baseline-structure-aware-25b, historical):**
- exact_match_rate=0.817, same_company_rate=0.100, irrelevant_rate=0.000
- duplicate_density=0.633, adjacent_chunk_rate=0.000 (blind — chunk_index not yet plumbed)
- dominant_failure=same_company (right company/quarter, wrong section)

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

### Key Diagnostic Findings (Opus analysis, this session)

- **`context_precision_retrieval_subset` is the metric to track, not overall `context_precision`.**
  The overall number is diluted by numeric/sentiment claims whose terse categorical ground_truth
  (`"Filed value: 82886 USD millions..."`, `"Expected sentiment: negative..."`) has no chunk-level
  relevance signal — RAGAS scores those contexts near 0 regardless of retrieval quality.
- **This repo's `context_precision` (`evaluation/ragas_eval.py`) is NOT the RAGAS-paper rank-weighted
  Average Precision.** It's order-insensitive `relevant / len(top-5 chunks)`, with each chunk
  truncated to 300 chars before the LLM judges it. Re-ordering chunks within top-5 does not move
  this metric — only reducing the count of off-topic chunks in the top-5 does.
- **Root cause of low context_precision: topical impurity in MDA chunks**, not chunk splitting.
  MSFT's `mda` section alone chunks into 33 pieces from ~6 total filing sections; a query about one
  metric (e.g. Azure growth) retrieves chunks packed with 8-10 unrelated metrics. High recall, low
  precision — chunk *selection*, not ordering, is the lever.
- **MMR lambda (re-ordering) is a weaker lever than chunk purity** given the metric is order-insensitive.
  Still worth ablating cheaply before the expensive re-chunk/re-embed path.
- **Next experiment:** `MMR_LAMBDA` ablation (0.7, then 1.0) against `baseline-recall-fix-25`.
- **After that:** chunk topical purity (Fix 6, deferred — requires full re-embed + re-index; highest
  ceiling, highest recall risk — do not attempt before the MMR ablation is measured).

### Metric Targets
- context_precision_retrieval_subset: 0.5+ (currently 0.2833) ← PRIMARY TARGET
- precision@5: 0.8+ (currently 0.7333)
- llm_judge: 4.0+ (currently 2.9720)
- numeric_pass_rate: 1.0 (locked — do not regress)
- recall@5: 1.0 (locked — do not regress)

### Running Evaluations
```bash
# Always flush Redis first
python -c "from azure_clients.redis_client import clear_all_caches; clear_all_caches(); print('done')"

# Phased eval (cost control: 10 → 25 → 50 → 75)
python evaluation/run_baseline_eval.py --max-claims 10 --run-name <name>
python evaluation/run_baseline_eval.py --max-claims 25 --run-name <name> --detail-report
```

### Experiment Discipline
- One variable change per experiment
- Run 10 claims first → check → 25 claims → confirm
- Never run 50/75 without reviewing 25-claim results
- Log all experiments in MLflow with descriptive run names

---

## Known Issues / Deferred Items

1. **AI Search duplicate chunks** — hybrid BM25+vector RRF returns same chunk_id twice
   in one search call. Dedup tested (baseline-dedup-k12-10) caused recall@5 regression.
   Root cause not yet proven: F0 tier artifact vs chunking vs RRF behavior.
   Next step: inspect actual duplicate chunk_ids and pairwise text similarity.

2. **Section-aware routing** — tested (baseline-section-routing-25), precision@5
   0.817→0.533. mda-only filter too restrictive for financial queries that span
   multiple sections. Needs redesigned intent→section mapping before re-enabling.

3. **context_precision gap** — DIAGNOSED, see "Key Diagnostic Findings" under Evaluation above.
   Not a rank-position problem — this repo's context_precision is order-insensitive. Root cause is
   topical impurity in MDA chunks + dilution from numeric/sentiment claims in the overall metric.
   Track `context_precision_retrieval_subset` going forward.

4. **Alias map split** — `calculate_metric.py` `_CONCEPT_ALIASES` should split into
   `FINANCIAL_METRIC_ALIASES` + `SEGMENT_METRIC_ALIASES`. Deferred (numeric_pass=1.0).

5. **Metric extraction normalization** — `numeric_validation_agent.py` extracts compound
   strings. Should extract `metric=revenue_growth_cc` + `company_segment=Azure` separately.

6. **`run_baseline_eval.py` refactor** — 700+ lines. Split into scoring.py,
   error_analysis.py, report.py. Deferred until baseline locked.

7. **ARCHITECTURE.md update** — gpt-5-mini model change pending (documentation only).

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

---

## Deviation Log (from locked spec)
- **#29** — `transcript_retrieval_results` added to GraphState
- **#30** — Structure-aware chunking (sentence-boundary, zero overlap) replaces recursive
- **#31** — `subsection` field added to AI Search index schema (filterable, not yet used)

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