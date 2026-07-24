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
**`baseline-evidence-consistency-25`** (pre-structure-aware, production reference):
- faithfulness=0.9274, answer_relevancy=0.8264, context_precision=0.2960
- precision@5=0.6500, recall@5=1.0000, llm_judge=3.0560, numeric_pass=1.0000

**`baseline-structure-aware-25`** (Phase 4, current best precision@5):
- faithfulness=0.9139, answer_relevancy=0.6228, context_precision=0.2560
- precision@5=0.8167, recall@5=1.0000, llm_judge=3.0240, numeric_pass=1.0000

**Retrieval error analysis (baseline-structure-aware-25b):**
- exact_match_rate=0.817, same_company_rate=0.100, irrelevant_rate=0.000
- duplicate_density=0.633, adjacent_chunk_rate=0.000
- dominant_failure=same_company (right company/quarter, wrong section)

### Metric Targets
- context_precision: 0.8+ (currently 0.25-0.30) ← PRIMARY TARGET
- precision@5: 0.8+ (currently 0.65-0.82)
- llm_judge: 4.0+ (currently ~3.0)
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

3. **context_precision gap** — precision@5=0.82 but context_precision=0.25.
   Right chunks retrieved but RAGAS context_precision measures ranking quality.
   Needs investigation: are relevant chunks at wrong rank positions?

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