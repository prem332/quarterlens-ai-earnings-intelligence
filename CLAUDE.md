# QuarterLens AI — Claude Code Instructions

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
3. **`gpt-5.4-mini` requires** `api_version="2024-12-01-preview"`, `max_completion_tokens` (not `max_tokens`), minimum 4096 tokens
4. **LangGraph is sole orchestrator** — CrewAI only in report_agent bull/bear debate
5. **No architectural changes without explicit confirmation**
6. **Single-variable ablations only** — never compound changes before measuring

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

### Locked Baseline
**`baseline-evidence-consistency-25`** (production reference):
- faithfulness=0.9274, answer_relevancy=0.8264, context_precision=0.2960
- precision@5=0.6500, recall@5=1.0000, llm_judge=3.0560

**`baseline-structure-aware-25`** (Phase 4 structure-aware chunking):
- faithfulness=0.9139, answer_relevancy=0.6228, context_precision=0.2560
- precision@5=0.8167, recall@5=1.0000, llm_judge=3.0240

### Running Evaluations
```bash
# Flush Redis before every eval run
python -c "from azure_clients.redis_client import clear_all_caches; clear_all_caches(); print('done')"

# Phased eval (cost control: 10 → 25 → 50 → 75)
python evaluation/run_baseline_eval.py --max-claims 10 --run-name <name>
python evaluation/run_baseline_eval.py --max-claims 25 --run-name <name>

# With retrieval error analysis detail report
python evaluation/run_baseline_eval.py --max-claims 25 --run-name <name> --detail-report
```

### Metric Targets (Phase 3/4)
- context_precision: 0.8+ (currently 0.25-0.30)
- precision@5: 0.8+ (currently 0.65-0.82 depending on run)
- llm_judge: 4.0+ (currently ~3.0)
- numeric_pass_rate: 1.0 (locked, do not regress)
- recall@5: 1.0 (locked, do not regress)

### MLflow Tracking
```bash
mlflow ui  # view at http://localhost:5000
```

---

## Environment Setup

```bash
# Activate venv
source .venv/Scripts/activate  # Git Bash on Windows

# Load secrets from Key Vault (via .env)
# All secrets use hyphen naming in KV: AZURE-SEARCH-ENDPOINT, etc.

# Env vars for ablation
MMR_LAMBDA=0.5          # MMR diversity/relevance balance
MMR_TOP_K=10            # candidates into cross-encoder
MAX_CHUNKS_PER_SECTION=0  # diversity cap (0=disabled)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
SECTION_ROUTING_ENABLED=0  # section-aware routing (0=disabled)
```

---

## Known Issues / Deferred Items

1. **AI Search duplicate chunks** — hybrid BM25+vector RRF can return same chunk_id twice
   in one search call on Free F0 tier. Dedup tested but caused recall@5 regression.
   Root cause not yet proven to be F0-specific vs chunking artifact.

2. **Section-aware routing** — tested (baseline-section-routing-25), caused precision@5
   0.817→0.533. mda-only filter too restrictive. Needs redesigned intent→section mapping.

3. **Alias map split** — `calculate_metric.py` `_CONCEPT_ALIASES` should be split into
   `FINANCIAL_METRIC_ALIASES` + `SEGMENT_METRIC_ALIASES`. Deferred (numeric_pass=1.0).

4. **Metric extraction normalization** — `numeric_validation_agent.py` extracts compound
   strings like `azure_other_cloud_services_revenue_growth_cc` instead of
   `metric=revenue_growth_cc` + `company_segment=Azure`. Deferred.

5. **`run_baseline_eval.py` refactor** — currently 700+ lines. Should split into:
   `scoring.py`, `error_analysis.py`, `report.py`. Deferred until baseline locked.

6. **ARCHITECTURE.md update** — gpt-5-mini model change (documentation only, pending).

---

## Deviation Log (from locked spec)
- **#28** — RecursiveCharacterTextSplitter deployed (since superseded by structure-aware)
- **#29** — `transcript_retrieval_results` added to GraphState
- **#30** — Structure-aware chunking (sentence-boundary, zero overlap) replaces recursive

---

## Folder Structure (key paths)
```
agents/           — LangGraph agent nodes
azure_clients/    — Azure SDK wrappers (NEVER rename to azure/)
data_pipeline/    — chunking, embedding, indexer
evaluation/       — run_baseline_eval.py, golden_dataset/, ragas_eval.py
golden_dataset/   — 75 hand-verified claims (claims/*.json)
graph/            — state.py, build_graph.py
observability/    — MLflow, Langfuse, Phoenix setup
tools/            — search_documents, rerank_documents, calculate_metric, run_finbert
```
