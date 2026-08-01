# 📊 QuarterLens AI

Production-grade Earnings Intelligence Platform powered by a Multi-Agent RAG pipeline, Azure OpenAI, Azure AI Search and LangGraph — built 100% on Microsoft Azure

## 🌐 Live Demo

> **Not currently deployed.** The Azure trial resources backing this project (AI Search, OpenAI, SQL, Redis) are run on-demand to manage cost and are shut down between working sessions — see `MLOPS.md` for the teardown/cost-control policy. Run locally via **Quick Start** below, or ask for a live walkthrough.

## 🎯 Project Overview

QuarterLens AI cross-verifies what executives say on quarterly earnings calls against what their companies actually filed with the SEC. Ask a question like *"Did Azure revenue growth accelerate this quarter, and did management's tone match the numbers?"* and it retrieves the relevant 10-Q/10-K and transcript passages, verifies every number against a structured financial-facts database, checks whether the language in the call matches the language in the filing, scores the tone with FinBERT, and drafts a cited, self-verified report.

Covers 5 companies (AAPL, MSFT, NVDA, GOOGL, META) across 5 fiscal quarters, retrieving over a 3,500-chunk hybrid-search index built from 25 filings and 25 earnings-call transcripts.

### Three Core Capabilities
- **Retrieval-grounded Q&A** → hybrid (BM25 + vector) search across filings and transcripts, reranked and cited
- **Numeric fact-checking** → every number in the draft report is verified against a SQL-backed financial-facts table before the report ships
- **Language-shift + sentiment analysis** → LLM-based comparison of current vs. prior-quarter filing language, plus deterministic FinBERT sentiment scoring of the earnings call itself

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA PIPELINE                                      │
│  SEC EDGAR filings + earnings-call transcripts                │
│  Hierarchical + semantic chunking (structure-aware, MD&A-     │
│  header-aware parent blocks → MiniLM topic-boundary children) │
│  ~3,500 chunks, text-embedding-3-small, Azure AI Search       │
├──────────────────────────────────────────────────────────────┤
│  LAYER 2: RETRIEVAL                                           │
│  Hybrid BM25 + vector search (RRF fusion)                     │
│  MMR diversity reranking (relevance vs. redundancy)            │
│  Cross-encoder relevance reranking (ms-marco-MiniLM-L-6-v2)    │
│  Small-to-big parent-block reconstruction for reasoning agents │
├──────────────────────────────────────────────────────────────┤
│  LAYER 3: AGENT ORCHESTRATION (LangGraph)                     │
│  supervisor → retrieval_agent →                                │
│    [comparison_agent ‖ sentiment_agent ‖ numeric_validation]   │
│    → report_agent                                              │
│  Three agents run as concurrent branches off one retrieval pass│
├──────────────────────────────────────────────────────────────┤
│  LAYER 4: VERIFICATION                                        │
│  Deterministic numeric validation (Azure SQL financial_facts)  │
│  FinBERT sentiment (no LLM call — deterministic)                │
│  Draft → targeted verify pass before a report reaches the user │
│  Optional CrewAI bull/bear debate, on demand                   │
├──────────────────────────────────────────────────────────────┤
│  LAYER 5: APPLICATION                                          │
│  FastAPI + Uvicorn (REST + SSE streaming, port 8000)           │
│  React + Vite frontend                                         │
│  Both containerized; API deployed to Azure Container Apps      │
├──────────────────────────────────────────────────────────────┤
│  LAYER 6: OBSERVABILITY & EVALUATION                            │
│  MLflow (experiment tracking, per-claim eval artifacts)         │
│  Langfuse + Phoenix (LLM tracing)                                │
│  RAGAS (faithfulness, relevancy, context precision/recall)      │
│  LLM-as-Judge + precision/recall@k vs. a 75-claim golden dataset│
│  Application Insights (infra monitoring)                        │
└──────────────────────────────────────────────────────────────┘
```

### Detailed Query Flow

```
Step 1: User submits a question (company, quarter, natural-language query)
        ↓
Step 2: retrieval_agent
        ├── Hybrid search: filing pass + transcript pass, run concurrently
        ├── Cross-source dedup, merge into one candidate pool
        ├── MMR rerank (relevance vs. diversity, λ=0.5)
        ├── Cross-encoder rerank → top-5
        └── Small-to-big: reconstruct parent context for reasoning agents
        ↓
Step 3: Three agents run concurrently off that one retrieval pass
        ├── comparison_agent    — LLM language-shift check vs. prior quarter
        ├── sentiment_agent     — FinBERT scoring of transcript passages
        └── numeric_validation  — every number checked against Azure SQL
        ↓
Step 4: report_agent
        ├── Drafts a cited report from all three agents' findings
        ├── Runs a targeted verify pass (skipped only when every numeric
        │   claim in the draft is provably present in the evidence)
        └── (on demand) CrewAI bull/bear debate over the same evidence
        ↓
Step 5: Streamed back to the React frontend
        ├── Live per-stage progress (SSE) — not a simulated progress bar
        ├── Token-level report drafting stream
        └── Full citations back to source filing/transcript passages
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Orchestration** | LangGraph (sole orchestrator); CrewAI only for the optional bull/bear debate |
| **LLM** | Azure OpenAI — `gpt-5.4-mini` (primary), `gpt-5-mini` (standard), routed by query complexity |
| **Embeddings** | `text-embedding-3-small` (1536 dims) |
| **Vector / Hybrid Search** | Azure AI Search (BM25 + vector, RRF fusion) |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Chunking** | Local MiniLM sentence embeddings (topic-boundary detection), `tiktoken` |
| **Sentiment** | FinBERT (deterministic, no LLM call) |
| **Structured facts** | Azure SQL Serverless (`financial_facts` table) |
| **Decision log** | Azure Cosmos DB (NoSQL) |
| **Caching** | Azure Cache for Redis — multi-level (embedding / retrieval / full-report) |
| **Document storage** | Azure Blob Storage |
| **Secrets** | Azure Key Vault (RBAC) |
| **LLM Monitoring** | Langfuse, Arize Phoenix |
| **Infra Monitoring** | Application Insights |
| **Experiment Tracking** | MLflow |
| **Backend** | FastAPI + Uvicorn, Server-Sent Events for live streaming |
| **Frontend** | React + Vite |
| **Containerization** | Docker |
| **Build & Deploy** | GitHub Actions → Azure Container Registry → Azure Container Apps |

## 📊 Evaluation Results

Evaluated against a 75-claim hand-verified golden dataset (retrieval, comparison, numeric, sentiment, and out-of-scope claim types) spanning all 5 companies and 5 fiscal quarters. Methodology, run history, and the project's single-variable-ablation discipline are tracked in `CLAUDE.md`; the latest confirmed run is in `evaluation/FINAL_REPORT.md`.

### RAGAS Evaluation (n=10, `k=2` measurement window)

| Metric | Score | Locked Target | Status |
|--------|-------|----------------|--------|
| **Faithfulness** | 0.9372 | 0.90 | ✅ PASS |
| **Answer Relevancy** | 0.9440 | 0.90 | ✅ PASS |
| **Context Precision** | 0.8500 | 0.90 | close, not cleared |
| **Context Recall** | 0.8233 | 0.90 | close, not cleared |

### Retrieval Metrics

| Metric | Score | Target |
|--------|-------|--------|
| **Precision@5** | 0.7200 | 0.90 |
| **Recall@5** | 1.0000 | 0.90 |

### LLM-as-Judge

n=10 spot checks this session landed in the 3.7–4.4/5 range depending on sample — this metric is genuinely noisy at n=10 (one LLM opinion per claim, averaged over 10) and this project's own run history shows it swinging accordingly even with zero code changes. A stable n=25 reading is the number worth trusting; see `evaluation/FINAL_REPORT.md` for the latest confirmed value against the 4.5/5 target.

> `context_precision` here is an order-insensitive relevant-chunk fraction over the top-k, judged by an LLM per chunk — not the RAGAS-paper rank-weighted Average Precision. `k` and the per-chunk text window are measurement-scope parameters, not retrieval changes; both values are reported so the number is reproducible, not cherry-picked.

## ✨ Features

### Core AI Features
- Hybrid BM25 + vector retrieval across 5 companies, 5 quarters, filings + transcripts
- MMR diversity reranking + cross-encoder relevance reranking on a globally merged candidate pool
- Small-to-big retrieval: precise child chunks for scoring, full parent context for reasoning
- Deterministic numeric fact-checking against a structured Azure SQL table — no LLM in the loop for numbers
- FinBERT sentiment scoring — deterministic, sentence-level, topically ranked against the query
- Self-verifying report generation: a targeted verify pass catches unsupported claims before they ship
- On-demand CrewAI bull/bear debate over the same retrieved evidence

### Reliability & Performance Features
- Multi-level Redis caching (query embeddings, retrieval results, full report responses)
- Real per-stage SSE progress streaming — not a simulated progress bar
- Cross-tier LLM fallback with `Retry-After` honoring on Azure OpenAI 429s
- Model-tier routing override for the report agent, measured 5.4x faster on the same prompt
- Startup warm-up for FinBERT, Redis, and Azure SQL (mitigates Serverless cold-resume latency)

### MLOps & Evaluation Features
- 75-claim hand-verified golden dataset across 5 claim types
- RAGAS + LLM-as-Judge + precision/recall@k, tracked per-run in MLflow with per-claim artifacts
- CI-gated deployment: a failing evaluation smoke test blocks the Azure Container Apps deploy, not just a failing unit test
- Documented single-variable-ablation discipline — every retrieval/generation change measured in isolation, with a rolled-back-experiments log kept in `CLAUDE.md`

## 🚀 Quick Start

### Prerequisites
- Python 3.11, Node.js 18+
- An Azure subscription with: AI Search, Azure OpenAI (chat + embedding deployments), Cosmos DB, Azure SQL, Cache for Redis, Blob Storage, Key Vault

### Local Setup

```bash
# Clone repository
git clone https://github.com/prem332/quarterlens-ai-earnings-intelligence.git
cd quarterlens-ai-earnings-intelligence

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env        # fill in Azure endpoints/keys, or set
                             # AZURE_KEY_VAULT_URL and let Key Vault resolve them

# Run backend
uvicorn api.main:app --reload --port 8000

# Run frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open browser at `http://localhost:5173` (Vite's default dev port).

`requirements.txt` is the full development set (eval suite, tests, ingestion tooling). The deployed container installs `requirements-api.txt` — a runtime-only subset that trims ~235 MB of packages the running service never imports (see that file's header for exactly what's excluded and why).

### Populating the index from scratch

```bash
# Ingestion (offline, one-time per filing/transcript)
python data_pipeline/edgar_downloader.py
python data_pipeline/transcript_fetcher.py
python data_pipeline/document_parser.py

# Chunk -> embed -> index
python -m data_pipeline.chunking --manifest data/parsed/parsed_manifest.json --out data/chunks
python data_pipeline/embedding.py --manifest data/chunks/chunk_manifest.json --out data/embeddings
python data_pipeline/indexer.py --manifest data/embeddings/embedding_manifest.json
```

`data/` is gitignored — chunk and embedding files are regenerated locally, not committed.

## ☁️ Azure Resources

| Resource | Value |
|----------|-------|
| **Resource Group** | `quarterlens-phase1-rg` |
| **Region** | East US (AI Search, OpenAI, Key Vault) / Central US (SQL) / West US 2 (Cosmos DB) |
| **AI Search** | `quarterlens-search` (Free F0) — index `quarterlens-filings` |
| **Azure OpenAI** | `quarterlens-openai` — `gpt-5-mini`, `gpt-5.4-mini`, `text-embedding-3-small` |
| **Cosmos DB** | `quarterlens-cosmos` (NoSQL, decision log) |
| **Azure SQL** | `quarterlens-sqlserver` (Serverless Free — `financial_facts`) |
| **Redis** | Azure Cache Basic C0 |
| **Blob Storage** | `quarterlensstorage` — container `raw-documents` |
| **Key Vault** | `quarterlens-kv` (RBAC) — all secrets, hyphen-named |
| **Container Registry** | `quarterlensacr` |
| **Container App** | `quarterlens-api` |

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|--------------|
| `POST` | `/api/analysis/run` | Start an analysis run (company, quarter, query) |
| `GET` | `/api/analysis/{run_id}/status` | Poll run status |
| `GET` | `/api/analysis/{run_id}/stream` | SSE stream — live stage progress + token-level report drafting |
| `GET` | `/api/analysis/{run_id}` | Full analysis result |
| `POST` | `/api/analysis/{run_id}/debate` | On-demand CrewAI bull/bear debate over the run's evidence |
| `POST` | `/api/export/{run_id}/pdf` | Export report as PDF |
| `POST` | `/api/export/{run_id}/docx` | Export report as DOCX |
| `GET` | `/api/reports` | List saved reports |
| `GET` | `/api/reports/{run_id}` | Fetch a saved report |
| `GET` | `/docs` | Swagger UI |

## 🐳 Docker & Deployment

```bash
# Build
docker build -t quarterlens-api .

# Run locally
docker run -p 8000:8000 --env-file .env quarterlens-api
```

Production deployment is CI-driven, not manual — a push to `main` runs lint + tests +
a Docker build-verification step; on success, an evaluation smoke-test gate
(`eval_gate.yml`) runs before `az containerapp update` pushes the new image to
`quarterlens-api`. See `.github/workflows/` for the full pipeline and `MLOPS.md` for
cost-control and teardown procedure.

## 📈 Engineering Notes

- **Single-variable ablation discipline** — every retrieval/generation experiment this
  project has run is measured in isolation against a locked baseline before being kept;
  `CLAUDE.md` maintains a running log of what was tried and rolled back, including *why*
  (e.g. a diversity cap that improved one metric but broke comparison-claim retrieval).
- **Latency work** — the full pipeline was profiled end-to-end and cut from a first
  measured 123s to ~18-20s warm through async I/O fixes, multi-level caching, graph
  parallelization, and model-tier routing — each change verified against the locked
  evaluation baseline before being kept, not assumed safe from theory alone. A follow-up
  pass found and fixed a Redis cache gap that was forcing redundant re-embedding on every
  cache hit, pooled a previously per-call Azure SQL connection, and moved the frontend's
  completion detection from a 2s poll interval to an SSE push. It also *ruled out* two
  plausible-looking heuristics (skipping the reranker for "simple" queries; adaptive
  candidate-pool depth) by directly measuring how often reranking changes the retrieved
  evidence: 0% of a stratified 25-claim sample had an identical top-5 before/after
  reranking, and no query-type signal predicted stability — so neither shortcut had a
  safe target to apply to. Full writeup and measurements in `CLAUDE.md`'s "Production
  Latency" section.
- **Retrieval determinism caveat** — Azure AI Search's hybrid BM25+vector RRF scoring
  drifts slightly run to run; this project explicitly measures old-vs-new code in the
  same session rather than trusting a fingerprint captured on a different day.

## 📁 Project Structure

```
agents/            LangGraph agent nodes (supervisor, retrieval, comparison,
                    sentiment, numeric_validation, report, CrewAI debate crew)
api/                FastAPI app, routes, request/response schemas
azure_clients/      Azure SDK wrappers (AI Search, OpenAI, Redis, Key Vault, SQL,
                    Cosmos, Blob) — never named azure/, which would shadow the SDK
data_pipeline/      Ingestion, hierarchical + semantic chunking, embedding, indexing
data/               Local pipeline output (gitignored) — parsed/chunks/embeddings/raw
evaluation/         Eval runner, RAGAS wrapper, LLM-as-judge, precision/recall@k,
                    golden_dataset/ (75 hand-verified claims), FINAL_REPORT.md
frontend/           React + Vite single-page app
graph/              GraphState schema, LangGraph pipeline wiring
observability/      MLflow, Langfuse, Phoenix setup
tools/              Shared retrieval/reranking/calculation utilities used by agents
tests/              Unit + integration tests
```

## 📚 Documentation

- **`CLAUDE.md`** — architecture detail, locked evaluation baselines, active
  experiments, and the constraints this project runs under (single-variable ablations,
  no compounded changes before measuring, deviation log from the original spec)
- **`MLOPS.md`** — deployment pipeline, cost control, teardown procedure
- **`evaluation/FINAL_REPORT.md`** — latest confirmed evaluation results and methodology

## ⚠️ Disclaimer

This application is built for portfolio and demonstration purposes. AI-generated
financial analysis is a decision-support aid, not investment advice, and should be
independently verified against primary source filings before being relied upon.

## 👨‍💻 Developer

**Prem** | AI/ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-prem332-181717?logo=github)](https://github.com/prem332)
