# 📊 QuarterLens AI

Production-grade Earnings Intelligence Platform powered by a Multi-Agent RAG pipeline, Azure OpenAI, Azure AI Search and LangGraph — built 100% on Microsoft Azure

---

## 🎬 Video Demo

*Demo video link to be added.*

> Demo will cover: problem statement, the multi-agent retrieval/verification pipeline, live analysis walkthrough, evaluation methodology (RAGAS + LLM-as-Judge against a 75-claim golden dataset), and production monitoring (Langfuse/Application Insights).

---

## 🌐 Live Demo

**https://quarterlens-api.calmsand-fcf08f52.eastus.azurecontainerapps.io**

> **Note:** The Azure resources backing this project (AI Search, OpenAI, SQL, Redis) are run on-demand to manage cost and may be paused between working sessions — see `docs/MLOPS.md` for the teardown/cost-control policy. If the link above is unresponsive, the resources are likely paused; run locally via **Quick Start** below, or ask for a live walkthrough. Consumption-tier Container Apps also scale to zero when idle, so the first request after a period of inactivity pays a real cold-start cost (container spin-up + model warm-up, ~50s) — subsequent requests are fast (see Evaluation Results below).

---

## 🎯 Project Overview

QuarterLens AI cross-verifies what executives say on quarterly earnings calls against what their companies actually filed with the SEC. Ask a question like *"Did Azure revenue growth accelerate this quarter, and did management's tone match the numbers?"* and it retrieves the relevant 10-Q/10-K and transcript passages, verifies every number against a structured financial-facts database, checks whether the language in the call matches the language in the filing, scores the tone with FinBERT, and drafts a cited, self-verified report.

Covers 5 companies (AAPL, MSFT, NVDA, GOOGL, META) across 5 fiscal quarters, retrieving over a 3,500-chunk hybrid-search index built from 25 filings and 25 earnings-call transcripts.

### Three Core Capabilities
- **Retrieval-grounded Q&A** — hybrid (BM25 + vector) search across filings and transcripts, reranked and cited
- **Numeric fact-checking** — every number in the draft report is verified against a SQL-backed financial-facts table before the report ships
- **Language-shift + sentiment analysis** — LLM-based comparison of current vs. prior-quarter filing language, plus deterministic FinBERT sentiment scoring of the earnings call itself

---

## 🏗️ Architecture

### High-Level System Architecture
```
User Browser
     │
Azure Container Apps (Consumption, scale-to-zero)
     │
FastAPI + Uvicorn (REST + SSE streaming, port 8000)
├── Input Guardrails (PII, prompt injection, harmful content, off-topic filtering)
├── Session/run tracking (UUID per analysis)
     │
LangGraph Multi-Agent Pipeline (StateGraph)
├── supervisor_init
├── retrieval_agent          — hybrid search + MMR + cross-encoder rerank
├── comparison_agent  ┐
├── sentiment_agent   ├── run as three concurrent branches off one retrieval pass
├── numeric_validation┘
├── report_agent             — draft + targeted verify pass
└── supervisor_finalize       — writes the audit trail to Cosmos DB
     │
MLflow + Langfuse + Phoenix (experiment tracking, LLM tracing)
     │
React + Vite Frontend (SSE live progress, citations, report export)
```

### Detailed Query Flow
```
Step 1: User submits a question (company, quarter, natural-language query)
        │
        Input guardrails: PII / prompt-injection / harmful-content / off-topic checks
        (runs before any LLM call — a rejection costs zero tokens)
        │
Step 2: retrieval_agent
        ├── Hybrid search: filing pass + transcript pass, run concurrently
        ├── Cross-source dedup, merge into one candidate pool
        ├── MMR rerank (relevance vs. diversity, λ=0.5)
        ├── Cross-encoder rerank → top-5
        └── Small-to-big: reconstruct parent context for reasoning agents
        │
Step 3: Three agents run concurrently off that one retrieval pass
        ├── comparison_agent    — LLM language-shift check vs. prior quarter
        ├── sentiment_agent     — FinBERT scoring of transcript passages
        └── numeric_validation  — every number checked against Azure SQL
        │
Step 4: report_agent
        ├── Drafts a cited report from all three agents' findings
        ├── Runs a targeted verify pass (skipped only when every numeric
        │   claim in the draft is provably present in the evidence)
        └── (on demand) CrewAI bull/bear debate over the same evidence
        │
Step 5: Streamed back to the React frontend
        ├── Live per-stage progress (SSE) — not a simulated progress bar
        ├── Token-level report drafting stream
        └── Full citations back to source filing/transcript passages
```

---

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
| **Containerization** | Docker (also validated on Kubernetes/AKS — see `k8s/`) |
| **Build & Deploy** | GitHub Actions (OIDC, no stored secret) → Azure Container Registry → Azure Container Apps |

---

## 📊 Evaluation Results

Evaluated against the full 75-claim hand-verified golden dataset (retrieval, comparison, numeric, sentiment, and out-of-scope claim types) spanning all 5 companies and 5 fiscal quarters. Headline numbers below are from real HTTP requests against the **deployed production API**, not an in-process pipeline call — every claim went through `POST /api/analysis/run` → poll `/status` → `GET /api/reports/{run_id}`, `no_cache=true` forced on every request so no result was served from cache, Redis flushed immediately beforehand. Methodology, run history, and the project's single-variable-ablation discipline are tracked in `CLAUDE.md`.

### RAGAS Evaluation (production, n=75, `k=2` measurement window)

| Metric | Score | Locked Target | Status |
|--------|-------|----------------|--------|
| **Faithfulness** | 0.9646 | 0.90 | ✅ PASS |
| **Answer Relevancy** | 0.9361 | 0.90 | ✅ PASS |
| **Context Precision** | 0.8176 | 0.60 | ✅ PASS |
| **Context Recall** | 0.8218 (0.8682 excl. out-of-scope) | 0.90 | close, not cleared (see Known Issues in `CLAUDE.md`) |

### Retrieval Metrics

| Metric | Score | Target |
|--------|-------|--------|
| **Precision@5** | 0.7222 | 0.60 |
| **Recall@5** | 1.0000 | — |

### LLM-as-Judge

**4.09 / 5 (81.8%)** at n=75 against production. Target is 4.5/5 (90%) — not cleared, and this project does not have an active plan to close that remaining gap further (the root cause is an architectural MMR concentration-vs-diversity tradeoff, fully diagnosed in `CLAUDE.md`, with four prior remediation attempts already tried and reverted for regressing other metrics).

### Production Latency (n=75 traces, real end-to-end HTTP wall time)

| Metric | Value |
|--------|------:|
| p50 | 5.99s |
| p90 | 9.27s |
| p95 | 9.41s |
| p99 | 21.44s |
| mean | 7.06s |
| Error rate | 1.3% (1/75 — an input-guardrail false positive, not an infra failure) |

Independently cross-validated via Langfuse's own OTEL trace instrumentation on the exact same run: pure backend execution time of p50=4.74s / p90=6.93s / p95=7.62s / p99=20.46s — consistently *lower* than the end-to-end numbers above at every percentile, exactly as expected since it excludes network time and polling overhead. Notably, both independent measurement methods caught the same p99 latency spike, real evidence of a genuine backend-side event (consistent with an Azure SQL Serverless cold-resume) rather than a measurement artifact.

**Real cost for this run: $0.947491 total** — input $0.725825, output $0.218435, input cached-tokens $0.003149.

> `context_precision` here is an order-insensitive relevant-chunk fraction over the top-k, judged by an LLM per chunk — not the RAGAS-paper rank-weighted Average Precision. `k` and the per-chunk text window are measurement-scope parameters, not retrieval changes; both values are reported so the number is reproducible, not cherry-picked.

### Automated Test Results

- **97/97 tests passing**, ruff lint clean
- **Unit tests** — `agents/` (router, comparison_agent, sentiment_agent), `numeric_validation_agent`, `tools/` (calculate_metric, rerank_documents, search_documents, run_finbert), `api/guardrails.py`
- **Integration tests** — full LangGraph pipeline wiring: fan-out/fan-in across the three parallel agents, the decision-log reducer, error routing
- All offline — no live Azure calls, no real model loads (Key-Vault-backed client singletons are stubbed, `azure.cosmos.CosmosClient` specifically patched since it uniquely makes a real network call at construction time)
- CI-gated: a failing eval-metric quality gate (RAGAS/judge/retrieval scores against regression-guard floors, not just a completion check) blocks the Azure Container Apps deploy — see "Deployment" below

---

## ✨ Features

### Core AI Features
- Hybrid BM25 + vector retrieval across 5 companies, 5 quarters, filings + transcripts
- MMR diversity reranking + cross-encoder relevance reranking on a globally merged candidate pool
- Small-to-big retrieval: precise child chunks for scoring, full parent context for reasoning
- Deterministic numeric fact-checking against a structured Azure SQL table — no LLM in the loop for numbers
- FinBERT sentiment scoring — deterministic, sentence-level, topically ranked against the query
- Self-verifying report generation: a targeted verify pass catches unsupported claims before they ship
- On-demand CrewAI bull/bear debate over the same retrieved evidence

### Security Features
- Input guardrails run before any LLM call (a rejection spends zero tokens): PII detection (email, phone, SSN, credit card), prompt-injection pattern matching (instruction-override, role-hijack, fake chat-role tags), harmful-content keyword filtering, off-topic/out-of-scope detection
- Secrets never baked into the image or committed — resolved from Azure Key Vault (RBAC) at runtime via the deployed workload's managed identity
- Live-verified against the deployed production API across all four guardrail categories, plus two confirmed real-world false-positive fixes (see `CLAUDE.md`)

### Reliability & Performance Features
- Multi-level Redis caching (query embeddings, retrieval results, full report responses)
- Real per-stage SSE progress streaming — not a simulated progress bar
- Cross-tier LLM fallback with `Retry-After` honoring on Azure OpenAI 429s
- Model-tier routing override for the report agent, measured 5.4x faster on the same prompt
- Startup warm-up for FinBERT, Redis, and Azure SQL (mitigates Serverless cold-resume latency)

### MLOps & Evaluation Features
- 75-claim hand-verified golden dataset across 5 claim types
- RAGAS + LLM-as-Judge + precision/recall@k, tracked per-run in MLflow with per-claim artifacts
- CI-gated deployment: a two-stage eval gate (cheap completion smoke test, then a 10-claim RAGAS/judge/retrieval quality gate against regression-guard floors) blocks the deploy on a real quality regression, not just a crash
- Independent cross-validation of every production latency/cost claim via Langfuse OTEL tracing, not a single self-reported source
- Documented single-variable-ablation discipline — every retrieval/generation change measured in isolation, with a rolled-back-experiments log kept in `CLAUDE.md`

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11, Node.js 18+
- An Azure subscription with: AI Search, Azure OpenAI (chat + embedding deployments), Cosmos DB, Azure SQL, Cache for Redis, Blob Storage, Key Vault — full setup runbook in `docs/AZURE_SETUP.md`

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

`requirements.txt` is the full development set (eval suite, tests, ingestion tooling). The deployed container installs `requirements/requirements-api.txt` — a runtime-only subset that trims ~235 MB of packages the running service never imports (see that file's header for exactly what's excluded and why).

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

---

## 🔐 Environment Variables

```env
# Optional: use real Key Vault instead of the vars below (requires az login,
# a managed identity, or service-principal env vars DefaultAzureCredential picks up)
AZURE_KEY_VAULT_URL=

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
AZURE_OPENAI_DEPLOYMENT_NAME=              # gpt-5.4-mini
AZURE_OPENAI_DEPLOYMENT_NAME_STANDARD=     # gpt-5-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=         # text-embedding-3-small

# Azure AI Search
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_ADMIN_KEY=
AZURE_SEARCH_INDEX=quarterlens-filings

# Azure Cosmos DB (decision log)
AZURE_COSMOS_URI=
AZURE_COSMOS_KEY=

# Azure Cache for Redis (L2/L3 retrieval + report cache)
AZURE_REDIS_HOST=
AZURE_REDIS_KEY=

# Azure Blob Storage (serves raw source documents for citation display)
AZURE_BLOB_CONNECTION_STRING=

# Azure SQL (financial_facts table)
AZURE_SQL_SERVER=
AZURE_SQL_DATABASE=
AZURE_SQL_USERNAME=
AZURE_SQL_PASSWORD=

# Observability -- all optional, each degrades gracefully if unset
APPLICATIONINSIGHTS_CONNECTION_STRING=
PHOENIX_ENDPOINT=
PHOENIX_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=

# SEC EDGAR fetcher (offline ingestion only, not required to run the API)
SEC_USER_AGENT=
```

Every secret is resolved from Azure Key Vault first if `AZURE_KEY_VAULT_URL` is set and reachable, falling back to these environment variables — see `azure_clients/key_vault_client.py`.

---

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
| `GET` | `/api/health` | Health check |
| `GET` | `/docs` | Swagger UI |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -q

# Run a specific level
pytest tests/unit/ -q
pytest tests/integration/ -q

# Lint
ruff check .
```

**Test Coverage:**
- Unit tests — `test_agents.py` (router, comparison_agent, sentiment_agent), `test_numeric_validation.py`, `test_tools.py` (calculate_metric, rerank_documents, search_documents, run_finbert), `test_guardrails.py`
- Integration tests — `test_full_pipeline.py`: graph wiring, parallel fan-out/fan-in, the decision-log `operator.add` reducer, error routing
- **Total: 97/97 tests passing**, ruff clean, all offline (no live Azure calls, no real model downloads)

---

## 🐳 Docker & Deployment

```bash
# Build
docker build -t quarterlens-api .

# Run locally
docker run -p 8000:8000 --env-file .env quarterlens-api
```

### Deployment Architecture

```
GitHub (push to main)
     │
GitHub Actions CI
├── ruff lint
├── pytest (97 tests)
└── Docker build verification
     │ (on pass)
GitHub Actions Deploy — OIDC login, no stored secret
├── eval-gate
│   ├── smoke-test      — 1 real claim, completion/time-budget check
│   └── quality-gate    — 10 claims, RAGAS/judge/retrieval regression-guard floors
│                          (must pass or the deploy is blocked)
├── build-and-push  →  Azure Container Registry (quarterlensacr, SHA-tagged + :latest)
└── deploy          →  az containerapp update
     │
Azure Container Apps (quarterlens-api)
├── Consumption tier, min-replicas=0 (scale-to-zero)
├── System-assigned managed identity → Key Vault (RBAC), no baked-in secrets
└── Post-deploy health check with patient retries (cold start ~50s)
```

Production deployment is CI-driven, not manual. See `.github/workflows/` for the full pipeline, `docs/AZURE_SETUP.md` for the complete infrastructure runbook, and `docs/MLOPS.md` for cost-control and teardown procedure.

A Kubernetes/AKS deployment path also exists and was live-validated (cluster created, pod healthy, `/api/health` verified through a real Service, then fully torn down) — see `k8s/` and `docs/AZURE_SETUP.md` §14. This is a deliberately separate proof-of-concept, not the production serving path.

---

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
  completion detection from a 2s poll interval to an SSE push.
- **Deploy-time bug caught by reading production logs, not trusting a green checkmark** —
  a passing CI/deploy pipeline still shipped a container where every Azure SQL connection
  silently failed: `msodbcsql18`'s `--no-install-recommends` install dropped
  `libgssapi-krb5-2`, a dependency the driver is unconditionally linked against regardless
  of auth method. `ldd` against the actual running container (`az containerapp exec`)
  found it in under a minute; the CI smoke test alone had no way to catch it.
- **Two real eval-gate bugs found and fixed via the gate's own first live runs** — a
  committed `mlflow.db` with a Windows-absolute artifact path crashed on Linux CI; and the
  gate initially never flushed Redis before running, so it could have passed indefinitely by
  replaying stale cached reports without ever re-testing the live pipeline. Both fixed and
  documented in `CLAUDE.md`.
- **Retrieval determinism caveat** — Azure AI Search's hybrid BM25+vector RRF scoring
  drifts slightly run to run; this project explicitly measures old-vs-new code in the
  same session rather than trusting a fingerprint captured on a different day.

---

## 📁 Project Structure

```
agents/             LangGraph agent nodes (supervisor, retrieval, comparison,
                     sentiment, numeric_validation, report, CrewAI debate crew)
api/                 FastAPI app, routes, request/response schemas, guardrails
azure_clients/       Azure SDK wrappers (AI Search, OpenAI, Redis, Key Vault, SQL,
                     Cosmos, Blob) — never named azure/, which would shadow the SDK
data_pipeline/       Ingestion, hierarchical + semantic chunking, embedding, indexing
data/                Local pipeline output (gitignored) — parsed/chunks/embeddings/raw
docs/                Azure setup runbook, MLOps/cost-control policy
evaluation/          Eval runner, RAGAS wrapper, LLM-as-judge, precision/recall@k,
                     deploy quality gate, golden_dataset/ (75 hand-verified claims)
frontend/            React + Vite single-page app
graph/               GraphState schema, LangGraph pipeline wiring
k8s/                 Kubernetes/AKS manifests (resume POC, not the production path)
observability/       MLflow, Langfuse, Phoenix setup
requirements/        Runtime-only dependency subset installed by the Docker image
tools/               Shared retrieval/reranking/calculation utilities used by agents
tests/               Unit + integration tests (97 tests, all offline)
```

---

## 📚 Documentation

- **`CLAUDE.md`** — architecture detail, locked evaluation baselines, active
  experiments, and the constraints this project runs under (single-variable ablations,
  no compounded changes before measuring, deviation log from the original spec)
- **`docs/AZURE_SETUP.md`** — full Azure infrastructure runbook, resource by resource,
  plus the CI/CD OIDC setup and the Kubernetes/AKS POC
- **`docs/MLOPS.md`** — deployment pipeline, cost control, teardown procedure
- **`evaluation/FINAL_REPORT.md`** — latest confirmed evaluation results and methodology

---

## 📡 Monitoring

Production observability spans three independent tools, cross-validated against each other rather than trusted individually:
- **Langfuse** — full LLM call tracing, per-call and per-trace latency percentiles, real token cost breakdown by type (input/output/cached)
- **MLflow** — per-run experiment tracking with per-claim eval artifacts (RAGAS scores, judge reasoning)
- **Application Insights** — infra-level monitoring for the Container App itself

---

## ⚠️ Disclaimer

This application is built for portfolio and demonstration purposes. AI-generated
financial analysis is a decision-support aid, not investment advice, and should be
independently verified against primary source filings before being relied upon.

---

## 👨‍💻 Developer

**Prem** | AI/ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-prem332-181717?logo=github)](https://github.com/prem332)

> Live Project: quarterlens-api.calmsand-fcf08f52.eastus.azurecontainerapps.io
