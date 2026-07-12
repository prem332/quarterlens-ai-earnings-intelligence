#!/usr/bin/env bash
# Run this from the root of your cloned quarterlens-ai-earnings-intelligence repo.
set -e

# --- Top-level files ---
touch README.md SETUP.md .gitignore requirements.txt docker-compose.yml Dockerfile
touch .env.example

# --- agents/ ---
mkdir -p agents
touch agents/__init__.py agents/supervisor.py agents/retrieval_agent.py \
      agents/comparison_agent.py agents/sentiment_agent.py \
      agents/numeric_validation_agent.py agents/report_agent.py agents/router.py

# --- graph/ ---
mkdir -p graph/nodes
touch graph/__init__.py graph/build_graph.py graph/state.py
touch graph/nodes/.gitkeep

# --- tools/ ---
mkdir -p tools
touch tools/__init__.py tools/search_documents.py tools/fetch_prior_quarter.py \
      tools/run_finbert.py tools/calculate_metric.py tools/tool_registry.py

# --- data_pipeline/ ---
mkdir -p data_pipeline
touch data_pipeline/__init__.py data_pipeline/edgar_downloader.py \
      data_pipeline/transcript_fetcher.py data_pipeline/document_parser.py \
      data_pipeline/chunking.py data_pipeline/embedding.py data_pipeline/indexer.py

# --- golden_dataset/ ---
mkdir -p golden_dataset/claims
touch golden_dataset/companies.yaml golden_dataset/build_golden_set.py golden_dataset/README.md
touch golden_dataset/claims/.gitkeep

# --- evaluation/ ---
mkdir -p evaluation
touch evaluation/__init__.py evaluation/ragas_eval.py evaluation/precision_recall_at_k.py \
      evaluation/llm_as_judge.py evaluation/run_baseline_eval.py evaluation/run_ablation.py \
      evaluation/eval_gate.py

# --- observability/ ---
mkdir -p observability
touch observability/__init__.py observability/phoenix_setup.py \
      observability/mlflow_tracking.py observability/decision_log.py

# --- azure/ ---
mkdir -p azure
touch azure/__init__.py azure/blob_client.py azure/ai_search_client.py \
      azure/cosmos_client.py azure/sql_client.py azure/key_vault_client.py \
      azure/openai_client.py azure/redis_client.py

# --- api/ ---
mkdir -p api/routes api/middleware api/schemas
touch api/__init__.py api/main.py
touch api/routes/analysis.py api/routes/reports.py api/routes/evidence.py api/routes/export.py
touch api/middleware/rate_limiter.py api/middleware/guardrails.py
touch api/schemas/.gitkeep

# --- frontend/ ---
mkdir -p frontend/src/pages frontend/src/components frontend/src/api
touch frontend/package.json
touch frontend/src/pages/Dashboard.jsx frontend/src/pages/NewAnalysis.jsx \
      frontend/src/pages/AnalysisReport.jsx frontend/src/pages/ReportHistory.jsx \
      frontend/src/pages/EvidenceExplorer.jsx
touch frontend/src/components/.gitkeep frontend/src/api/.gitkeep

# --- finetuning/ (Phase 2) ---
mkdir -p finetuning
touch finetuning/prepare_dataset.py finetuning/run_finetune_job.py \
      finetuning/evaluate_finetuned_vs_baseline.py

# --- infra/ ---
mkdir -p infra/bicep
touch infra/bicep/phase1_core.bicep infra/bicep/redis.bicep \
      infra/bicep/event_grid.bicep infra/bicep/front_door.bicep
touch infra/deploy.sh

# --- .github/workflows/ ---
mkdir -p .github/workflows
touch .github/workflows/eval_gate.yml .github/workflows/deploy.yml
# ci.yml is added separately with real content, not just touched

# --- tests/ ---
mkdir -p tests/unit tests/integration
touch tests/unit/test_agents.py tests/unit/test_tools.py tests/unit/test_numeric_validation.py
touch tests/integration/test_full_pipeline.py

# --- scripts/ ---
mkdir -p scripts
touch scripts/seed_golden_dataset.py scripts/run_local_pipeline.py \
      scripts/generate_ablation_report.py

echo "Skeleton created. Empty dirs need .gitkeep (already added) since git doesn't track empty folders."
