# syntax=docker/dockerfile:1

# ── Stage 1: build the React/Vite frontend ──────────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build
# -> /app/frontend/dist

# ── Stage 2: Python backend, serving the API + the built frontend ──────────
FROM python:3.10-slim AS backend

# pyodbc/azure_clients/sql_client.py requires "ODBC Driver 18 for SQL Server"
# at the OS level (not a pip package) — install it from Microsoft's apt repo.
#
# NOT using packages-microsoft-prod.deb (Microsoft's own installer package)
# here: its bundled keyring carries a SHA1-based key self-certification that
# apt's newer sequoia/sqv verification backend rejects outright ("SHA1 is not
# considered secure since 2026-02-01") — a Microsoft-side trust-file staleness
# issue, not anything specific to this Dockerfile. Fetching the raw key
# directly and registering the repo with an explicit signed-by keyring is
# the current non-deprecated method and avoids that stale bundled cert.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg unixodbc-dev gcc g++ \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && apt-get purge -y --auto-remove curl gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fixed cache path so the model bake-in below (run as root, at build time)
# and the app's runtime reads (run as the `app` user, after chown below)
# resolve to the exact same directory.
ENV HF_HOME=/app/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake both HF models into the image at build time instead of letting them
# lazy-download from HuggingFace Hub on first use at runtime.
#
# Root-caused a production hang with this: sentiment_agent's first-ever
# call in a fresh container lazily pulled ProsusAI/finbert (~440MB) from HF
# Hub with no pre-warm (unlike the cross-encoder, which IS warmed at
# startup — see tools/rerank_documents.py's warm_up()). Confirmed via
# per-node timing logs on a real stuck production request: every node
# up through comparison_agent completed normally, then sentiment_agent
# never returned. The literal "sending unauthenticated requests to the
# HF Hub" warning showed up during diagnosis, proving a live download was
# in flight, not a cached load. Datacenter egress IPs (Container Apps'
# included) are exactly the kind HF Hub's anonymous-download throttling
# targets hardest, which fits a hang that never reproduced locally or in
# GitHub Actions (both effectively used a warm cache or got through fast).
# Warming the cross-encoder at startup only reduces this risk for that one
# model and only for the *first* request after a cold start; baking both
# in here removes the runtime HF Hub dependency entirely, for good.
RUN python3 -c "from transformers import pipeline; pipeline('text-classification', model='ProsusAI/finbert', tokenizer='ProsusAI/finbert')" \
    && python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')"

# First-party source only — data/, mlruns/, evaluation/, golden_dataset/claims/,
# frontend/node_modules and frontend/src are excluded via .dockerignore. The
# live API queries Azure AI Search directly; it never reads local data/ files.
COPY agents/ ./agents/
COPY api/ ./api/
COPY azure_clients/ ./azure_clients/
COPY data_pipeline/ ./data_pipeline/
COPY graph/ ./graph/
COPY observability/ ./observability/
COPY tools/ ./tools/

# Built frontend from stage 1 — api/main.py serves this from frontend/dist
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health').read()" || exit 1

# Secrets (Azure Key Vault URL, or individual AZURE_* / .env-style fallbacks
# per azure_clients/key_vault_client.py) are supplied at runtime via
# environment variables or --env-file — never baked into the image.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
