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
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg unixodbc-dev gcc g++ \
    && curl -sSL -O https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && apt-get purge -y --auto-remove curl gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
