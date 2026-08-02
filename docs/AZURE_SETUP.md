# QuarterLens AI — Azure Setup, End to End

**Scope note:** this is a reproducible runbook — the real resource names, tiers, regions,
and configuration values documented here are exactly what backs the live deployment
(cross-checked against `CLAUDE.md` and live `az` queries). The `az` commands below are the
correct commands to (re)create this exact setup from scratch, not a literal transcript of
every historical command originally run (that predates this document). Where a command was
actually executed and logged during a real session — the AKS section — it's marked as such.

---

## 1. Architecture summary

```
GitHub (push to main)
        │
        ▼
GitHub Actions CI  ──lint + pytest + Docker build verify──▶  (on pass)
        │
        ▼
GitHub Actions Deploy ──OIDC login (no stored secret)──▶ Azure
        │
        ├── eval-gate (smoke-test + quality-gate) ──must pass──▶
        ├── build-and-push  ──▶  Azure Container Registry (quarterlensacr)
        └── deploy          ──▶  Azure Container Apps (quarterlens-api)
                                        │
                                        ├── Azure AI Search    (quarterlens-search)
                                        ├── Azure OpenAI       (quarterlens-openai)
                                        ├── Azure Cosmos DB    (quarterlens-cosmos)
                                        ├── Azure SQL          (quarterlens-sqlserver)
                                        ├── Azure Cache Redis  (quarterlens-redis)
                                        ├── Azure Blob Storage (quarterlensstorage)
                                        └── Azure Key Vault    (quarterlens-kv) — all
                                            secrets resolved here at runtime via the
                                            Container App's system-assigned managed
                                            identity + RBAC, never baked into the image
```

All resources live in one resource group: **`quarterlens-phase1-rg`**.

---

## 2. Prerequisites

- An Azure subscription (`az login` to authenticate)
- Azure CLI ≥ 2.60
- A GitHub repo with Actions enabled (for the CI/CD path)
- Docker (for local builds only — not required for the Azure-hosted build, which runs in
  GitHub Actions)

```bash
az login
az account set --subscription "<subscription-id>"
```

---

## 3. Resource group

```bash
az group create -n quarterlens-phase1-rg -l eastus
```

---

## 4. Key Vault (create this early — everything else's secrets land here)

```bash
az keyvault create \
  -n quarterlens-kv \
  -g quarterlens-phase1-rg \
  -l eastus \
  --enable-rbac-authorization true
```

RBAC authorization (not the legacy access-policy model) — every secret is granted via
**role assignments** (`Key Vault Secrets User` for read-only identities, `Key Vault Secrets
Officer` for the identity that writes secrets), not a vault-level access-policy list.
Secret names use hyphens (`AZURE-OPENAI-ENDPOINT`, not underscores) —
`azure_clients/key_vault_client.py` converts between the two conventions automatically.

---

## 5. Azure AI Search

```bash
az search service create \
  -n quarterlens-search \
  -g quarterlens-phase1-rg \
  -l eastus \
  --sku free
```

Free (F0) tier — no cost, at the expense of shared/multi-tenant compute (this is the
documented external latency floor for fresh queries, see `CLAUDE.md`'s "Sub-stage
retrieval profiling"). Index (`quarterlens-filings`) is created by
`data_pipeline/indexer.py`, not by `az` directly — see README's "Populating the index from
scratch."

```bash
# Store the admin key in Key Vault
az search admin-key show -n quarterlens-search -g quarterlens-phase1-rg \
  --query primaryKey -o tsv
az keyvault secret set --vault-name quarterlens-kv \
  -n AZURE-SEARCH-ENDPOINT --value "https://quarterlens-search.search.windows.net"
az keyvault secret set --vault-name quarterlens-kv \
  -n AZURE-SEARCH-ADMIN-KEY --value "<key from above>"
```

---

## 6. Azure OpenAI

```bash
az cognitiveservices account create \
  -n quarterlens-openai \
  -g quarterlens-phase1-rg \
  -l eastus \
  --kind OpenAI \
  --sku S0
```

Deployments needed (create via Azure AI Foundry portal or `az cognitiveservices account
deployment create`):
- `gpt-5-mini` — standard tier, 10K TPM (dev/simple-query routing)
- `gpt-5.4-mini` — Global Standard (production/primary routing) — **requires
  `api_version="2024-12-01-preview"` and `max_completion_tokens` (not `max_tokens`),
  minimum 4096 tokens** (see `azure_clients/openai_client.py`'s own header comment)
- `text-embedding-3-small` — 1536-dim embeddings

```bash
az keyvault secret set --vault-name quarterlens-kv -n AZURE-OPENAI-ENDPOINT \
  --value "https://quarterlens-openai.openai.azure.com/"
az keyvault secret set --vault-name quarterlens-kv -n AZURE-OPENAI-KEY --value "<key>"
az keyvault secret set --vault-name quarterlens-kv -n AZURE-OPENAI-DEPLOYMENT-NAME \
  --value "gpt-5.4-mini"
az keyvault secret set --vault-name quarterlens-kv -n AZURE-OPENAI-DEPLOYMENT-NAME-STANDARD \
  --value "gpt-5-mini"
```

---

## 7. Azure Cosmos DB (decision log)

```bash
az cosmosdb create \
  -n quarterlens-cosmos \
  -g quarterlens-phase1-rg \
  -l westus2 \
  --enable-free-tier true
```

Free Tier covers the first 1000 RU/s + 25GB storage forever — the app's own
`decision_log` container is provisioned at 400 RU/s, comfortably inside that (confirmed
zero-cost regardless of traffic, verified live this session). Database/container
(`quarterlens` / `decision_log`, partition key `/run_id`) are created idempotently at
runtime by `azure_clients/cosmos_client.py`'s `_get_or_create_container()`, not by `az`.

```bash
az cosmosdb keys list -n quarterlens-cosmos -g quarterlens-phase1-rg \
  --query primaryMasterKey -o tsv
az keyvault secret set --vault-name quarterlens-kv -n AZURE-COSMOS-URI \
  --value "https://quarterlens-cosmos.documents.azure.com:443/"
az keyvault secret set --vault-name quarterlens-kv -n AZURE-COSMOS-KEY --value "<key>"
```

---

## 8. Azure SQL (financial_facts table)

```bash
az sql server create \
  -n quarterlens-sqlserver \
  -g quarterlens-phase1-rg \
  -l centralus \
  --admin-user <admin-username> \
  --admin-password <admin-password>

az sql db create \
  -n quarterlens-db \
  -s quarterlens-sqlserver \
  -g quarterlens-phase1-rg \
  --edition GeneralPurpose \
  --family Gen5 \
  --capacity 1 \
  --compute-model Serverless \
  --auto-pause-delay 60
```

Serverless, Free tier, 60-minute auto-pause — confirmed idle-cost-safe (verified live:
`status=Paused` with zero compute billing while idle). Cold-resume from pause takes up to
~49s (documented, `azure_clients/sql_client.py`'s own header comment); the connection
itself is pooled and health-checked, not reopened per call.

```bash
# Allow Azure services (including Container Apps) through the firewall
az sql server firewall-rule create \
  -g quarterlens-phase1-rg -s quarterlens-sqlserver \
  -n AllowAzureServices --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0

az keyvault secret set --vault-name quarterlens-kv -n AZURE-SQL-SERVER \
  --value "quarterlens-sqlserver.database.windows.net"
az keyvault secret set --vault-name quarterlens-kv -n AZURE-SQL-DATABASE --value "quarterlens-db"
az keyvault secret set --vault-name quarterlens-kv -n AZURE-SQL-USERNAME --value "<admin-username>"
az keyvault secret set --vault-name quarterlens-kv -n AZURE-SQL-PASSWORD --value "<admin-password>"
```

---

## 9. Azure Cache for Redis

```bash
az redis create \
  -n quarterlens-redis \
  -g quarterlens-phase1-rg \
  -l eastus \
  --sku Basic \
  --vm-size c0
```

Basic C0 — the one resource in this stack with no idle-cost mitigation (no auto-pause, no
scale-to-zero; confirmed via live inspection this session — it bills continuously whether
or not there's traffic, roughly ₹1.5-2/hour). Multi-level cache: L1 in-process embedding
cache, L2 retrieval-result cache, L3 full-report cache (24h TTL) — see
`azure_clients/redis_client.py`.

```bash
az redis list-keys -n quarterlens-redis -g quarterlens-phase1-rg \
  --query primaryKey -o tsv
az keyvault secret set --vault-name quarterlens-kv -n AZURE-REDIS-HOST \
  --value "quarterlens-redis.redis.cache.windows.net"
az keyvault secret set --vault-name quarterlens-kv -n AZURE-REDIS-KEY --value "<key>"
```

---

## 10. Blob Storage

```bash
az storage account create \
  -n quarterlensstorage \
  -g quarterlens-phase1-rg \
  -l eastus \
  --sku Standard_LRS

az storage container create \
  --account-name quarterlensstorage \
  -n raw-documents
```

```bash
az storage account show-connection-string -n quarterlensstorage -g quarterlens-phase1-rg \
  --query connectionString -o tsv
az keyvault secret set --vault-name quarterlens-kv -n AZURE-BLOB-CONNECTION-STRING \
  --value "<connection string>"
```

---

## 11. Azure Container Registry + Container Apps

```bash
az acr create -n quarterlensacr -g quarterlens-phase1-rg -l eastus --sku Basic

az containerapp env create \
  -n quarterlens-env \
  -g quarterlens-phase1-rg \
  -l eastus

az containerapp create \
  -n quarterlens-api \
  -g quarterlens-phase1-rg \
  --environment quarterlens-env \
  --image quarterlensacr.azurecr.io/quarterlens-ai:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 0 --max-replicas 2 \
  --system-assigned \
  --registry-server quarterlensacr.azurecr.io
```

`--min-replicas 0` — Consumption tier scale-to-zero, the deliberate cost-control choice for
this app (see `docs/MLOPS.md`). `--system-assigned` provisions a managed identity, which is
then RBAC-granted read access to Key Vault (below) — the app never holds a Key Vault
credential directly.

```bash
# Grant the Container App's managed identity read access to Key Vault
PRINCIPAL_ID=$(az containerapp show -n quarterlens-api -g quarterlens-phase1-rg \
  --query identity.principalId -o tsv)
KV_ID=$(az keyvault show -n quarterlens-kv --query id -o tsv)
az role assignment create --assignee "$PRINCIPAL_ID" \
  --role "Key Vault Secrets User" --scope "$KV_ID"

# The one env var the container needs directly -- everything else resolves from Key Vault
az containerapp update -n quarterlens-api -g quarterlens-phase1-rg \
  --set-env-vars AZURE_KEY_VAULT_URL="https://quarterlens-kv.vault.azure.net/"
```

---

## 12. CI/CD — GitHub Actions with OIDC (no stored long-lived secret)

Federated credential setup (Azure AD app registration + federated identity linking it to
this specific GitHub repo/branch), so `azure/login@v2` in CI authenticates without a
service-principal secret sitting in GitHub Secrets:

```bash
az ad app create --display-name quarterlens-github-actions
APP_ID=$(az ad app list --display-name quarterlens-github-actions --query "[0].appId" -o tsv)
az ad sp create --id "$APP_ID"

az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "quarterlens-main-branch",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:prem332/quarterlens-ai-earnings-intelligence:ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'

# Grant this identity what it needs to build/push/deploy
az role assignment create --assignee "$APP_ID" --role "AcrPush" \
  --scope $(az acr show -n quarterlensacr --query id -o tsv)
az role assignment create --assignee "$APP_ID" --role "Container Apps Contributor" \
  --scope $(az containerapp show -n quarterlens-api -g quarterlens-phase1-rg --query id -o tsv)
```

GitHub repo secrets required: `AZURE_CLIENT_ID` (the app ID above), `AZURE_TENANT_ID`,
`AZURE_SUBSCRIPTION_ID`. GitHub repo variable: `AZURE_KEY_VAULT_URL`.

**Pipeline, exactly as wired in `.github/workflows/`:**
1. `ci.yml` — every push, every branch: ruff lint, `pytest tests/ -q` (97 tests), Docker
   build verification (build only, no push).
2. `eval_gate.yml` — triggered by `deploy.yml` after CI succeeds on `main`. Two jobs,
   cheap-first: `smoke-test` (one real claim through the full pipeline, completion/
   non-empty/time-budget check) then `quality-gate` (10 stratified claims scored against
   the 7 locked RAGAS/judge/retrieval metrics, regression-guard floors — fails the build on
   a real quality regression, not just a crash).
3. `deploy.yml` — `build-and-push` (Docker build → ACR, SHA-tagged + `:latest`) then
   `deploy` (`az containerapp update --image ...`) then a post-deploy health check with
   patient retries (cold start after scale-to-zero + model warm-up can take ~50s).

---

## 13. Monitoring / observability setup

All optional, each degrades gracefully if unset (see `.env.example`):

```bash
# Application Insights (infra monitoring)
az monitor app-insights component create \
  -a quarterlens-insights -g quarterlens-phase1-rg -l eastus
az monitor app-insights component show -a quarterlens-insights -g quarterlens-phase1-rg \
  --query connectionString -o tsv
```

- **MLflow** — local sqlite tracking store (`mlflow.db`, gitignored — see `CLAUDE.md` for
  why it's not committed), or set `MLFLOW_TRACKING_URI` to a remote server.
- **Langfuse** — set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` (used for
  the production cross-validation runs documented in `CLAUDE.md`).
- **Arize Phoenix** — set `PHOENIX_ENDPOINT`/`PHOENIX_API_KEY`.

---

## 14. Kubernetes / AKS (resume POC — not the production serving path)

Production stays on Container Apps for the scale-to-zero cost economics on a low-traffic
app (see above). This is a deliberately separate, timeboxed proof-of-concept, actually run
and torn down in a single session — commands below are the **real, executed** commands,
not just a reference:

```bash
# One-time prerequisite (this subscription had never used AKS before)
az provider register --namespace Microsoft.ContainerService

# East US quota on this subscription excludes amd64 B-series VMs entirely --
# Standard_D2s_v3 is the cheapest amd64 SKU on the allowed list
az aks create -g quarterlens-phase1-rg -n quarterlens-aks \
  --node-count 1 --node-vm-size Standard_D2s_v3 \
  --enable-managed-identity --generate-ssh-keys \
  --attach-acr quarterlensacr --tier free

# Grant the AKS kubelet identity Key Vault read access -- same RBAC pattern as
# Container Apps' managed identity above. Run via PowerShell, not Git Bash: Git Bash
# mangles any CLI argument starting with "/" (a --scope value) into a bogus
# Windows path, breaking this command with a confusing MissingSubscription error.
IDENTITY=$(az aks show -g quarterlens-phase1-rg -n quarterlens-aks \
  --query identityProfile.kubeletidentity.objectId -o tsv)
az role assignment create --assignee-object-id "$IDENTITY" \
  --assignee-principal-type ServicePrincipal \
  --role "Key Vault Secrets User" \
  --scope $(az keyvault show -n quarterlens-kv --query id -o tsv)

az aks get-credentials -g quarterlens-phase1-rg -n quarterlens-aks
kubectl apply -f k8s/namespace.yaml -f k8s/configmap.yaml \
  -f k8s/deployment.yaml -f k8s/service.yaml
kubectl -n quarterlens get pods -w
kubectl -n quarterlens port-forward svc/quarterlens-api 8000:80
curl http://localhost:8000/api/health   # -> {"status":"ok"}

# Teardown, same sitting -- verified the resource group returned to its exact
# original set afterward, no orphaned node resource group
az aks delete -g quarterlens-phase1-rg -n quarterlens-aks --yes
az role assignment delete --assignee-object-id "$IDENTITY" \
  --role "Key Vault Secrets User" --scope $(az keyvault show -n quarterlens-kv --query id -o tsv)
```

Manifests live in `k8s/` — `namespace.yaml`, `configmap.yaml`, `deployment.yaml` (`Service`
account, resource requests/limits sized for `Standard_D2s_v3`, `startupProbe` tuned for the
real model-warm-up cost), `service.yaml` (`ClusterIP`, not `LoadBalancer` — a Standard Load
Balancer bills continuously for no benefit in a same-session verify-and-teardown POC).

---

## 15. Cost control / teardown

Full policy in `docs/MLOPS.md`. Summary of what's already zero-cost-when-idle vs. not
(verified live, this session):

| Resource | Idle cost |
|---|---|
| Container App (`min-replicas=0`) | $0 |
| Azure SQL (Serverless, 60-min auto-pause) | $0 once paused |
| Cosmos DB (Free Tier, 400 RU/s provisioned) | $0 |
| AI Search (Free F0) | $0 |
| **Azure Cache for Redis (Basic C0)** | **~₹1.5-2/hour, always** — no pause/scale-to-zero exists for this tier; delete it if leaving the project idle for an extended period |

Full teardown:

```bash
az group delete -n quarterlens-phase1-rg --yes --no-wait
```

Rebuildable from `data/embeddings/` via `data_pipeline/indexer.py` — export any wanted
reports from Blob Storage first.
