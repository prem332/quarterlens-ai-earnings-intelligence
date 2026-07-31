# MLOps Process — QuarterLens AI

This describes how code moves from a local change to a running production
endpoint, and what's automated vs. deliberately kept manual. Written to
match what's actually implemented in this repo as of this branch, not an
aspirational target — update it when the pipeline changes, don't let it drift.

## Pipeline overview

```
push (any branch)
  └─▶ CI (.github/workflows/ci.yml)
        ├─ lint (ruff, pinned to E9+F — real bugs, not style opinions)
        ├─ test (pytest tests/ — unit tests, currently: input guardrails)
        └─ docker-build (builds the full multi-stage image, never pushed —
                          verifies the image builds, nothing more)

push to main, CI green
  └─▶ Deploy (.github/workflows/deploy.yml, triggered by CI's completion)
        ├─ eval-gate (.github/workflows/eval_gate.yml)
        │     └─ evaluation/smoke_test.py — one real claim through the full
        │        compiled_graph pipeline against live Azure resources.
        │        Asserts: no error, non-empty report, completes within
        │        budget. NOT the RAGAS/LLM-judge suite — see below.
        ├─ build-and-push — tags image with git SHA + `latest`, pushes to ACR
        └─ deploy — az containerapp update to the new image tag, then a
                     real HTTP health check against the live endpoint
```

## Why the eval gate is a smoke test, not the real eval suite

`evaluation/run_baseline_eval.py` (RAGAS scoring + LLM-judge) makes many LLM
calls per claim — it's how every locked baseline in this repo was measured,
and it costs real Azure OpenAI budget every time it runs. Running it as a
blocking CI gate would mean spending that budget on every merge to main,
regardless of whether the change could plausibly affect retrieval/generation
quality (most don't — most merges are UI, CI config, or infra changes).

The smoke test costs roughly one `report_agent` run (draft + verify + bull/bear
debate, ~4 LLM calls) instead of the dozens per claim the full suite makes.
It catches what actually breaks a deploy — a broken Azure client, a bad
prompt template, a schema mismatch, an unhandled exception — without
re-measuring quality on every push.

**The full eval suite stays a deliberate, manual, budget-tracked action**,
run the same way it always has been this project (see CLAUDE.md's
"Running Evaluations" and "Experiment Discipline" sections): single-variable
changes, 10 → 25 claim phasing, results logged to MLflow, compared against
the locked baseline band before anything is called a regression or a win.
Promoting a new baseline as "locked" in `evaluation/FINAL_REPORT.md` is a
human decision made after reading real per-claim results — it is not, and
should not become, an automatic gate.

## What's versioned, and how

| Thing | Versioning mechanism |
|---|---|
| Code | git — this repo |
| Container image | tagged by git SHA on every deploy (plus a floating `latest`); previous tags stay in ACR, so any prior image can be redeployed by tag |
| AI Search index / chunking / embeddings | **not automated.** Re-indexing is a deliberate, measured action (see CLAUDE.md's "Known Issues" #3 and the deferred Fix 6) — never triggered by a code push. Changing the index without a full eval comparison risks moving locked metrics silently. |
| Golden dataset (`golden_dataset/claims/`) | git — hand-verified claim JSONs, changed only by explicit review |
| Eval results / experiment history | MLflow (`mlflow.db`, `mlruns/`) — every eval run, logged with a descriptive run name per CLAUDE.md's discipline |
| Locked baseline metrics | `evaluation/FINAL_REPORT.md` — updated only after a human reads per-claim results, not automatically |

This is a RAG system, not a fine-tuned model — there's no model-weights
versioning story to build. The things that actually determine output quality
(the index, the chunking strategy, the prompts) are versioned by git plus
the manual eval-and-lock discipline above, which is stricter than blind
automation would be: every change to any of them is required to clear a
measured comparison against the locked band before it's trusted.

## Identity, secrets, and config

- **Runtime (the deployed container):** a system-assigned managed identity
  on the Container App, granted the `Key Vault Secrets User` RBAC role on
  `quarterlens-kv`. The app needs exactly one environment variable
  (`AZURE_KEY_VAULT_URL`) — everything else resolves through
  `azure_clients/key_vault_client.py`'s existing Key Vault → `.env` fallback
  chain, unchanged from local dev. No secrets are baked into the image or
  set as container env vars.
- **CI/CD (GitHub Actions):** OIDC federated credential on a Microsoft Entra
  app registration — `azure/login@v2` authenticates the runner without a
  stored client secret. `evaluation/smoke_test.py`'s Key Vault access in the
  eval-gate job piggybacks on that same authenticated `az` CLI session via
  `DefaultAzureCredential`'s `AzureCliCredential` fallback — one identity,
  one auth story, not a separate secret-passing path for CI vs. runtime.
- Required GitHub repo config (Settings → Secrets and variables → Actions):
  - Secrets: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`
    (from the federated app registration — none of these are the client
    *secret*; OIDC needs no secret)
  - Variables: `AZURE_KEY_VAULT_URL`
  - Federated credential subject must match `repo:<org>/<repo>:ref:refs/heads/main`
    (or the specific branch/environment used) for `workflow_run`-triggered
    jobs — this is configured once on the app registration, not per-workflow.

## Cost control

The Container App runs `min-replicas: 0`, so it sleeps after ~5 minutes idle
and bills nothing while asleep. Waking it costs a cold start — image pull,
torch/transformers imports, and cross-encoder + FinBERT loading, measured at
17s minimum and 46s under load.

`scripts/demo_mode.sh on` sets `min-replicas: 1` to remove that entirely, for
demos and interviews. It bills continuously: at 2 vCPU / 4 GiB that is
~172,800 vCPU-seconds/day, which exhausts the monthly Container Apps free
grant in roughly a day. Turn it off (`demo_mode.sh off`) straight after.

Azure SQL is Serverless with auto-pause left ON, deliberately. A paused
database costs nothing but the first connection pays a resume — measured at
49.4s versus 0.95s awake. `api/main.py` fires `sql_client.warm_up()` as a
background task at startup so the resume overlaps container start and the
user's own retrieval/sentiment stages rather than landing inside numeric
validation. Disabling auto-pause would remove the risk entirely but bills
compute continuously.

**Tearing down.** These resources bill indefinitely once any free credit is
gone. To stop everything:

```bash
az group delete -n quarterlens-phase1-rg --yes --no-wait
```

That deletes the AI Search index, Cosmos, SQL, Redis, Blob storage, Container
App, and Container Registry together. The index is rebuildable from
`data/embeddings/` via `data_pipeline/indexer.py`, and reports live in Blob —
export anything worth keeping first.

## Rollback

Azure Container Apps keeps prior revisions by default. Two options, no
custom tooling needed:
- `az containerapp revision list` → `az containerapp revision activate <old-revision>`
  and shift ingress traffic back to it, or
- Redeploy an older image tag directly: `az containerapp update --image
  quarterlensacr.azurecr.io/quarterlens-ai:<older-git-sha>`

## What's deliberately NOT automated

Consistent with this project's whole experimentation discipline (single-variable
changes, confirm before anything costly/hard-to-reverse):
- The full RAGAS/LLM-judge eval suite — manual, budget-tracked, per CLAUDE.md
- AI Search re-indexing / re-embedding / re-chunking — never triggered by CI
- MMR/ablation parameter changes — proposed and run individually, not swept
- Promoting a new "locked" baseline — a human reads the numbers first

## Current status

Everything above is implemented as workflow/code in this repo. **Not yet
provisioned**: the Azure Container Registry (`quarterlensacr`), the Container
Apps environment + app (`quarterlens-api` in `quarterlens-phase1-rg`), and
the GitHub OIDC federated credential / repo secrets. `deploy.yml` and
`eval_gate.yml` will not run successfully until those exist — provisioning
them is a separate, explicit, cost-incurring step (creating billable Azure
resources), not bundled into this doc's authoring.
