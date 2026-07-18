"""
finetuning/run_finetune_job.py

Submits gpt-4o-mini fine-tuning job to Azure OpenAI (Foundry).

Steps:
  1. Upload training.jsonl and validation.jsonl via SDK (files API unaffected by version issue)
  2. Create fine-tuning job via direct REST POST with api-version=2025-04-01-preview
     (SDK fine_tuning.jobs.create ignores client api_version on older SDK builds)
  3. Poll job status until succeeded / failed
  4. Print fine-tuned model ID → add to Key Vault as AZURE-OPENAI-DEPLOYMENT-NAME-FINETUNED

Note on region: gpt-4o-mini Global training works from East US resource.
trainingType=globalstandard requires api-version=2025-04-01-preview.

Usage:
  python -m finetuning.run_finetune_job
  python -m finetuning.run_finetune_job --dry-run   # validate files only, no upload
"""

import argparse
import json
import time
import logging
from pathlib import Path

import requests
from openai import AzureOpenAI
from azure_clients.key_vault_client import kv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_FINETUNING_DIR = Path(__file__).parent
_TRAIN_PATH = _FINETUNING_DIR / "training.jsonl"
_VAL_PATH = _FINETUNING_DIR / "validation.jsonl"

# ── Fine-tuning config ────────────────────────────────────────────────────────

_BASE_MODEL = "gpt-4o-mini-2024-07-18"
_N_EPOCHS = 3
_SEED = 42
_POLL_INTERVAL_SEC = 60
_FT_API_VERSION = "2025-04-01-preview"  # required for trainingType=globalstandard


def _get_secrets() -> tuple[str, str]:
    endpoint = kv.get_secret("AZURE-OPENAI-ENDPOINT").rstrip("/")
    api_key = kv.get_secret("AZURE-OPENAI-KEY")
    return endpoint, api_key


def _make_sdk_client(endpoint: str, api_key: str) -> AzureOpenAI:
    """SDK client for file uploads only — files API is not version-sensitive."""
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=_FT_API_VERSION,
    )


def _validate_files() -> None:
    for path in (_TRAIN_PATH, _VAL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        lines = path.read_text(encoding="utf-8-sig").strip().splitlines()
        if not lines:
            raise ValueError(f"Empty file: {path}")
        logger.info("Validated %s — %d lines", path.name, len(lines))


def _upload_file(client: AzureOpenAI, path: Path) -> str:
    logger.info("Uploading %s ...", path.name)
    with open(path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    file_id = response.id
    logger.info("Uploaded %s → file_id=%s", path.name, file_id)
    _wait_for_file_processed(client, file_id, path.name)
    return file_id


def _wait_for_file_processed(client: AzureOpenAI, file_id: str, name: str) -> None:
    """Poll file status until 'processed'. Required before referencing in a job."""
    logger.info("Waiting for %s (%s) to finish processing ...", name, file_id)
    for attempt in range(30):
        file_obj = client.files.retrieve(file_id)
        status = file_obj.status
        logger.info("  file status=%s (attempt %d)", status, attempt + 1)
        if status == "processed":
            return
        if status == "error":
            raise RuntimeError(f"File {file_id} failed processing: {file_obj}")
        time.sleep(5)
    raise RuntimeError(f"File {file_id} did not reach 'processed' status after 150s")


def _submit_job_rest(endpoint: str, api_key: str, train_file_id: str, val_file_id: str) -> str:
    """
    Submit fine-tuning job via direct REST POST.
    Uses api-version=2025-04-01-preview which correctly handles trainingType=globalstandard.
    The SDK's fine_tuning.jobs.create hard-codes an older api-version on some builds.
    """
    url = f"{endpoint}/openai/fine_tuning/jobs?api-version={_FT_API_VERSION}"
    payload = {
        "model": _BASE_MODEL,
        "training_file": train_file_id,
        "validation_file": val_file_id,
        "hyperparameters": {"n_epochs": _N_EPOCHS},
        "seed": _SEED,
        "trainingType": "globalstandard",
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    logger.info(
        "Submitting fine-tuning job via REST: model=%s, epochs=%d, trainingType=globalstandard",
        _BASE_MODEL, _N_EPOCHS,
    )
    resp = requests.post(url, headers=headers, json=payload, timeout=30)

    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Fine-tuning job submission failed [{resp.status_code}]: {resp.text}"
        )

    job = resp.json()
    job_id = job["id"]
    logger.info("Job submitted → job_id=%s", job_id)
    return job_id


def _poll_job_rest(endpoint: str, api_key: str, job_id: str) -> str:
    """Poll job status via REST until terminal state. Returns fine-tuned model ID."""
    url = f"{endpoint}/openai/fine_tuning/jobs/{job_id}?api-version={_FT_API_VERSION}"
    headers = {"api-key": api_key}

    logger.info("Polling job %s every %ds ...", job_id, _POLL_INTERVAL_SEC)
    while True:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            logger.warning("Poll request failed [%d]: %s", resp.status_code, resp.text)
            time.sleep(_POLL_INTERVAL_SEC)
            continue

        job = resp.json()
        status = job.get("status", "unknown")
        trained_tokens = job.get("trained_tokens")

        logger.info(
            "  status=%s | trained_tokens=%s",
            status,
            trained_tokens if trained_tokens else "pending",
        )

        if status == "succeeded":
            model_id = job.get("fine_tuned_model")
            logger.info("Job succeeded — fine_tuned_model=%s", model_id)
            return model_id

        if status in ("failed", "cancelled"):
            error = job.get("error")
            raise RuntimeError(
                f"Fine-tuning job {job_id} ended with status={status}. Error: {error}"
            )

        time.sleep(_POLL_INTERVAL_SEC)


def run(dry_run: bool = False) -> None:
    _validate_files()

    if dry_run:
        print("\n[DRY RUN] File validation passed. No upload or job submission performed.")
        print(f"  Training file  : {_TRAIN_PATH}")
        print(f"  Validation file: {_VAL_PATH}")
        print(f"  Base model     : {_BASE_MODEL}")
        print(f"  Epochs         : {_N_EPOCHS}")
        print(f"  API version    : {_FT_API_VERSION}")
        print(f"  Training type  : globalstandard")
        return

    endpoint, api_key = _get_secrets()
    client = _make_sdk_client(endpoint, api_key)

    # 1. Upload files via SDK
    train_file_id = _upload_file(client, _TRAIN_PATH)
    val_file_id = _upload_file(client, _VAL_PATH)

    # 2. Submit job via REST (bypasses SDK api_version pinning issue)
    job_id = _submit_job_rest(endpoint, api_key, train_file_id, val_file_id)

    # 3. Poll to completion via REST
    fine_tuned_model = _poll_job_rest(endpoint, api_key, job_id)

    # 4. Summary
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"  Job ID              : {job_id}")
    print(f"  Base model          : {_BASE_MODEL}")
    print(f"  Fine-tuned model ID : {fine_tuned_model}")
    print(f"  Training file ID    : {train_file_id}")
    print(f"  Validation file ID  : {val_file_id}")
    print()
    print("Next step — add to Key Vault:")
    print(f"  az keyvault secret set \\")
    print(f"    --vault-name quarterlens-kv \\")
    print(f"    --name AZURE-OPENAI-DEPLOYMENT-NAME-FINETUNED \\")
    print(f"    --value \"{fine_tuned_model}\"")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit gpt-4o-mini fine-tuning job")
    parser.add_argument("--dry-run", action="store_true", help="Validate files only, no API calls")
    args = parser.parse_args()
    run(dry_run=args.dry_run)