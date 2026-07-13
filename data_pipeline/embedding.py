from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embedding")

API_VERSION = "2024-10-21"      # stable; supports text-embedding-3-small
BATCH_SIZE = 100                # chunks per embedding request
EMBED_DIM = 1536                # text-embedding-3-small native dimension
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0             # seconds, doubled per retry


def make_client() -> AzureOpenAI:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_KEY")
    if not endpoint or not key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set (see .env)."
        )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=API_VERSION,
    )


def _embed_batch(
    client: AzureOpenAI, deployment: str, texts: list[str]
) -> list[list[float]]:
    """Embed one batch with retry/backoff. Returns vectors in input order."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=deployment, input=texts)
            # API guarantees resp.data is index-aligned to input
            return [item.embedding for item in resp.data]
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning("  batch failed (attempt %d/%d): %s — retrying in %.0fs",
                        attempt, MAX_RETRIES, e, wait)
            time.sleep(wait)
    return []  # unreachable


def embed_chunks(
    client: AzureOpenAI,
    deployment: str,
    chunks: list[dict],
) -> list[dict]:
    """
    Add an 'embedding' field to each chunk dict.
    Chunks are processed in batches of BATCH_SIZE.
    """
    out: list[dict] = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vectors = _embed_batch(client, deployment, [c["text"] for c in batch])
        for chunk, vec in zip(batch, vectors):
            out.append({**chunk, "embedding": vec})
    return out


def run(chunk_manifest_path: str, out_root: str) -> None:
    manifest_p = Path(chunk_manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Chunk manifest not found: {chunk_manifest_path}")

    deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT must be set (see .env).")

    chunk_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    out_root_p = Path(out_root)
    client = make_client()
    embedding_manifest: list[dict] = []

    for entry in chunk_manifest:
        chunks_path = Path(entry["chunks_path"])
        if not chunks_path.exists():
            log.warning("Missing chunk file, skipping: %s", chunks_path)
            continue

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        log.info(
            "Embedding %s %s (%s) — %d chunks",
            entry["ticker"], entry["fiscal_label"], entry["form"], len(chunks)
        )

        embedded = embed_chunks(client, deployment, chunks)

        out_dir = out_root_p / entry["ticker"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{entry['fiscal_label']}_{entry['form'].replace('-', '')}_embeddings.json"
        out_file.write_text(
            json.dumps(embedded, ensure_ascii=False), encoding="utf-8"
        )

        log.info("  %s %s: %d vectors -> %s",
                 entry["ticker"], entry["fiscal_label"], len(embedded), out_file.name)

        embedding_manifest.append({
            **{k: entry[k] for k in
               ("ticker", "cik", "fiscal_label", "form", "report_date", "accession")},
            "chunk_count":     entry["chunk_count"],
            "embedded_count":  len(embedded),
            "embeddings_path": str(out_file),
        })

    embedding_manifest_path = out_root_p / "embedding_manifest.json"
    embedding_manifest_path.write_text(
        json.dumps(embedding_manifest, indent=2), encoding="utf-8"
    )

    total = sum(e["embedded_count"] for e in embedding_manifest)
    log.info(
        "Done. %d/%d filings embedded, %d total vectors. Manifest: %s",
        len(embedding_manifest), len(chunk_manifest), total, embedding_manifest_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunked filings for QuarterLens AI.")
    parser.add_argument("--manifest", default="data/chunks/chunk_manifest.json")
    parser.add_argument("--out", default="data/embeddings")
    args = parser.parse_args()
    run(args.manifest, args.out)


if __name__ == "__main__":
    main()