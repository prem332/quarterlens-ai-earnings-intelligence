"""
api/blob_helpers.py

Async-safe Blob Storage access for the API layer.

Why this exists: every route was declared `async def` but called the
synchronous Azure Blob SDK directly, which blocks the event loop for the
whole round trip. Measured against East US from this machine:

    BlobClient()      Key Vault fetch + client construction, per request
    blob_exists()     ~1,023 ms
    download_blob()   ~522 ms

The frontend polls GET /{run_id}/status every 2 seconds while an analysis
runs, and the pipeline runs as an asyncio task on that same loop. So each
poll froze the pipeline for 1.5-2.5s out of every 2s. Measured effect on a
real NVDA run: 55.7s of a 127.0s wall clock was spent between pipeline
nodes, doing nothing. The identical query through the graph directly --
same instrumentation, no HTTP server -- took 23.3s with ~0s of inter-node
gap. Confirmed it was not the Phoenix/Langfuse tracing by re-running
standalone with both enabled: still ~0.1s of gap.

Two fixes here:
  1. Reuse one BlobClient instead of constructing one (and re-reading the
     Key Vault secret) on every request.
  2. Run the blocking SDK calls in a worker thread so the event loop stays
     free for the pipeline.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from azure_clients.blob_client import BlobClient


@lru_cache(maxsize=1)
def get_blob() -> BlobClient:
    """Process-wide BlobClient. The underlying BlobServiceClient is
    thread-safe, so sharing it across to_thread workers is fine."""
    return BlobClient()


async def blob_exists(container: str, path: str) -> bool:
    return await asyncio.to_thread(get_blob().blob_exists, container, path)


async def download_blob(container: str, path: str) -> bytes:
    return await asyncio.to_thread(get_blob().download_blob, container, path)


async def upload_blob(container: str, path: str, data: bytes, overwrite: bool = True) -> None:
    await asyncio.to_thread(get_blob().upload_blob, container, path, data, overwrite)


async def list_blobs(container: str, prefix: str) -> list[str]:
    return await asyncio.to_thread(get_blob().list_blobs, container, prefix)
