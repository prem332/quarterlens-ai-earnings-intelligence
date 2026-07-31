"""
azure_clients/redis_client.py

Multi-level semantic caching for QuarterLens AI.

Cache levels:
  L1 — Embedding cache (in-process dict, then Redis; TTL 7d)
       Caches text → embedding vector, for both single embed() calls and
       embed_batch() chunk embeddings. Two tiers on purpose: the dict is
       the instant path within one process, Redis makes entries survive a
       restart and be shared across replicas. That matters here because
       min-replicas is 0 (scale-to-zero), so a dict-only cache was being
       thrown away on every cold start.

  L2 — Retrieval result cache (Redis, TTL 30min)
       Caches (query+company+quarter) → chunk list. Avoids AI Search +
       MMR + reranker for repeated queries on the same filing.

  L3 — Full report cache (Redis, TTL 24h)
       Caches (query+company+quarter) → final report string. Avoids
       entire 5-agent pipeline for repeated analysis requests.

  Parent blocks (Redis, TTL 24h)
       Caches parent_id → reconstructed parent text for small-to-big
       expansion. Parent content is immutable for a given index build,
       so this only ever needs fetching once.

Design:
  - Lazy singleton: Redis connects once per process on first cache call
  - SSL on port 6380 (Azure Cache for Redis requirement)
  - Graceful degradation: any Redis failure returns None (cache miss)
    so the pipeline continues normally — caching never breaks the app
  - Hit/miss stats tracked for MLflow ablation logging

Key schema:
  L2: "retrieval::{hash(query+company+quarter)}"
  L3: "report::{hash(query+company+quarter)}"

Secrets: AZURE-REDIS-HOST, AZURE-REDIS-KEY from Key Vault
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# TTLs
_L2_TTL_SECONDS = 30 * 60           # 30 minutes — retrieval results
_L3_TTL_SECONDS = 24 * 60 * 60      # 24 hours   — full reports
_L1_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days     — embeddings (text→vector is
                                    # deterministic for a fixed model, so this
                                    # could be permanent; the TTL exists only to
                                    # bound memory on the Basic C0 instance)
_PARENT_TTL_SECONDS = 24 * 60 * 60  # 24 hours   — reconstructed parent blocks

# L1 in-process embedding cache (Python dict — fast path, cleared on restart;
# backed by Redis below so a restart doesn't lose the entries entirely)
_embedding_cache: dict[str, list[float]] = {}
_embedding_hits = 0
_embedding_misses = 0

# L2/L3 Redis hit/miss counters — separate per level for reporting
_l2_hits = 0
_l2_misses = 0
_l3_hits = 0
_l3_misses = 0

# Redis client singleton
_redis_client = None


def _get_redis():
    """Lazy Redis connection — connects once per process."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    try:
        import redis
        from azure_clients.key_vault_client import kv

        host = kv.get_secret("AZURE-REDIS-HOST")
        key = kv.get_secret("AZURE-REDIS-KEY")

        _redis_client = redis.Redis(
            host=host,
            port=6380,
            password=key,
            ssl=True,                    # Azure Cache for Redis requires SSL
            ssl_cert_reqs=None,          # Azure uses self-signed cert
            socket_connect_timeout=5,
            socket_timeout=5,
            decode_responses=True,       # return str not bytes
        )
        # Verify connection
        _redis_client.ping()
        logger.info("RedisClient: connected to %s:6380", host)
        return _redis_client

    except Exception as exc:
        logger.warning("RedisClient: connection failed — cache disabled. Error: %s", exc)
        _redis_client = None
        return None


def _cache_key(prefix: str, query: str, company: str, quarter: str, doc_type: str = "all") -> str:
    """Deterministic cache key from query coordinates.

    doc_type must be part of the key: retrieval_agent runs the filing pass
    (doc_type=None) and transcript pass (doc_type="transcript") back-to-back
    with identical query+company+quarter. Without doc_type in the key, the
    second call collides with the first call's cache entry and silently
    returns the wrong pass's results instead of running its own filtered
    search (confirmed via direct reproduction — see fix commit)."""
    raw = f"{query.strip().lower()}::{company.upper()}::{quarter.upper()}::{(doc_type or 'all').lower()}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}::{digest}"


# ── L1: Embedding Cache (in-process dict → Redis) ─────────────────────────────

def _embedding_key(text: str) -> str:
    """Hash the text — chunk contents run to thousands of characters, which is
    far too long to use directly as a Redis key."""
    return f"emb::{hashlib.sha256(text.strip().encode()).hexdigest()[:32]}"


def get_embedding_cached(text: str) -> Optional[list[float]]:
    """
    L1 cache get — in-process dict first, then Redis. Returns None on miss.

    Counts one hit or one miss per call regardless of which tier served it,
    so l1_hit_rate keeps meaning the same thing it did before Redis backing
    was added.
    """
    global _embedding_hits, _embedding_misses
    key = text.strip()
    if key in _embedding_cache:
        _embedding_hits += 1
        logger.debug("L1 cache HIT (memory): embedding for '%s...'", key[:40])
        return _embedding_cache[key]

    client = _get_redis()
    if client is not None:
        try:
            value = client.get(_embedding_key(text))
            if value:
                embedding = json.loads(value)
                _embedding_cache[key] = embedding   # promote to the fast path
                _embedding_hits += 1
                logger.debug("L1 cache HIT (redis): embedding for '%s...'", key[:40])
                return embedding
        except Exception as exc:
            logger.warning("L1 redis get failed (non-fatal): %s", exc)

    _embedding_misses += 1
    return None


def set_embedding_cached(text: str, embedding: list[float]) -> None:
    """L1 cache set — writes to both the in-process dict and Redis."""
    _embedding_cache[text.strip()] = embedding

    client = _get_redis()
    if client is None:
        return
    try:
        client.setex(_embedding_key(text), _L1_TTL_SECONDS, json.dumps(embedding))
    except Exception as exc:
        logger.warning("L1 redis set failed (non-fatal): %s", exc)


def get_embeddings_batch_cached(texts: list[str]) -> list[Optional[list[float]]]:
    """
    Batch L1 get — one entry per input text, None where not cached.

    IN-PROCESS ONLY, deliberately — unlike the single-text functions above,
    this does NOT read from Redis. Measured reasons, not a style choice:

      * One 1536-dim vector is ~21 KB as JSON. A single MMR call covers ~24
        chunks = ~0.49 MB moved per retrieval, which cancels out most of the
        embedding API call it was meant to replace.
      * Caching every indexed chunk would need ~72 MB on a 250 MB Basic C0
        instance, crowding out the L2 retrieval and L3 report entries that
        deliver much larger wins per byte stored.

    Query embeddings (embed(), one small vector, reused across sessions) are
    still Redis-backed — that trade is clearly worth it. Chunk embeddings are
    not. If chunk vectors ever need to survive a restart, the right fix is the
    retrievable-embedding index (they are already stored in AI Search), not a
    second copy in Redis.

    Order-preserving and index-aligned with `texts` so the caller can embed
    only the misses and splice results back into place.
    """
    global _embedding_hits, _embedding_misses
    out: list[Optional[list[float]]] = [None] * len(texts)
    for i, text in enumerate(texts):
        key = text.strip()
        if key in _embedding_cache:
            out[i] = _embedding_cache[key]
            _embedding_hits += 1
        else:
            _embedding_misses += 1
    return out


def set_embeddings_batch_cached(texts: list[str], embeddings: list[list[float]]) -> None:
    """Batch L1 set — in-process only; see get_embeddings_batch_cached."""
    for text, embedding in zip(texts, embeddings):
        _embedding_cache[text.strip()] = embedding


# ── L3b: Full run cache (API path) ────────────────────────────────────────────
#
# get_report_cached/set_report_cached above store only the report STRING and are
# used solely by evaluation/run_baseline_eval.py. The API needs more than that:
# the report page also renders comparison findings, and the on-demand CrewAI
# debate endpoint reads retrieval_results back off the stored run. So the API
# caches the whole result payload instead.
#
# comparison_quarters is part of the key -- the same question against different
# comparison periods is a different analysis and must not collide.

def _run_key(query: str, company: str, quarter: str, comparison_quarters: list[str]) -> str:
    raw = (
        f"{query.strip().lower()}::{company.upper()}::{quarter.upper()}"
        f"::{','.join(sorted(q.upper() for q in comparison_quarters))}"
    )
    return f"run::{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def get_run_cached(
    query: str, company: str, quarter: str, comparison_quarters: list[str],
) -> Optional[dict]:
    """Full cached run payload for an identical request, or None."""
    global _l3_hits, _l3_misses
    client = _get_redis()
    if client is None:
        return None
    try:
        value = client.get(_run_key(query, company, quarter, comparison_quarters))
        if value:
            _l3_hits += 1
            logger.info("L3 cache HIT: full run for %s/%s", company, quarter)
            return json.loads(value)
        _l3_misses += 1
        return None
    except Exception as exc:
        logger.warning("L3 run cache get failed (non-fatal): %s", exc)
        return None


def set_run_cached(
    query: str, company: str, quarter: str, comparison_quarters: list[str], payload: dict,
) -> None:
    """Cache a completed run. Callers must not pass failed or empty-report runs."""
    client = _get_redis()
    if client is None:
        return
    try:
        client.setex(
            _run_key(query, company, quarter, comparison_quarters),
            _L3_TTL_SECONDS,
            json.dumps(payload),
        )
        logger.debug("L3 cache SET: full run for %s/%s", company, quarter)
    except Exception as exc:
        logger.warning("L3 run cache set failed (non-fatal): %s", exc)


# ── Parent block cache (small-to-big expansion) ───────────────────────────────

def get_parent_cached(parent_id: str) -> Optional[str]:
    """Reconstructed parent block text for a parent_id, or None."""
    client = _get_redis()
    if client is None or not parent_id:
        return None
    try:
        return client.get(f"parent::{parent_id}")
    except Exception as exc:
        logger.warning("Parent cache get failed (non-fatal): %s", exc)
        return None


def set_parent_cached(parent_id: str, content: str) -> None:
    """Store a reconstructed parent block. Immutable per index build."""
    client = _get_redis()
    if client is None or not parent_id:
        return
    try:
        client.setex(f"parent::{parent_id}", _PARENT_TTL_SECONDS, content)
    except Exception as exc:
        logger.warning("Parent cache set failed (non-fatal): %s", exc)


# ── L2: Retrieval Result Cache ────────────────────────────────────────────────

def get_retrieval_cached(
    query: str,
    company: str,
    quarter: str,
    doc_type: str = "all",
) -> Optional[list[dict]]:
    """
    L2 cache get — returns cached chunk list or None.
    TTL: 30 minutes. doc_type distinguishes the filing pass (None/"all") from
    the transcript pass — see _cache_key docstring for why this matters.
    """
    global _l2_hits, _l2_misses
    client = _get_redis()
    if client is None:
        return None

    key = _cache_key("retrieval", query, company, quarter, doc_type)
    try:
        value = client.get(key)
        if value:
            _l2_hits += 1
            logger.info("L2 cache HIT: retrieval for %s/%s (doc_type=%s)", company, quarter, doc_type)
            return json.loads(value)
        _l2_misses += 1
        return None
    except Exception as exc:
        logger.warning("L2 cache get failed (non-fatal): %s", exc)
        return None


def set_retrieval_cached(
    query: str,
    company: str,
    quarter: str,
    chunks: list[dict],
    doc_type: str = "all",
) -> None:
    """L2 cache set — stores chunk list in Redis with 30min TTL."""
    client = _get_redis()
    if client is None:
        return

    key = _cache_key("retrieval", query, company, quarter, doc_type)
    try:
        client.setex(key, _L2_TTL_SECONDS, json.dumps(chunks))
        logger.debug("L2 cache SET: retrieval for %s/%s", company, quarter)
    except Exception as exc:
        logger.warning("L2 cache set failed (non-fatal): %s", exc)


# ── L3: Full Report Cache ─────────────────────────────────────────────────────

def get_report_cached(
    query: str,
    company: str,
    quarter: str,
) -> Optional[str]:
    """
    L3 cache get — returns cached report string or None.
    TTL: 24 hours.
    """
    global _l3_hits, _l3_misses
    client = _get_redis()
    if client is None:
        return None

    key = _cache_key("report", query, company, quarter)
    try:
        value = client.get(key)
        if value:
            _l3_hits += 1
            logger.info("L3 cache HIT: report for %s/%s", company, quarter)
            return value
        _l3_misses += 1
        return None
    except Exception as exc:
        logger.warning("L3 cache get failed (non-fatal): %s", exc)
        return None


def set_report_cached(
    query: str,
    company: str,
    quarter: str,
    report: str,
) -> None:
    """L3 cache set — stores report string in Redis with 24h TTL."""
    client = _get_redis()
    if client is None:
        return

    key = _cache_key("report", query, company, quarter)
    try:
        client.setex(key, _L3_TTL_SECONDS, report)
        logger.debug("L3 cache SET: report for %s/%s", company, quarter)
    except Exception as exc:
        logger.warning("L3 cache set failed (non-fatal): %s", exc)


# ── Stats for MLflow ablation logging ─────────────────────────────────────────

def get_cache_stats() -> dict[str, Any]:
    """
    Returns hit/miss stats across all cache levels — L1, and L2/L3 both
    separately and combined (combined fields preserved for backward
    compatibility with existing report/MLflow field names).
    Call after eval run to log to MLflow.
    """
    l1_total = _embedding_hits + _embedding_misses
    l2_total = _l2_hits + _l2_misses
    l3_total = _l3_hits + _l3_misses
    redis_total = l2_total + l3_total

    return {
        "l1_embedding_hits": _embedding_hits,
        "l1_embedding_misses": _embedding_misses,
        "l1_hit_rate": round(_embedding_hits / l1_total, 4) if l1_total else 0.0,
        "l2_hits": _l2_hits,
        "l2_misses": _l2_misses,
        "l2_hit_rate": round(_l2_hits / l2_total, 4) if l2_total else 0.0,
        "l3_hits": _l3_hits,
        "l3_misses": _l3_misses,
        "l3_hit_rate": round(_l3_hits / l3_total, 4) if l3_total else 0.0,
        "l2_l3_redis_hits": _l2_hits + _l3_hits,
        "l2_l3_redis_misses": _l2_misses + _l3_misses,
        "l2_l3_hit_rate": round((_l2_hits + _l3_hits) / redis_total, 4) if redis_total else 0.0,
    }


def clear_all_caches() -> None:
    """Clear L1 in-memory cache and flush Redis (use for testing only)."""
    global _embedding_cache, _embedding_hits, _embedding_misses
    global _l2_hits, _l2_misses, _l3_hits, _l3_misses

    _embedding_cache.clear()
    _embedding_hits = 0
    _embedding_misses = 0
    _l2_hits = 0
    _l2_misses = 0
    _l3_hits = 0
    _l3_misses = 0

    client = _get_redis()
    if client:
        try:
            client.flushdb()
            logger.info("Redis cache flushed.")
        except Exception as exc:
            logger.warning("Redis flush failed: %s", exc)