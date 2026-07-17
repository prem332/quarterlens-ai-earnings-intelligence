"""
azure_clients/redis_client.py

Multi-level semantic caching for QuarterLens AI.

Cache levels:
  L1 — Embedding cache (Python dict, in-process, instant)
       Caches query → embedding vector. Avoids repeat embed() API calls.

  L2 — Retrieval result cache (Redis, TTL 30min)
       Caches (query+company+quarter) → chunk list. Avoids AI Search +
       MMR + reranker for repeated queries on the same filing.

  L3 — Full report cache (Redis, TTL 24h)
       Caches (query+company+quarter) → final report string. Avoids
       entire 5-agent pipeline for repeated analysis requests.

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
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# TTLs
_L2_TTL_SECONDS = 30 * 60       # 30 minutes — retrieval results
_L3_TTL_SECONDS = 24 * 60 * 60  # 24 hours   — full reports

# L1 in-process embedding cache (Python dict — no TTL, cleared on restart)
_embedding_cache: dict[str, list[float]] = {}
_embedding_hits = 0
_embedding_misses = 0

# L2/L3 Redis hit/miss counters
_redis_hits = 0
_redis_misses = 0

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


def _cache_key(prefix: str, query: str, company: str, quarter: str) -> str:
    """Deterministic cache key from query coordinates."""
    raw = f"{query.strip().lower()}::{company.upper()}::{quarter.upper()}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}::{digest}"


# ── L1: Embedding Cache ───────────────────────────────────────────────────────

def get_embedding_cached(text: str) -> Optional[list[float]]:
    """
    L1 cache get — returns cached embedding or None.
    Key: exact query string (case-sensitive).
    """
    global _embedding_hits, _embedding_misses
    key = text.strip()
    if key in _embedding_cache:
        _embedding_hits += 1
        logger.debug("L1 cache HIT: embedding for '%s...'", key[:40])
        return _embedding_cache[key]
    _embedding_misses += 1
    return None


def set_embedding_cached(text: str, embedding: list[float]) -> None:
    """L1 cache set — stores embedding in process memory."""
    _embedding_cache[text.strip()] = embedding


# ── L2: Retrieval Result Cache ────────────────────────────────────────────────

def get_retrieval_cached(
    query: str,
    company: str,
    quarter: str,
) -> Optional[list[dict]]:
    """
    L2 cache get — returns cached chunk list or None.
    TTL: 30 minutes.
    """
    global _redis_hits, _redis_misses
    client = _get_redis()
    if client is None:
        return None

    key = _cache_key("retrieval", query, company, quarter)
    try:
        value = client.get(key)
        if value:
            _redis_hits += 1
            logger.info("L2 cache HIT: retrieval for %s/%s", company, quarter)
            return json.loads(value)
        _redis_misses += 1
        return None
    except Exception as exc:
        logger.warning("L2 cache get failed (non-fatal): %s", exc)
        return None


def set_retrieval_cached(
    query: str,
    company: str,
    quarter: str,
    chunks: list[dict],
) -> None:
    """L2 cache set — stores chunk list in Redis with 30min TTL."""
    client = _get_redis()
    if client is None:
        return

    key = _cache_key("retrieval", query, company, quarter)
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
    global _redis_hits, _redis_misses
    client = _get_redis()
    if client is None:
        return None

    key = _cache_key("report", query, company, quarter)
    try:
        value = client.get(key)
        if value:
            _redis_hits += 1
            logger.info("L3 cache HIT: report for %s/%s", company, quarter)
            return value
        _redis_misses += 1
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
    Returns hit/miss stats across all cache levels.
    Call after eval run to log to MLflow.
    """
    l1_total = _embedding_hits + _embedding_misses
    redis_total = _redis_hits + _redis_misses

    return {
        "l1_embedding_hits": _embedding_hits,
        "l1_embedding_misses": _embedding_misses,
        "l1_hit_rate": round(_embedding_hits / l1_total, 4) if l1_total else 0.0,
        "l2_l3_redis_hits": _redis_hits,
        "l2_l3_redis_misses": _redis_misses,
        "l2_l3_hit_rate": round(_redis_hits / redis_total, 4) if redis_total else 0.0,
    }


def clear_all_caches() -> None:
    """Clear L1 in-memory cache and flush Redis (use for testing only)."""
    global _embedding_cache, _embedding_hits, _embedding_misses
    global _redis_hits, _redis_misses

    _embedding_cache.clear()
    _embedding_hits = 0
    _embedding_misses = 0
    _redis_hits = 0
    _redis_misses = 0

    client = _get_redis()
    if client:
        try:
            client.flushdb()
            logger.info("Redis cache flushed.")
        except Exception as exc:
            logger.warning("Redis flush failed: %s", exc)