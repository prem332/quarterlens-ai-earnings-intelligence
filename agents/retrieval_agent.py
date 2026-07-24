"""
agents/retrieval_agent.py

Hybrid retrieval with global reranking across filing and transcript candidates.

Phase 3/4 retrieval pipeline:
    1. search_documents(filing,     top=10) — raw BM25+vector, no reranking
    2. search_documents(transcript, top=10) — raw BM25+vector, no reranking
    3. Chunk-id deduplication across sources (Fix 1 — Phase 3)
    4. Preserve transcript candidates → transcript_retrieval_results (for sentiment_agent)
    5. Merge (up to 20 unique candidates)
    6. Global MMR (→ 10 diverse candidates across both source types)
    7. Global cross-encoder rerank (→ top 5)
    8. retrieval_results ← globally reranked evidence

Fix 1 (Phase 3): chunk_id deduplication after merge.
    AI Search can return the same chunk in both the filing and transcript passes.
    Deduplicating by chunk_id before MMR ensures each chunk only occupies one
    candidate slot. Filing chunks take priority; transcript duplicates are dropped.
    Confirmed fix for NVDA_FY2026-Q3_cmp_001 identical rank1/rank2 chunks.

Note on diversity cap: tested at cap=2 (baseline-diversity-cap-10) —
    precision@5 dropped 0.76→0.44 because comparison claims need multiple chunks
    from same section. Cap kept at 0 (disabled). Do not re-enable without ablation.

Note on section routing: tested (baseline-section-routing-25) —
    precision@5 dropped 0.817→0.533 because mda-only filter cuts correct evidence
    from risk_factors/business for financial queries. Routing disabled. Do not
    re-enable without redesigning the intent→section mapping.

Why two separate retrieval outputs:
    retrieval_results            → reasoning agents (comparison, report, numeric)
    transcript_retrieval_results → sentiment_agent (FinBERT) needs full transcript pool
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

from graph.state import GraphState, DecisionLogEntry, RetrievalResult
from tools.search_documents import search_documents, mmr_rerank
from tools.rerank_documents import rerank_documents

# Retrieval config
_CANDIDATE_K = 12   # raw candidates per source — slightly over 10 to compensate
                    # for AI Search returning duplicate chunk_ids within one call
                    # (hybrid BM25+vector RRF can surface same doc twice).
                    # After dedup, ~10 unique candidates remain per source.
_FINAL_TOP_K = 5    # final chunks after global cross-encoder rerank

# MMR lambda — overridable via env var for ablation:
#   MMR_LAMBDA=0.7 python evaluation/run_baseline_eval.py --run-name baseline-lambda-070
_MMR_LAMBDA: float = float(os.environ.get("MMR_LAMBDA", "0.5"))

# MMR top_k — candidates passed to cross-encoder:
#   MMR_TOP_K=12 python evaluation/run_baseline_eval.py --run-name baseline-mmrtopk-12
_MMR_TOP_K: int = int(os.environ.get("MMR_TOP_K", "10"))

# Post-reranking diversity cap — disabled (0). Do not enable without ablation.
_MAX_CHUNKS_PER_SECTION: int = int(os.environ.get("MAX_CHUNKS_PER_SECTION", "0"))


def retrieval_agent(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    t0      = time.time()
    company = state["company"]
    quarter = state["quarter"]
    query   = state["query"]

    # ── 1. Raw retrieval — no reranking, independent per source ──────────
    filing_raw     = _raw_search(query, company, quarter, doc_type=None,         label="filing")
    transcript_raw = _raw_search(query, company, quarter, doc_type="transcript", label="transcript")

    # ── 2. Chunk-id deduplication (Fix 1) ────────────────────────────────
    # AI Search can return the same chunk in both passes (e.g. a chunk that
    # matches both the unfiltered filing query and the transcript query).
    # Deduplicating before MMR ensures each chunk occupies only one slot.
    filing_raw, transcript_raw = _dedup_across_sources(filing_raw, transcript_raw)

    # ── 3. Preserve transcript candidates for sentiment_agent ─────────────
    transcript_retrieval_results = _to_retrieval_results(transcript_raw, company, quarter)

    # ── 4. Merge candidates for global reranking ──────────────────────────
    merged = filing_raw + transcript_raw   # up to 20 unique candidates

    # ── 5. Global MMR — diversity across the full merged pool ─────────────
    from azure_clients.openai_client import openai_client
    query_embedding = openai_client.embed(query)   # L1-cached

    mmr_candidates = mmr_rerank(
        chunks=merged,
        query_embedding=query_embedding,
        top_k=_MMR_TOP_K,
        lambda_param=_MMR_LAMBDA,
    )

    # ── 6. Global cross-encoder rerank ────────────────────────────────────
    ranked = rerank_documents(
        query=query,
        chunks=mmr_candidates,
        top_k=_FINAL_TOP_K,
    )

    # ── 7. Optional diversity cap (disabled by default) ───────────────────
    final = _apply_diversity_cap(ranked, _MAX_CHUNKS_PER_SECTION, _FINAL_TOP_K)

    # ── 8. Map to RetrievalResult for GraphState ──────────────────────────
    retrieval_results = _to_retrieval_results(final, company, quarter)

    entry: DecisionLogEntry = {
        "agent":         "retrieval_agent",
        "tool_called":   "search_documents + mmr_rerank + rerank_documents",
        "input_summary": (
            f"company={company} quarter={quarter} "
            f"filing_raw={len(filing_raw)} transcript_raw={len(transcript_raw)}"
        ),
        "output_summary": (
            f"retrieval_results={len(retrieval_results)} (globally reranked) "
            f"transcript_retrieval_results={len(transcript_retrieval_results)} "
            f"pipeline: merged={len(merged)} → mmr={len(mmr_candidates)} → final={len(final)}"
        ),
        "confidence":  None,
        "tokens_used": None,
        "latency_ms":  round((time.time() - t0) * 1000, 1),
    }

    return {
        "retrieval_results":            retrieval_results,
        "transcript_retrieval_results": transcript_retrieval_results,
        "decision_log_entries":         [entry],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dedup_across_sources(
    filing_raw: list[dict],
    transcript_raw: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Deduplicate chunks by chunk_id:
      1. Within filing_raw (AI Search can return same chunk twice in one call)
      2. Within transcript_raw
      3. Across sources (filing takes priority)

    Fixes: NVDA_FY2026-Q3_cmp_001 — identical chunk_id at ranks 1 and 2
    caused by AI Search returning the same chunk twice within one search call.
    """
    seen: set[str] = set()
    deduped_filing: list[dict] = []
    for c in filing_raw:
        cid = c.get("chunk_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            deduped_filing.append(c)
        elif not cid:
            deduped_filing.append(c)  # keep chunks without chunk_id

    deduped_transcript: list[dict] = []
    for c in transcript_raw:
        cid = c.get("chunk_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            deduped_transcript.append(c)
        elif not cid:
            deduped_transcript.append(c)

    return deduped_filing, deduped_transcript


def _apply_diversity_cap(
    ranked: list[dict],
    max_per_section: int,
    top_k: int,
) -> list[dict]:
    """
    Optional post-reranking diversity cap. Disabled by default (max_per_section=0).
    When enabled, enforces max_per_section chunks per (accession, section).
    Cross-encoder order preserved — only slot selection is constrained.
    """
    if max_per_section <= 0:
        return ranked[:top_k]
    section_counts: dict[tuple, int] = defaultdict(int)
    selected: list[dict] = []
    for chunk in ranked:
        if len(selected) >= top_k:
            break
        key = (chunk.get("accession", ""), chunk.get("section", "").lower())
        if section_counts[key] < max_per_section:
            selected.append(chunk)
            section_counts[key] += 1
    return selected


def _raw_search(
    query:    str,
    company:  str,
    quarter:  str,
    doc_type: str | None,
    label:    str,
) -> list[dict]:
    """
    Raw hybrid search for one source type — no MMR, no cross-encoder.
    Returns [] on failure so the pipeline degrades gracefully.
    """
    try:
        result = search_documents(
            query=query,
            doc_type=doc_type,
            company=company,
            quarter=quarter,
            top=_CANDIDATE_K,
            mmr=False,
            rerank=False,
            use_cache=True,
        )
        return result.get("results", [])
    except Exception as exc:
        print(f"[retrieval_agent] raw search failed ({label}): {exc}")
        return []


def _to_retrieval_results(
    chunks:  list[dict],
    company: str,
    quarter: str,
) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=     h.get("chunk_id", ""),
            content=      h.get("content", ""),
            company=      h.get("company",  company),
            quarter=      h.get("quarter",  quarter),
            doc_type=     h.get("doc_type", ""),
            fiscal_label= h.get("fiscal_label", quarter),
            score=        float(h.get("rerank_score", h.get("score", 0.0))),
            accession=    h.get("accession", ""),
            section=      h.get("section", ""),
            chunk_index=  int(h.get("chunk_index", -1)),
            chunk_total=  int(h.get("chunk_total", -1)),
        )
        for h in chunks
    ]