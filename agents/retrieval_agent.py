"""
agents/retrieval_agent.py

Hybrid retrieval with global reranking across filing and transcript candidates.

Phase 3 retrieval pipeline:
    1. search_documents(filing,     top=10) — raw BM25+vector, no reranking
    2. search_documents(transcript, top=10) — raw BM25+vector, no reranking
    3. Preserve transcript candidates → transcript_retrieval_results (for sentiment_agent)
    4. Merge (20 candidates)
    5. Global MMR (→ 10 diverse candidates across both source types)
    6. Global cross-encoder rerank (→ top 5)
    7. retrieval_results ← globally reranked evidence (for comparison/report/numeric agents)

Why two separate retrieval outputs:
    retrieval_results            → reasoning agents (comparison, report, numeric)
                                   need highest-relevance evidence across both sources
    transcript_retrieval_results → sentiment_agent (FinBERT) needs maximum transcript
                                   coverage; global reranking can leave 0-1 transcript
                                   chunks in top-5 if filing evidence scores higher
"""

from __future__ import annotations

import os
import time

from graph.state import GraphState, DecisionLogEntry, RetrievalResult
from tools.search_documents import search_documents, mmr_rerank
from tools.rerank_documents import rerank_documents

# Retrieval config
_CANDIDATE_K = 10   # raw candidates per source (filing + transcript)
_MMR_TOP_K   = 10   # candidates after global MMR
_FINAL_TOP_K = 5    # final chunks after global cross-encoder rerank

# MMR lambda — overridable via env var for ablation runs:
#   MMR_LAMBDA=0.7 python evaluation/run_baseline_eval.py --run-name baseline-lambda-070
_MMR_LAMBDA: float = float(os.environ.get("MMR_LAMBDA", "0.5"))

# MMR top_k — candidates passed to cross-encoder; ablation target (10 vs 12 vs 15):
#   MMR_TOP_K=12 python evaluation/run_baseline_eval.py --run-name baseline-mmrtopk-12
_MMR_TOP_K: int = int(os.environ.get("MMR_TOP_K", "10"))


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

    # ── 2. Preserve transcript candidates for sentiment_agent ─────────────
    # sentiment_agent reads transcript_retrieval_results directly so FinBERT
    # is not constrained by the top-5 global ranking (which may be filing-heavy).
    transcript_retrieval_results = _to_retrieval_results(transcript_raw, company, quarter)

    # ── 3. Merge candidates for global reranking ──────────────────────────
    merged = filing_raw + transcript_raw   # up to 20 candidates

    # ── 4. Global MMR — diversity across the full merged pool ─────────────
    # query embedding is L1-cached (already computed inside search_documents)
    from azure_clients.openai_client import openai_client
    query_embedding = openai_client.embed(query)

    mmr_candidates = mmr_rerank(
        chunks=merged,
        query_embedding=query_embedding,
        top_k=_MMR_TOP_K,
        lambda_param=_MMR_LAMBDA,
    )

    # ── 5. Global cross-encoder rerank ────────────────────────────────────
    ranked = rerank_documents(
        query=query,
        chunks=mmr_candidates,
        top_k=_FINAL_TOP_K,
    )

    # ── 6. Map to RetrievalResult for GraphState ──────────────────────────
    retrieval_results = _to_retrieval_results(ranked, company, quarter)

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
            f"pipeline: merged={len(merged)} → mmr={len(mmr_candidates)} → final={len(ranked)}"
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


def _raw_search(
    query:    str,
    company:  str,
    quarter:  str,
    doc_type: str | None,
    label:    str,
) -> list[dict]:
    """
    Raw hybrid search for one source type — no MMR, no cross-encoder.
    doc_type='transcript' filters the index on form='transcript'.
    doc_type=None returns all forms (10-Q, 10-K, transcript) for the filing pass;
    the cross-encoder will naturally rank filing content higher for financial queries.
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
        )
        for h in chunks
    ]