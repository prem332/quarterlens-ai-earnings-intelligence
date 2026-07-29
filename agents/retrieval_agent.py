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
import re
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

# Boilerplate demotion — SUBTRACTS a fixed penalty from rerank_score for near-
# content-free procedural chunks (transcript operator intros, risk-factors Item 1A
# preambles) that can win the cross-encoder argmax on clean lexical overlap alone.
# Additive, not multiplicative: cross-encoder rerank_score is an unbounded raw
# logit that's frequently negative (observed range roughly -7 to +9 this session)
# — a multiplicative scale factor (e.g. x0.7) makes a negative score LESS negative,
# promoting it instead of demoting it. Subtraction demotes correctly regardless of
# sign. Soft demotion, not a filter: reorders within the already-diversified
# candidate pool rather than shrinking it, so it can't reproduce the
# diversity-cap/section-routing regressions (both failed by removing candidates).
#   BOILERPLATE_PENALTY=0 python evaluation/run_baseline_eval.py --run-name ablate-nopenalty
_BOILERPLATE_PENALTY: float = float(os.environ.get("BOILERPLATE_PENALTY", "10.0"))
_TRANSCRIPT_OPERATOR_RE = re.compile(r"^\s*operator\s*:", re.IGNORECASE)
_SECTION_PREAMBLE_PHRASES = (
    "other than the risk factors listed below",
    "item 1a. risk factors",
    "item 1a risk factors",
)

# Demoting the operator's chunk_index==0 introduction just shifted the problem
# one slot over: the executive's opening remarks and early call-transition
# turns (chunk_index 1-3) are real content, not empty boilerplate, but broadly
# summarize the whole quarter (or are administrative, like "we'll now open the
# call for questions") and
# can still crowd out a segment-specific answer (e.g. a CFO's "we delivered
# $57B revenue..." opener winning rank-1 over the specific Gaming-segment
# figure actually asked about). Can't pattern-match this the way "Operator:"
# was matched — every speaker turn starts with a name, so name-matching would
# false-positive on genuinely on-topic later turns too. Use topical overlap
# with the query instead, same technique validated in sentiment_agent.py, but
# only for early transcript turns specifically (opening remarks are where this
# failure mode actually occurs) and with stopwords excluded (unlike
# sentiment_agent's version) since query/chunk share common words too easily
# otherwise, diluting the signal.
#   EARLY_TURN_OVERLAP_THRESHOLD=0 python evaluation/run_baseline_eval.py --run-name ablate-noearlyturn
_EARLY_TURN_MAX_INDEX = 3
_EARLY_TURN_OVERLAP_THRESHOLD: float = float(
    os.environ.get("EARLY_TURN_OVERLAP_THRESHOLD", "0.25")
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "in", "on", "at", "to", "of", "for", "and", "or", "but", "with", "as",
    "by", "from", "this", "that", "these", "those", "it", "its", "we", "our",
    "what", "which", "who", "how", "did", "do", "does", "will", "would",
    "q1", "q2", "q3", "q4", "fy", "fy2025", "fy2026",
}


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _topic_overlap(query: str, text: str) -> float:
    """Fraction of query's non-stopword tokens also present in text."""
    qt = _tokens(query)
    if not qt:
        return 1.0  # nothing to compare against — don't demote on no signal
    return len(qt & _tokens(text)) / len(qt)


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

    # ── 6. Global cross-encoder rerank (full candidate pool, penalty applied,
    #        then sliced to top-5 — not truncated to top-5 before the penalty,
    #        or a demoted chunk could never be replaced by one ranked lower) ──
    ranked = rerank_documents(
        query=query,
        chunks=mmr_candidates,
        top_k=len(mmr_candidates),
    )
    ranked = _demote_boilerplate(ranked, query)

    # ── 7. Optional diversity cap (disabled by default) ───────────────────
    final = _apply_diversity_cap(ranked, _MAX_CHUNKS_PER_SECTION, _FINAL_TOP_K)

    # ── 8. Map to RetrievalResult for GraphState ──────────────────────────
    retrieval_results = _to_retrieval_results(final, company, quarter)

    # ── 8b. Small-to-big: reconstruct parent blocks for reasoning agents ──
    retrieval_results = _expand_parents(retrieval_results, company, quarter)

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


def _is_boilerplate(chunk: dict, query: str = "") -> bool:
    """
    Two signatures for chunks that can win the cross-encoder argmax without
    being on-topic for the specific question:

    1. Narrow, high-precision pattern match: a transcript operator's
       introduction (stem overlap on "operator"/"operating"), a risk-factors
       Item 1A preamble (lexical "risk(s)" match) — both chunk_index==0,
       near-zero substantive content regardless of query.
    2. Early transcript turns (chunk_index 1-3, right after the operator's
       introduction) that are real content but broadly summarize the whole
       quarter rather than the specific topic asked about — judged by low
       topical overlap with the query, not a fixed pattern (see module note).
    """
    idx = int(chunk.get("chunk_index", -1))
    content = (chunk.get("content") or "").strip()

    if idx == 0:
        if chunk.get("doc_type") == "transcript":
            if _TRANSCRIPT_OPERATOR_RE.match(content):
                return True
        elif (chunk.get("section") or "").lower() in ("risk_factors", "business"):
            lowered = content.lower()[:200]
            if any(p in lowered for p in _SECTION_PREAMBLE_PHRASES):
                return True

    if (
        chunk.get("doc_type") == "transcript"
        and 0 <= idx <= _EARLY_TURN_MAX_INDEX
        and query
        and _topic_overlap(query, content) < _EARLY_TURN_OVERLAP_THRESHOLD
    ):
        return True

    return False


def _demote_boilerplate(ranked: list[dict], query: str = "") -> list[dict]:
    """Subtract the penalty from rerank_score for boilerplate chunks and re-sort.
    See _BOILERPLATE_PENALTY docstring above — demotes, never removes, a candidate."""
    out = []
    for chunk in ranked:
        c = dict(chunk)
        if _is_boilerplate(c, query):
            c["rerank_score"] = c.get("rerank_score", 0.0) - _BOILERPLATE_PENALTY
        out.append(c)
    out.sort(key=lambda c: c["rerank_score"], reverse=True)
    return out


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
            parent_id=    h.get("parent_id", ""),
            parent_index= int(h.get("parent_index", 0)),
            parent_total= int(h.get("parent_total", 1)),
            parent_content="",
        )
        for h in chunks
    ]


def _expand_parents(
    results: list[RetrievalResult],
    company: str,
    quarter: str,
) -> list[RetrievalResult]:
    """
    Small-to-big: reconstruct each child's L2 parent block by fetching its
    siblings (one filtered query per unique parent, cached within the call) and
    concatenating them in parent_index order. `content` stays the precise child
    (retrieval + RAGAS precision); `parent_content` carries the rich context that
    the reasoning agents consume. Degenerate parents (transcripts, single-child)
    skip the fetch — parent_content == content.
    """
    from tools.search_documents import fetch_parent_siblings

    cache: dict[str, str] = {}
    for r in results:
        pid = r.get("parent_id", "")
        if not pid or r.get("parent_total", 1) <= 1:
            r["parent_content"] = r.get("content", "")
            continue
        if pid not in cache:
            siblings = fetch_parent_siblings(pid, company=company, quarter=quarter)
            cache[pid] = "".join(s["content"] for s in siblings) if siblings else r.get("content", "")
        r["parent_content"] = cache[pid]
    return results