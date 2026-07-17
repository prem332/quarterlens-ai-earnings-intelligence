"""
evaluation/precision_recall_at_k.py
Precision@k and Recall@k for QuarterLens AI retrieval evaluation.

Ground truth relevant documents are identified by filing coordinates
(accession + section) from the golden dataset — index-independent anchors
that survive re-chunking. At eval time, retrieved chunk IDs are resolved
back to their filing coordinates for comparison.

Usage:
    from evaluation.precision_recall_at_k import compute_retrieval_metrics

    metrics = compute_retrieval_metrics(
        retrieved_chunks=[...],   # list of chunk dicts from AI Search
        ground_truth_anchors=[{"accession": "...", "section": "mda"}],
        k=5,
    )
    # {"precision_at_k": 0.8, "recall_at_k": 1.0, "k": 5}
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)


def _anchor_key(accession: str, section: str) -> str:
    """Canonical string key for a filing coordinate anchor."""
    return f"{accession.strip()}::{section.strip().lower()}"


def compute_retrieval_metrics(
    retrieved_chunks: list[dict[str, Any]],
    ground_truth_anchors: list[dict[str, str]],
    k: int = 5,
) -> dict[str, float]:
    """
    Compute Precision@k and Recall@k for a single retrieval result.

    Args:
        retrieved_chunks:      Ordered list of chunk dicts from AI Search.
                               Each must have "accession" and "section" fields.
        ground_truth_anchors:  List of filing-coordinate dicts from the golden
                               claim, each with "accession" and "section".
        k:                     Cutoff rank.

    Returns:
        Dict with precision_at_k, recall_at_k, k, num_relevant_retrieved,
        num_ground_truth.
    """
    if not ground_truth_anchors:
        log.warning("compute_retrieval_metrics called with empty ground_truth_anchors")
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "k": k,
                "num_relevant_retrieved": 0, "num_ground_truth": 0}

    gt_keys = {
        _anchor_key(a["accession"], a["section"])
        for a in ground_truth_anchors
        if "accession" in a and "section" in a
    }

    top_k = retrieved_chunks[:k]
    retrieved_keys = [
        _anchor_key(c.get("accession", ""), c.get("section", ""))
        for c in top_k
    ]

    hits = sum(1 for key in retrieved_keys if key in gt_keys)

    precision = hits / k if k > 0 else 0.0
    recall = min(1.0, hits / len(gt_keys)) if gt_keys else 0.0

    return {
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "k": k,
        "num_relevant_retrieved": hits,
        "num_ground_truth": len(gt_keys),
    }


def compute_batch_retrieval_metrics(
    batch: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, float]:
    """
    Compute mean Precision@k and Recall@k over a batch of retrieval results.

    Args:
        batch: List of dicts, each with:
               - "retrieved_chunks":     list of chunk dicts
               - "ground_truth_anchors": list of anchor dicts
               Optional:
               - "claim_id": str (for logging)
        k:     Cutoff rank.

    Returns:
        Dict with mean_precision_at_k, mean_recall_at_k, k, num_samples.
    """
    if not batch:
        return {"mean_precision_at_k": 0.0, "mean_recall_at_k": 0.0,
                "k": k, "num_samples": 0}

    precisions, recalls = [], []
    for item in batch:
        result = compute_retrieval_metrics(
            retrieved_chunks=item.get("retrieved_chunks", []),
            ground_truth_anchors=item.get("ground_truth_anchors", []),
            k=k,
        )
        precisions.append(result["precision_at_k"])
        recalls.append(result["recall_at_k"])

    mean_p = round(sum(precisions) / len(precisions), 4)
    mean_r = round(sum(recalls) / len(recalls), 4)

    log.info(
        "Retrieval metrics @k=%d: precision=%.4f recall=%.4f (n=%d)",
        k, mean_p, mean_r, len(batch),
    )

    return {
        "mean_precision_at_k": mean_p,
        "mean_recall_at_k": mean_r,
        "k": k,
        "num_samples": len(batch),
    }