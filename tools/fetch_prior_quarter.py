from __future__ import annotations

import re
from typing import Optional

from tools.search_documents import search_documents

# Ordered fiscal quarter sequence — used for offset arithmetic
_QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]


def _parse_fiscal_label(label: str) -> tuple[int, int]:
    """
    Parse 'Q2_FY2025' → (quarter_index=1, fiscal_year=2025).
    Raises ValueError on bad format.
    """
    m = re.fullmatch(r"Q([1-4])_FY(\d{4})", label.strip())
    if not m:
        raise ValueError(f"Invalid fiscal label '{label}'. Expected format: Q1_FY2025")
    q_idx = int(m.group(1)) - 1   # 0-based index into _QUARTER_ORDER
    fy = int(m.group(2))
    return q_idx, fy


def _resolve_prior_label(current_quarter: str, quarters_back: int) -> str:
    """
    Subtract `quarters_back` from `current_quarter` and return the resolved label.

    Example: current='Q2_FY2025', quarters_back=3 → 'Q3_FY2024'
    """
    if quarters_back < 1:
        raise ValueError("quarters_back must be >= 1")

    q_idx, fy = _parse_fiscal_label(current_quarter)

    total_quarters = q_idx - quarters_back
    # Wrap backwards across fiscal years
    while total_quarters < 0:
        total_quarters += 4
        fy -= 1

    resolved_q = _QUARTER_ORDER[total_quarters % 4]
    return f"{resolved_q}_FY{fy}"


def fetch_prior_quarter(
    company: str,
    current_quarter: str,
    quarters_back: int,
    query: str,
    doc_type: Optional[str] = None,
    top: int = 5,
) -> dict:
    """
    Retrieve chunks from a prior quarter for the Comparison Agent.

    Args:
        company:         Ticker symbol e.g. 'AAPL'.
        current_quarter: Current fiscal label e.g. 'Q2_FY2025'.
        quarters_back:   How many quarters to step back (1–4 typical; max ~20 for 5-year history).
        query:           The passage or claim to search for in the prior quarter.
        doc_type:        Optional — '10-Q', '10-K', or 'transcript'.
        top:             Number of chunks to return.

    Returns:
        dict with resolved quarter label, results list, and count.
    """
    target_quarter = _resolve_prior_label(current_quarter, quarters_back)

    search_result = search_documents(
        query=query,
        doc_type=doc_type,
        company=company,
        quarter=target_quarter,
        top=top,
    )

    return {
        "company": company,
        "requested_quarter": target_quarter,
        "quarters_back": quarters_back,
        "results": search_result["results"],
        "count": search_result["count"],
    }