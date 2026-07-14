"""

Deterministic numeric verification for the Numeric Validation Agent.
Never uses LLM arithmetic — fetches typed facts from Azure SQL `financial_facts`
and computes a registered formula in Python.

financial_facts schema:
    cik, accession, company, fiscal_label, concept, value, unit, period_start, period_end

Supported formulas:
    yoy_growth      — (current - prior_year) / |prior_year| * 100
    qoq_growth      — (current - prior_quarter) / |prior_quarter| * 100
    gross_margin    — (revenue - cogs) / revenue * 100
    operating_margin — operating_income / revenue * 100
    net_margin      — net_income / revenue * 100
    ratio           — numerator_concept / denominator_concept

Tool signature (matches tool_registry.py):
    calculate_metric(
        formula,
        company,
        fiscal_label,
        concept,                        # primary concept (all single-concept formulas)
        prior_fiscal_label=None,        # required for yoy_growth / qoq_growth
        denominator_concept=None,       # required for ratio
        cogs_concept=None,              # required for gross_margin (default: CostOfRevenue)
        operating_income_concept=None,  # required for operating_margin (default: OperatingIncomeLoss)
        net_income_concept=None,        # required for net_margin (default: NetIncomeLoss)
    )

Returns:
    {
        "formula": str,
        "company": str,
        "fiscal_label": str,
        "concept": str,
        "result": float | None,
        "unit": str,
        "inputs": dict,      # raw values used in computation
        "error": str | None  # populated if a fact is missing or divide-by-zero
    }
"""

from __future__ import annotations

from typing import Optional

from azure_clients.sql_client import sql_client

# ---------------------------------------------------------------------------
# Default us-gaap concept names for common metrics
# ---------------------------------------------------------------------------
_DEFAULT_REVENUE_CONCEPT = "Revenues"          # or RevenueFromContractWithCustomerExcludingAssessedTax
_DEFAULT_COGS_CONCEPT = "CostOfRevenue"
_DEFAULT_OPERATING_INCOME_CONCEPT = "OperatingIncomeLoss"
_DEFAULT_NET_INCOME_CONCEPT = "NetIncomeLoss"

_SUPPORTED_FORMULAS = {
    "yoy_growth",
    "qoq_growth",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "ratio",
}


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def _fetch_value(company: str, fiscal_label: str, concept: str) -> tuple[Optional[float], Optional[str]]:
    """
    Fetch a single (value, unit) for a company + fiscal_label + concept.
    Returns (None, None) if not found.
    """
    query = """
        SELECT TOP 1 value, unit
        FROM financial_facts
        WHERE company = ?
          AND fiscal_label = ?
          AND concept = ?
        ORDER BY period_end DESC
    """
    rows = sql_client.execute_query(query, (company, fiscal_label, concept))
    if not rows:
        return None, None
    return float(rows[0]["value"]), rows[0].get("unit", "")


# ---------------------------------------------------------------------------
# Formula implementations
# ---------------------------------------------------------------------------

def _safe_pct_change(current: float, prior: float) -> Optional[float]:
    if prior == 0:
        return None
    return round((current - prior) / abs(prior) * 100, 4)


def _growth(company: str, current_label: str, prior_label: str, concept: str) -> dict:
    cur_val, unit = _fetch_value(company, current_label, concept)
    pri_val, _ = _fetch_value(company, prior_label, concept)

    inputs = {
        f"{current_label}_{concept}": cur_val,
        f"{prior_label}_{concept}": pri_val,
    }

    if cur_val is None:
        return {"result": None, "unit": unit or "", "inputs": inputs,
                "error": f"No fact found: {concept} / {current_label}"}
    if pri_val is None:
        return {"result": None, "unit": unit or "", "inputs": inputs,
                "error": f"No fact found: {concept} / {prior_label}"}
    if pri_val == 0:
        return {"result": None, "unit": "%", "inputs": inputs,
                "error": "Divide-by-zero: prior period value is 0"}

    return {"result": _safe_pct_change(cur_val, pri_val), "unit": "%", "inputs": inputs, "error": None}


def _margin(company: str, fiscal_label: str, numerator_concept: str, revenue_concept: str) -> dict:
    num_val, unit = _fetch_value(company, fiscal_label, numerator_concept)
    rev_val, _ = _fetch_value(company, fiscal_label, revenue_concept)

    inputs = {numerator_concept: num_val, revenue_concept: rev_val}

    if num_val is None:
        return {"result": None, "unit": "%", "inputs": inputs,
                "error": f"No fact found: {numerator_concept} / {fiscal_label}"}
    if rev_val is None:
        return {"result": None, "unit": "%", "inputs": inputs,
                "error": f"No fact found: {revenue_concept} / {fiscal_label}"}
    if rev_val == 0:
        return {"result": None, "unit": "%", "inputs": inputs,
                "error": "Divide-by-zero: revenue is 0"}

    return {"result": round(num_val / rev_val * 100, 4), "unit": "%", "inputs": inputs, "error": None}


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def calculate_metric(
    formula: str,
    company: str,
    fiscal_label: str,
    concept: str,
    prior_fiscal_label: Optional[str] = None,
    denominator_concept: Optional[str] = None,
    cogs_concept: Optional[str] = None,
    operating_income_concept: Optional[str] = None,
    net_income_concept: Optional[str] = None,
) -> dict:
    """
    Deterministically verify a numeric claim using filed financial_facts.

    Args:
        formula:                  One of the _SUPPORTED_FORMULAS.
        company:                  Ticker e.g. 'AAPL'.
        fiscal_label:             Primary period e.g. 'Q2_FY2025'.
        concept:                  Primary us-gaap concept e.g. 'Revenues'.
        prior_fiscal_label:       Required for yoy_growth / qoq_growth.
        denominator_concept:      Required for 'ratio'.
        cogs_concept:             For gross_margin (default: CostOfRevenue).
        operating_income_concept: For operating_margin (default: OperatingIncomeLoss).
        net_income_concept:       For net_margin (default: NetIncomeLoss).

    Returns:
        dict with 'result', 'unit', 'inputs', and 'error'.
    """
    base = {"formula": formula, "company": company, "fiscal_label": fiscal_label, "concept": concept}

    if formula not in _SUPPORTED_FORMULAS:
        return {**base, "result": None, "unit": "", "inputs": {},
                "error": f"Unknown formula '{formula}'. Supported: {sorted(_SUPPORTED_FORMULAS)}"}

    # --- yoy_growth / qoq_growth ---
    if formula in ("yoy_growth", "qoq_growth"):
        if not prior_fiscal_label:
            return {**base, "result": None, "unit": "%", "inputs": {},
                    "error": "prior_fiscal_label is required for growth formulas"}
        result = _growth(company, fiscal_label, prior_fiscal_label, concept)
        return {**base, **result}

    # --- gross_margin ---
    if formula == "gross_margin":
        cogs = cogs_concept or _DEFAULT_COGS_CONCEPT
        revenue = concept or _DEFAULT_REVENUE_CONCEPT
        cur_rev, unit = _fetch_value(company, fiscal_label, revenue)
        cur_cogs, _ = _fetch_value(company, fiscal_label, cogs)
        inputs = {revenue: cur_rev, cogs: cur_cogs}
        if cur_rev is None or cur_cogs is None:
            missing = revenue if cur_rev is None else cogs
            return {**base, "result": None, "unit": "%", "inputs": inputs,
                    "error": f"No fact found: {missing} / {fiscal_label}"}
        if cur_rev == 0:
            return {**base, "result": None, "unit": "%", "inputs": inputs,
                    "error": "Divide-by-zero: revenue is 0"}
        gross_profit = cur_rev - cur_cogs
        return {**base, "result": round(gross_profit / cur_rev * 100, 4), "unit": "%",
                "inputs": inputs, "error": None}

    # --- operating_margin ---
    if formula == "operating_margin":
        op_concept = operating_income_concept or _DEFAULT_OPERATING_INCOME_CONCEPT
        rev_concept = concept or _DEFAULT_REVENUE_CONCEPT
        result = _margin(company, fiscal_label, op_concept, rev_concept)
        return {**base, **result}

    # --- net_margin ---
    if formula == "net_margin":
        ni_concept = net_income_concept or _DEFAULT_NET_INCOME_CONCEPT
        rev_concept = concept or _DEFAULT_REVENUE_CONCEPT
        result = _margin(company, fiscal_label, ni_concept, rev_concept)
        return {**base, **result}

    # --- ratio ---
    if formula == "ratio":
        if not denominator_concept:
            return {**base, "result": None, "unit": "", "inputs": {},
                    "error": "denominator_concept is required for 'ratio'"}
        num_val, unit = _fetch_value(company, fiscal_label, concept)
        den_val, _ = _fetch_value(company, fiscal_label, denominator_concept)
        inputs = {concept: num_val, denominator_concept: den_val}
        if num_val is None:
            return {**base, "result": None, "unit": unit or "", "inputs": inputs,
                    "error": f"No fact found: {concept} / {fiscal_label}"}
        if den_val is None:
            return {**base, "result": None, "unit": unit or "", "inputs": inputs,
                    "error": f"No fact found: {denominator_concept} / {fiscal_label}"}
        if den_val == 0:
            return {**base, "result": None, "unit": "", "inputs": inputs,
                    "error": "Divide-by-zero: denominator is 0"}
        return {**base, "result": round(num_val / den_val, 6), "unit": f"{concept}/{denominator_concept}",
                "inputs": inputs, "error": None}

    # Should be unreachable given the guard above
    return {**base, "result": None, "unit": "", "inputs": {}, "error": "Unhandled formula path"}