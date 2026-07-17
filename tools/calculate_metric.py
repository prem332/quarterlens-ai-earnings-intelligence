"""
tools/calculate_metric.py

Deterministic numeric verification for the Numeric Validation Agent.
Never uses LLM arithmetic — fetches typed facts from Azure SQL financial_facts
and computes a registered formula in Python.

financial_facts schema:
    ticker, cik, accession, form, fiscal_label,
    concept, xbrl_tag, value, unit, period_start, period_end, fy, fp

Supported formulas (auto-detected from metric name via alias map):
    revenue / revenues / total_revenue     → fetch Revenues
    gross_profit / gross_margin            → GrossProfit or (Revenues - CostOfRevenue) / Revenues
    operating_income / operating_margin    → OperatingIncomeLoss (margin = / Revenues)
    net_income / net_margin                → NetIncomeLoss (margin = / Revenues)
    eps / eps_diluted                      → EarningsPerShareDiluted
    eps_basic                              → EarningsPerShareBasic
    r_and_d / research_and_development     → ResearchAndDevelopmentExpense
    sga                                    → SellingGeneralAndAdministrativeExpense
    cash                                   → CashAndCashEquivalentsAtCarryingValue
    total_assets / assets                  → Assets
    total_liabilities / liabilities        → Liabilities
    stockholders_equity / equity           → StockholdersEquity
    yoy_growth / qoq_growth                → growth formula on resolved concept

Tool signature (called by numeric_validation_agent):
    calculate_metric(company, fiscal_label, metric, prior_fiscal_label=None)

Returns:
    {
        "metric":        str,
        "company":       str,
        "fiscal_label":  str,
        "concept":       str,   # resolved SQL concept name
        "value":         float | None,
        "unit":          str,
        "inputs":        dict,
        "error":         str | None
    }
"""

from __future__ import annotations

import logging
from typing import Optional

from azure_clients.sql_client import sql_client

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alias map: LLM-extracted metric name → SQL concept name
# ---------------------------------------------------------------------------
_CONCEPT_ALIASES: dict[str, str] = {
    # Revenue
    "revenue":                              "Revenues",
    "revenues":                             "Revenues",
    "total_revenue":                        "Revenues",
    "net_revenue":                          "Revenues",
    "revenue_growth_yoy":                   "Revenues",
    "revenue_growth_qoq":                   "Revenues",
    "revenue_yoy":                          "Revenues",
    "revenue_qoq":                          "Revenues",
    # Cost
    "cost_of_revenue":                      "CostOfRevenue",
    "cogs":                                 "CostOfRevenue",
    "cost_of_goods_sold":                   "CostOfRevenue",
    # Gross
    "gross_profit":                         "GrossProfit",
    "gross_margin":                         "GrossProfit",   # margin computed separately
    # Operating
    "operating_income":                     "OperatingIncomeLoss",
    "operating_income_loss":                "OperatingIncomeLoss",
    "operating_margin":                     "OperatingIncomeLoss",  # margin computed separately
    "operating_expenses":                   "OperatingExpenses",
    # Net income
    "net_income":                           "NetIncomeLoss",
    "net_income_loss":                      "NetIncomeLoss",
    "net_margin":                           "NetIncomeLoss",  # margin computed separately
    "net_profit":                           "NetIncomeLoss",
    # EPS
    "eps":                                  "EarningsPerShareDiluted",
    "eps_diluted":                          "EarningsPerShareDiluted",
    "earnings_per_share":                   "EarningsPerShareDiluted",
    "earnings_per_share_diluted":           "EarningsPerShareDiluted",
    "eps_basic":                            "EarningsPerShareBasic",
    "earnings_per_share_basic":             "EarningsPerShareBasic",
    # R&D
    "r_and_d":                              "ResearchAndDevelopmentExpense",
    "research_and_development":             "ResearchAndDevelopmentExpense",
    "rd_expense":                           "ResearchAndDevelopmentExpense",
    # SG&A
    "sga":                                  "SellingGeneralAndAdministrativeExpense",
    "selling_general_administrative":       "SellingGeneralAndAdministrativeExpense",
    "sg_and_a":                             "SellingGeneralAndAdministrativeExpense",
    # Balance sheet
    "cash":                                 "CashAndCashEquivalentsAtCarryingValue",
    "cash_and_equivalents":                 "CashAndCashEquivalentsAtCarryingValue",
    "total_assets":                         "Assets",
    "assets":                               "Assets",
    "total_liabilities":                    "Liabilities",
    "liabilities":                          "Liabilities",
    "stockholders_equity":                  "StockholdersEquity",
    "equity":                               "StockholdersEquity",
    "shareholders_equity":                  "StockholdersEquity",
    # Shares
    "shares_outstanding":                   "WeightedAverageNumberOfSharesOutstandingBasic",
    "diluted_shares":                       "WeightedAverageNumberOfDilutedSharesOutstanding",
}

# Metrics that represent margin (%) — value / Revenues * 100
_MARGIN_METRICS = {
    "gross_margin", "operating_margin", "net_margin",
}

# Metrics that represent YoY growth
_YOY_METRICS = {
    "revenue_growth_yoy", "revenue_yoy",
}

# Metrics that represent QoQ growth
_QOQ_METRICS = {
    "revenue_growth_qoq", "revenue_qoq",
}


# ---------------------------------------------------------------------------
# SQL helpers — use 'ticker' column (not 'company')
# ---------------------------------------------------------------------------

def _fetch_value(
    ticker: str,
    fiscal_label: str,
    concept: str,
) -> tuple[Optional[float], Optional[str]]:
    """
    Fetch a single (value, unit) for a ticker + fiscal_label + concept.
    Uses 'ticker' column — matches financial_facts schema.
    Returns (None, None) if not found.
    """
    sql = """
        SELECT TOP 1 value, unit
        FROM dbo.financial_facts
        WHERE ticker = ?
          AND fiscal_label = ?
          AND concept = ?
        ORDER BY period_end DESC
    """
    try:
        with sql_client.connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, ticker, fiscal_label, concept)
            row = cur.fetchone()
            if row is None:
                return None, None
            return float(row[0]), row[1]
    except Exception as exc:
        log.warning("SQL fetch failed for %s/%s/%s: %s", ticker, fiscal_label, concept, exc)
        return None, None


def _resolve_concept(metric: str) -> Optional[str]:
    """Map LLM-extracted metric name to SQL concept. Case-insensitive."""
    return _CONCEPT_ALIASES.get(metric.lower().strip())


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def calculate_metric(
    company: str,
    fiscal_label: str,
    metric: str,
    prior_fiscal_label: Optional[str] = None,
) -> dict:
    """
    Deterministically verify a numeric claim using filed financial_facts.

    Args:
        company:             Ticker e.g. 'AAPL'.
        fiscal_label:        Period e.g. 'FY2025-Q3'.
        metric:              LLM-extracted metric name e.g. 'revenue_growth_yoy'.
        prior_fiscal_label:  Prior period for growth calculations (optional).

    Returns:
        dict with 'value', 'unit', 'concept', 'inputs', 'error'.
    """
    base = {
        "metric": metric,
        "company": company,
        "fiscal_label": fiscal_label,
        "concept": None,
        "value": None,
        "unit": "",
        "inputs": {},
        "error": None,
    }

    metric_lower = metric.lower().strip()
    concept = _resolve_concept(metric_lower)

    if concept is None:
        # Try treating the metric as a direct concept name
        concept = metric
        log.warning("No alias for metric '%s' — trying as direct concept name", metric)

    base["concept"] = concept

    # ── Margin metrics ────────────────────────────────────────────────────
    if metric_lower in _MARGIN_METRICS:
        val, unit = _fetch_value(company, fiscal_label, concept)
        rev_val, _ = _fetch_value(company, fiscal_label, "Revenues")
        base["inputs"] = {concept: val, "Revenues": rev_val}

        if val is None:
            base["error"] = f"No fact found: {concept} / {fiscal_label}"
            return base
        if rev_val is None or rev_val == 0:
            base["error"] = "No revenue fact found or revenue is zero"
            return base

        base["value"] = round(val / rev_val * 100, 4)
        base["unit"] = "%"
        return base

    # ── YoY growth ────────────────────────────────────────────────────────
    if metric_lower in _YOY_METRICS or "yoy" in metric_lower:
        if not prior_fiscal_label:
            base["error"] = "prior_fiscal_label required for YoY growth"
            return base
        cur_val, unit = _fetch_value(company, fiscal_label, concept)
        pri_val, _ = _fetch_value(company, prior_fiscal_label, concept)
        base["inputs"] = {fiscal_label: cur_val, prior_fiscal_label: pri_val}

        if cur_val is None or pri_val is None:
            base["error"] = f"Missing fact for growth calculation"
            return base
        if pri_val == 0:
            base["error"] = "Prior period value is zero"
            return base

        base["value"] = round((cur_val - pri_val) / abs(pri_val) * 100, 4)
        base["unit"] = "%"
        return base

    # ── QoQ growth ────────────────────────────────────────────────────────
    if metric_lower in _QOQ_METRICS or "qoq" in metric_lower:
        if not prior_fiscal_label:
            base["error"] = "prior_fiscal_label required for QoQ growth"
            return base
        cur_val, unit = _fetch_value(company, fiscal_label, concept)
        pri_val, _ = _fetch_value(company, prior_fiscal_label, concept)
        base["inputs"] = {fiscal_label: cur_val, prior_fiscal_label: pri_val}

        if cur_val is None or pri_val is None:
            base["error"] = "Missing fact for growth calculation"
            return base
        if pri_val == 0:
            base["error"] = "Prior period value is zero"
            return base

        base["value"] = round((cur_val - pri_val) / abs(pri_val) * 100, 4)
        base["unit"] = "%"
        return base

    # ── Direct value lookup (default) ─────────────────────────────────────
    val, unit = _fetch_value(company, fiscal_label, concept)
    base["inputs"] = {concept: val}

    if val is None:
        base["error"] = f"No fact found: {concept} / {fiscal_label}"
        return base

    base["value"] = round(float(val), 4)
    base["unit"] = unit or ""
    return base