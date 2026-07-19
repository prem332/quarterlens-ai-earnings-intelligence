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
    *_cc metrics                           → validated on reported (non-CC) basis;
                                             CC adjustment not in XBRL
    segment metrics                        → unsupported, clean error returned

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

import re

from azure_clients.sql_client import sql_client

log = logging.getLogger(__name__)


def _normalize_metric(metric: str) -> str:
    """
    Normalize metric name to snake_case for alias lookup.
    Handles: spaces, hyphens, parentheses, mixed case.
    Examples:
        "Revenue Growth CC"     → "revenue_growth_cc"
        "Revenue-Growth-CC"     → "revenue_growth_cc"
        "revenue growth (cc)"   → "revenue_growth_cc"
        "Gross Margin % Change" → "gross_margin_pct_change"
    """
    s = metric.lower().strip()
    s = s.replace("(cc)", "cc").replace("(c.c.)", "cc")
    s = re.sub(r"[%]", "pct", s)
    s = re.sub(r"[^\w\s]", " ", s)   # strip punctuation except underscore
    s = re.sub(r"\s+", "_", s)       # spaces → underscore
    s = re.sub(r"_+", "_", s)        # collapse multiple underscores
    return s.strip("_")

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
    "total_revenue_growth_yoy":             "Revenues",
    "total_revenue_yoy_growth":             "Revenues",
    "total_revenue_growth_yoy_constant_currency": "Revenues",
    "revenue_growth_cc":                    "Revenues",
    "revenue_yoy_growth":                   "Revenues",   # variant of revenue_growth_yoy
    # Sequential (QoQ) growth variants
    "revenue_growth_sequential":            "Revenues",
    "revenue_growth_sequential_absolute":   "Revenues",
    "revenue_growth_seq":                   "Revenues",
    "revenue_cc":                           "Revenues",
    "revenue_growth_yoy_cc":                "Revenues",
    # Cost
    "cost_of_revenue":                      "CostOfRevenue",
    "cogs":                                 "CostOfRevenue",
    "cost_of_goods_sold":                   "CostOfRevenue",
    # Gross profit / margin
    "gross_profit":                         "GrossProfit",
    "gross_margin":                         "GrossProfit",
    "gross_margin_dollars":                 "GrossProfit",
    "gross_margin_dollars_growth_cc":       "GrossProfit",   # YoY on reported GP
    "gross_margin_dollars_growth":          "GrossProfit",
    "gross_margin_pct_change":              "GrossProfit",   # margin pct change → margin formula
    "gross_margin_growth_cc":               "GrossProfit",
    # Operating income / margin
    "operating_income":                     "OperatingIncomeLoss",
    "operating_income_loss":                "OperatingIncomeLoss",
    "operating_margin":                     "OperatingIncomeLoss",
    "operating_income_growth_cc":           "OperatingIncomeLoss",  # YoY on reported OI
    "operating_income_growth":              "OperatingIncomeLoss",
    "operating_income_growth_yoy":          "OperatingIncomeLoss",
    # Operating expenses
    "operating_expenses":                   "OperatingExpenses",
    "operating_expenses_growth_cc":         "OperatingExpenses",    # YoY on reported OpEx
    "operating_expenses_growth":            "OperatingExpenses",
    "operating_expenses_yoy_growth":        "OperatingExpenses",    # variant
    # Gross margin QoQ change
    "gross_margin_qoq_change":              "GrossProfit",          # margin pct change QoQ
    # Net income
    "net_income":                           "NetIncomeLoss",
    "net_income_loss":                      "NetIncomeLoss",
    "net_margin":                           "NetIncomeLoss",
    "net_profit":                           "NetIncomeLoss",
    # EPS
    "eps":                                  "EarningsPerShareDiluted",
    "eps_diluted":                          "EarningsPerShareDiluted",
    "earnings_per_share":                   "EarningsPerShareDiluted",
    "earnings_per_share_diluted":           "EarningsPerShareDiluted",
    "eps_diluted_yoy_growth":               "EarningsPerShareDiluted",
    "eps_diluted_growth_yoy":               "EarningsPerShareDiluted",
    "eps_basic":                            "EarningsPerShareBasic",
    "earnings_per_share_basic":             "EarningsPerShareBasic",
    # Gross margin pct (alias for gross_margin margin formula)
    "gross_margin_pct":                     "GrossProfit",
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

# Metrics computed as (value / Revenues) * 100
_MARGIN_METRICS = {
    "gross_margin", "operating_margin", "net_margin", "gross_margin_pct",
}

# Metrics computed as YoY growth % on resolved concept
# Includes CC variants — validated on reported basis with note
_YOY_METRICS = {
    "revenue_growth_yoy", "revenue_yoy",
    "revenue_growth_cc", "revenue_cc", "revenue_growth_yoy_cc",
    "gross_margin_dollars_growth_cc", "gross_margin_dollars_growth",
    "gross_margin_growth_cc",
    "operating_income_growth_cc", "operating_income_growth",
    "operating_income_growth_yoy",
    "operating_expenses_growth_cc", "operating_expenses_growth",
}

# Metrics computed as QoQ growth % on resolved concept
_QOQ_METRICS = {
    "revenue_growth_qoq", "revenue_qoq",
    "revenue_growth_sequential", "revenue_growth_sequential_absolute", "revenue_growth_seq",
}

# Margin pct change — compute margin for current and prior, then diff
_MARGIN_CHANGE_METRICS = {
    "gross_margin_pct_change", "gross_margin_qoq_change",
}

# Categorized unsupported metrics — not in SEC XBRL financial_facts.
# Category drives the error message for cleaner evaluation output.
_UNSUPPORTED_METRIC_CATEGORIES: dict[str, str] = {
    # Segment revenue — Azure / MSFT
    "azure_other_cloud_services_revenue_growth_cc": "segment_kpi",
    "azure_other_cloud_services_revenue":           "segment_kpi",
    "on_premises_server_revenue_growth_cc":         "segment_kpi",
    "on_premises_server_revenue":                   "segment_kpi",
    "intelligent_cloud_revenue":                    "segment_kpi",
    "more_personal_computing_revenue":              "segment_kpi",
    "productivity_business_revenue":                "segment_kpi",
    "segment_gross_margin_dollars_growth_cc":       "segment_kpi",
    "segment_gross_margin_dollars":                 "segment_kpi",
    "segment_gross_margin":                         "segment_kpi",
    # Product revenue — AAPL
    "iphone_revenue":              "product_kpi",
    "mac_revenue":                 "product_kpi",
    "ipad_revenue":                "product_kpi",
    "services_revenue":            "product_kpi",
    "services_revenue_yoy_growth": "product_kpi",
    "services_revenue_growth_yoy": "product_kpi",
    "wearables_revenue":           "product_kpi",
    # Product / forward-looking KPIs — NVDA
    "blackwell_rubin_visible_revenue":    "product_kpi",
    "blackwell_revenue":                  "product_kpi",
    "hopper_revenue":                     "product_kpi",
    "annual_ai_infrastructure_build":     "operational_kpi",
    "ai_infrastructure_build":            "operational_kpi",
    "data_center_revenue":                "segment_kpi",
    "gaming_revenue":                     "segment_kpi",
    "professional_visualization_revenue": "segment_kpi",
    "automotive_revenue":                 "segment_kpi",
    # Segment revenue — GOOGL
    "search_advertising_revenue": "segment_kpi",
    "youtube_revenue":            "segment_kpi",
    "google_services_revenue":    "segment_kpi",
    "google_cloud_revenue":       "segment_kpi",
    # Operational KPIs — META
    "family_daily_active_people": "operational_kpi",
    "dap":                        "operational_kpi",
    # Operational KPIs — general
    "bookings":                              "operational_kpi",
    "arr":                                   "operational_kpi",
    "rpo":                                   "operational_kpi",
    "commercial_cloud_bookings":             "operational_kpi",
    "remaining_performance_obligation":      "operational_kpi",
    "office_365_seats":                      "operational_kpi",
    "github_users":                          "operational_kpi",
    "xbox_mau":                              "operational_kpi",
    "copilot_users":                         "operational_kpi",
}

_UNSUPPORTED_CATEGORY_MESSAGES = {
    "segment_kpi":       "Segment KPI not filed in SEC XBRL — requires MD&A table extraction",
    "product_kpi":       "Product-level KPI not filed in SEC XBRL — requires MD&A table extraction",
    "operational_kpi":   "Operational KPI not filed in SEC XBRL — not derivable from financial_facts",
    "derived_financial": "Derived financial metric not in XBRL — requires cash flow statement or non-GAAP calculation",
}


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def _fetch_value(
    ticker: str,
    fiscal_label: str,
    concept: str,
) -> tuple[Optional[float], Optional[str]]:
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
    """Map normalized metric name to SQL concept. Accepts already-normalized input."""
    return _CONCEPT_ALIASES.get(metric)


# Pattern-based unsupported category detection — catches long-tail LLM-extracted metrics
# that aren't worth enumerating individually in _UNSUPPORTED_METRIC_CATEGORIES.
_SEGMENT_PATTERNS = (
    "azure_", "intelligent_cloud", "more_personal_computing", "productivity_business",
    "windows_oem", "search_advertising", "gaming_revenue", "xbox_", "office_",
    "google_cloud", "google_services", "youtube_", "waymo_",
    "iphone_", "mac_", "ipad_", "wearables_", "services_gross_margin",
    "blackwell_", "hopper_", "data_center_revenue", "automotive_revenue",
    "professional_visualization", "reality_labs_",
)
_PRODUCT_PATTERNS = (
    "_yoy_growth", "_qoq_growth", "_sequential_growth",  # growth of segment = still segment
)
_OPERATIONAL_PATTERNS = (
    "pull_forward", "pull_ahead", "tariff_", "upgrade_record",
    "features_count", "infrastructure_build", "launch_impact",
    "employees", "employee_", "headcount",
    "primary_factors", "guidance_low", "guidance_high",
    "_guidance", "oire_guidance", "tax_rate_guidance",
)
_DERIVED_FINANCIAL_PATTERNS = (
    "free_cash_flow", "capital_expenditures", "capex",
    "share_repurchases", "buyback", "dividends_paid", "cash_dividend",
    "cash_and_marketable", "operating_cash_flow",
    "adjusted_net_income", "adjusted_eps", "net_income_excluding",
    "eps_diluted_excluding", "tax_rate", "interest_and_other",
    "total_expenses", "expense_growth", "debt",
    "gross_margin_pct",   # alias for gross_margin — add to aliases too
    "products_revenue", "products_gross_margin", "services_gross_margin",
    "company_gross_margin",
)


def _pattern_unsupported_category(metric: str) -> Optional[str]:
    """
    Pattern-based fallback for metrics not in _UNSUPPORTED_METRIC_CATEGORIES.
    Avoids enumerating every LLM-extracted long-tail variant individually.
    Returns category string or None if no pattern matches.
    """
    for pat in _SEGMENT_PATTERNS:
        if pat in metric:
            return "segment_kpi"
    for pat in _OPERATIONAL_PATTERNS:
        if pat in metric:
            return "operational_kpi"
    for pat in _DERIVED_FINANCIAL_PATTERNS:
        if pat in metric:
            return "derived_financial"
    return None


def _is_cc_metric(metric: str) -> bool:
    return metric.lower().strip().endswith("_cc")


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

    metric_lower = _normalize_metric(metric)

    # ── Segment / unsupported metrics — clean early exit ─────────────────
    # 1. Exact match in categorized dict
    category = _UNSUPPORTED_METRIC_CATEGORIES.get(metric_lower)
    if category:
        base["error"] = _UNSUPPORTED_CATEGORY_MESSAGES[category]
        log.info("Unsupported metric '%s' [%s] — returning clean error", metric, category)
        return base

    # 2. Pattern-based fallback — catches long-tail LLM-extracted variants
    category = _pattern_unsupported_category(metric_lower)
    if category:
        base["error"] = _UNSUPPORTED_CATEGORY_MESSAGES[category]
        log.info("Unsupported metric '%s' [%s, pattern match] — returning clean error", metric, category)
        return base

    concept = _resolve_concept(metric_lower)

    if concept is None:
        # Unknown metric — return clean error, no SQL attempt
        base["error"] = f"Unknown metric '{metric}' (normalized: '{metric_lower}') — no alias mapping found"
        log.warning("No alias for metric '%s' — no SQL lookup attempted", metric)
        return base

    base["concept"] = concept

    # ── CC metric note — validated on reported basis ──────────────────────
    cc_note = " (validated on reported basis; CC adjustment not in XBRL)" if _is_cc_metric(metric_lower) else ""

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

    # ── Margin pct change (e.g. gross_margin_pct_change) ─────────────────
    if metric_lower in _MARGIN_CHANGE_METRICS:
        if not prior_fiscal_label:
            base["error"] = "prior_fiscal_label required for margin pct change"
            return base

        cur_val,  _ = _fetch_value(company, fiscal_label,       concept)
        cur_rev,  _ = _fetch_value(company, fiscal_label,       "Revenues")
        pri_val,  _ = _fetch_value(company, prior_fiscal_label, concept)
        pri_rev,  _ = _fetch_value(company, prior_fiscal_label, "Revenues")

        base["inputs"] = {
            f"{fiscal_label}_{concept}": cur_val,
            f"{fiscal_label}_Revenues":  cur_rev,
            f"{prior_fiscal_label}_{concept}": pri_val,
            f"{prior_fiscal_label}_Revenues":  pri_rev,
        }

        if any(v is None for v in (cur_val, cur_rev, pri_val, pri_rev)):
            base["error"] = "Missing fact for margin pct change calculation"
            return base
        if cur_rev == 0 or pri_rev == 0:
            base["error"] = "Revenue is zero — cannot compute margin"
            return base

        cur_margin = cur_val / cur_rev * 100
        pri_margin = pri_val / pri_rev * 100
        base["value"] = round(cur_margin - pri_margin, 4)
        base["unit"] = "pp"   # percentage points
        return base

    # ── YoY growth ────────────────────────────────────────────────────────
    if metric_lower in _YOY_METRICS or ("yoy" in metric_lower and "cc" not in metric_lower) or metric_lower in _YOY_METRICS:
        if not prior_fiscal_label:
            base["error"] = "prior_fiscal_label required for YoY growth"
            return base
        cur_val, unit = _fetch_value(company, fiscal_label, concept)
        pri_val, _    = _fetch_value(company, prior_fiscal_label, concept)
        base["inputs"] = {fiscal_label: cur_val, prior_fiscal_label: pri_val}

        if cur_val is None or pri_val is None:
            base["error"] = "Missing fact for growth calculation"
            return base
        if pri_val == 0:
            base["error"] = "Prior period value is zero"
            return base

        base["value"] = round((cur_val - pri_val) / abs(pri_val) * 100, 4)
        base["unit"] = "%" + cc_note
        return base

    # ── CC growth — same as YoY, noted ───────────────────────────────────
    if _is_cc_metric(metric_lower):
        if not prior_fiscal_label:
            base["error"] = "prior_fiscal_label required for CC growth"
            return base
        cur_val, unit = _fetch_value(company, fiscal_label, concept)
        pri_val, _    = _fetch_value(company, prior_fiscal_label, concept)
        base["inputs"] = {fiscal_label: cur_val, prior_fiscal_label: pri_val}

        if cur_val is None or pri_val is None:
            base["error"] = "Missing fact for CC growth calculation"
            return base
        if pri_val == 0:
            base["error"] = "Prior period value is zero"
            return base

        base["value"] = round((cur_val - pri_val) / abs(pri_val) * 100, 4)
        base["unit"] = "%" + cc_note
        return base

    # ── QoQ growth ────────────────────────────────────────────────────────
    if metric_lower in _QOQ_METRICS or "qoq" in metric_lower:
        if not prior_fiscal_label:
            base["error"] = "prior_fiscal_label required for QoQ growth"
            return base
        cur_val, unit = _fetch_value(company, fiscal_label, concept)
        pri_val, _    = _fetch_value(company, prior_fiscal_label, concept)
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

    # ── Direct value lookup (default) ────────────────────────────────────
    val, unit = _fetch_value(company, fiscal_label, concept)
    base["inputs"] = {concept: val}

    if val is None:
        base["error"] = f"No fact found: {concept} / {fiscal_label}"
        return base

    base["value"] = round(float(val), 4)
    base["unit"] = unit or ""
    return base