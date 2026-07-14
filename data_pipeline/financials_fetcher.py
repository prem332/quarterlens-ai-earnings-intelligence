from __future__ import annotations

import argparse
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import json
import requests

from azure_clients.sql_client import SQLClient

logger = logging.getLogger(__name__)

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts"
# SEC requires a descriptive User-Agent or returns 403. Keep this consistent
# with edgar_downloader — set SEC_USER_AGENT in .env.
USER_AGENT = os.getenv("SEC_USER_AGENT", "QuarterLens AI research (contact@example.com)")
REQUEST_DELAY = 0.5   # seconds between per-CIK fetches; SEC allows <=10 req/s
FETCH_RETRIES = 3

# Duration windows (days) for selecting the reporting-period figure and
# excluding YTD/comparative durations. Apple's 52/53-week fiscal calendar means
# quarters run 13-14 weeks and years 364/371 days, so the windows are widened
# past the naive 90 / 365.
QUARTER_DAYS = (80, 100)
ANNUAL_DAYS = (350, 380)


@dataclass(frozen=True)
class Concept:
    name: str                 # canonical name -> stored in `concept`
    tags: tuple[str, ...]     # us-gaap alias tags, tried in priority order
    period_type: str          # "duration" | "instant"
    unit: str                 # companyfacts units key: USD | USD/shares | shares
    required: bool            # absence is a real bug vs. legitimate non-disclosure


# Curated ~16 concepts. Ordered alias lists absorb cross-company/cross-quarter
# tag drift (e.g. Apple tags revenue as RevenueFromContract..., others as Revenues).
CONCEPTS: tuple[Concept, ...] = (
    Concept("Revenues",
            ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"),
            "duration", "USD", required=True),
    Concept("CostOfRevenue",
            ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
            "duration", "USD", required=False),
    Concept("GrossProfit", ("GrossProfit",), "duration", "USD", required=False),
    Concept("OperatingIncomeLoss", ("OperatingIncomeLoss",), "duration", "USD", required=True),
    Concept("NetIncomeLoss", ("NetIncomeLoss",), "duration", "USD", required=True),
    Concept("ResearchAndDevelopmentExpense",
            ("ResearchAndDevelopmentExpense",), "duration", "USD", required=False),
    Concept("SellingGeneralAndAdministrativeExpense",
            ("SellingGeneralAndAdministrativeExpense",), "duration", "USD", required=False),
    Concept("OperatingExpenses",
            ("OperatingExpenses", "CostsAndExpenses"), "duration", "USD", required=False),
    Concept("EarningsPerShareBasic", ("EarningsPerShareBasic",),
            "duration", "USD/shares", required=True),
    Concept("EarningsPerShareDiluted", ("EarningsPerShareDiluted",),
            "duration", "USD/shares", required=True),
    Concept("WeightedAverageNumberOfSharesOutstandingBasic",
            ("WeightedAverageNumberOfSharesOutstandingBasic",),
            "duration", "shares", required=True),
    Concept("WeightedAverageNumberOfDilutedSharesOutstanding",
            ("WeightedAverageNumberOfDilutedSharesOutstanding",),
            "duration", "shares", required=True),
    Concept("CashAndCashEquivalentsAtCarryingValue",
            ("CashAndCashEquivalentsAtCarryingValue",), "instant", "USD", required=True),
    Concept("Assets", ("Assets",), "instant", "USD", required=True),
    Concept("Liabilities", ("Liabilities",), "instant", "USD", required=False),
    Concept("StockholdersEquity",
            ("StockholdersEquity",
             "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
            "instant", "USD", required=True),
)


def _norm_accn(s: str) -> str:
    return s.replace("-", "")


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("filings", data.get("documents", []))
    if not data:
        raise ValueError(f"No filings found in manifest: {path}")
    return list(data)


def fetch_companyfacts(cik: str, session: requests.Session) -> dict[str, Any]:
    url = f"{COMPANYFACTS_URL}/CIK{cik}.json"
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == FETCH_RETRIES:
                raise
            logger.warning("companyfacts fetch failed for %s (%s); retry %d/%d",
                           cik, exc, attempt, FETCH_RETRIES)
            time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


def _select(entries: list[dict], accession: str, form: str, period_type: str) -> dict | None:
    """Pick the reporting-period value for one filing: filter to this filing's
    accession, then choose the period-matched duration (or the latest instant),
    taking max end-date to prefer the current period over prior-year comparatives.
    """
    acc = _norm_accn(accession)
    cand = [e for e in entries if _norm_accn(e.get("accn", "")) == acc]
    if not cand:
        return None

    if period_type == "instant":
        inst = [e for e in cand if not e.get("start") and e.get("end")]
        return max(inst, key=lambda e: e["end"]) if inst else None

    lo, hi = QUARTER_DAYS if form.upper().startswith("10-Q") else ANNUAL_DAYS
    matched = []
    for e in cand:
        s, en = e.get("start"), e.get("end")
        if not s or not en:
            continue
        if lo <= (date.fromisoformat(en) - date.fromisoformat(s)).days <= hi:
            matched.append(e)
    return max(matched, key=lambda e: e["end"]) if matched else None


def extract_filing_facts(cf: dict, filing: dict) -> tuple[list[dict], list[str]]:
    ticker = filing.get("ticker") or filing.get("symbol") or filing.get("company")
    cik = str(filing["cik"]).zfill(10)
    accession = filing["accession"]
    form = filing["form"]
    fiscal_label = filing["fiscal_label"]
    gaap = cf.get("facts", {}).get("us-gaap", {})

    rows: list[dict] = []
    missing: list[str] = []
    for c in CONCEPTS:
        picked = None
        for tag in c.tags:
            entries = gaap.get(tag, {}).get("units", {}).get(c.unit)
            if not entries:
                continue
            e = _select(entries, accession, form, c.period_type)
            if e:
                picked = (tag, e)
                break
        if picked is None:
            missing.append(c.name)
            continue
        tag, e = picked
        rows.append({
            "ticker": ticker,
            "cik": cik,
            "accession": accession,
            "form": form,
            "fiscal_label": fiscal_label,
            "concept": c.name,
            "xbrl_tag": tag,
            "value": Decimal(str(e["val"])),
            "unit": c.unit,
            "period_start": date.fromisoformat(e["start"]) if e.get("start") else None,
            "period_end": date.fromisoformat(e["end"]),
            "fy": e.get("fy"),
            "fp": e.get("fp"),
        })
    return rows, missing


def _report_coverage(client: SQLClient, manifest: list[dict]) -> bool:
    """Warn on any missing concept; escalate if a REQUIRED concept is absent.
    Non-fatal by design: optional concepts (GrossProfit, Liabilities, ...) are
    legitimately not tagged by every filer, so a hard assert-all would be wrong.
    Returns True if all required concepts are present for all filings.
    """
    required = {c.name for c in CONCEPTS if c.required}
    all_names = {c.name for c in CONCEPTS}
    ok = True
    for f in manifest:
        cik = str(f["cik"]).zfill(10)
        present = {r["concept"] for r in client.fetch_facts(cik, f["fiscal_label"])}
        missing = all_names - present
        if not missing:
            continue
        missing_required = missing & required
        label = f"{f.get('ticker')} {f['fiscal_label']}"
        if missing_required:
            ok = False
            logger.error("COVERAGE — %s missing REQUIRED: %s", label, ", ".join(sorted(missing_required)))
        optional_missing = missing - required
        if optional_missing:
            logger.warning("COVERAGE — %s missing optional: %s", label, ", ".join(sorted(optional_missing)))
    if ok:
        logger.info("Coverage OK: all required concepts present for all %d filings", len(manifest))
    return ok


def run(manifest_path: str | Path, schema_path: str | Path, apply_schema: bool = True) -> None:
    manifest = load_manifest(manifest_path)
    client = SQLClient()
    if apply_schema:
        client.apply_schema(schema_path)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})

    by_cik: dict[str, list[dict]] = defaultdict(list)
    for f in manifest:
        by_cik[str(f["cik"]).zfill(10)].append(f)

    total = 0
    for cik, filings in by_cik.items():
        cf = fetch_companyfacts(cik, session)
        rows: list[dict] = []
        for f in filings:
            frows, missing = extract_filing_facts(cf, f)
            rows.extend(frows)
            if missing:
                logger.debug("%s %s: unresolved %s", f.get("ticker"), f["fiscal_label"], missing)
        loaded = client.load_facts(rows)
        total += loaded
        logger.info("cik %s: loaded %d facts across %d filing(s)", cik, loaded, len(filings))
        time.sleep(REQUEST_DELAY)

    logger.info("Loaded %d facts across %d filings; verifying coverage...", total, len(manifest))
    _report_coverage(client, manifest)
    logger.info("Table row count: %d", client.count())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Load us-gaap numeric facts into financial_facts.")
    ap.add_argument("--manifest", default="data/manifest.json",
                    help="Path to edgar_downloader's manifest.json")
    ap.add_argument("--schema", default="data_pipeline/financial_facts.sql")
    ap.add_argument("--no-apply-schema", action="store_true",
                    help="Skip schema apply (already created)")
    args = ap.parse_args()
    run(args.manifest, args.schema, apply_schema=not args.no_apply_schema)


if __name__ == "__main__":
    main()