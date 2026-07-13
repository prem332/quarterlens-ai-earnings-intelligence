
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("edgar_downloader")

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

WANTED_FORMS = {"10-Q", "10-K"}
REQUEST_INTERVAL = 0.15  # seconds between requests (~6-7 req/s, under SEC's 10/s)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_session() -> requests.Session:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError(
            "SEC_USER_AGENT env var is required by SEC EDGAR, e.g. "
            "'QuarterLens research contact@example.com'"
        )
    s = requests.Session()
    s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
    return s


def _get(session: requests.Session, url: str) -> requests.Response:
    time.sleep(REQUEST_INTERVAL)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp


def derive_fiscal_label(report_date: date, fye_month: int) -> str:
    """Map a filing's period-end date to a 'FY{year}-Q{n}' label using the
    company's fiscal-year-end month. Handles non-calendar fiscal years
    (AAPL/MSFT/NVDA). reportDate remains the authoritative anchor downstream;
    this label is for human-readable matching against companies.yaml."""
    m = report_date.month
    quarter = ((m - fye_month - 1) % 12) // 3 + 1
    fy = report_date.year if m <= fye_month else report_date.year + 1
    return f"FY{fy}-Q{quarter}"


def verify_cik(session: requests.Session, ticker: str, cik: str) -> None:
    """Best-effort: warn if the hardcoded CIK doesn't match SEC's ticker map."""
    try:
        data = _get(session, TICKER_MAP_URL).json()
        by_ticker = {v["ticker"].upper(): f"{int(v['cik_str']):010d}" for v in data.values()}
        official = by_ticker.get(ticker.upper())
        if official and official != cik:
            log.warning("CIK mismatch for %s: config=%s official=%s", ticker, cik, official)
    except Exception as e:  # non-fatal
        log.warning("CIK verification skipped (%s)", e)


def iter_recent_filings(submissions: dict):
    """Yield dicts for each 'recent' filing from an EDGAR submissions payload."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    for i in range(len(forms)):
        yield {
            "form": recent["form"][i],
            "accession": recent["accessionNumber"][i],
            "filing_date": recent["filingDate"][i],
            "report_date": recent["reportDate"][i],
            "primary_doc": recent["primaryDocument"][i],
        }


def download_filing(session, cik: str, filing: dict, out_dir: Path, label: str) -> Path:
    acc_nodash = filing["accession"].replace("-", "")
    url = ARCHIVE_URL.format(
        cik_int=int(cik), acc_nodash=acc_nodash, doc=filing["primary_doc"]
    )
    ext = Path(filing["primary_doc"]).suffix or ".htm"
    dest = out_dir / f"{label}_{filing['form'].replace('-', '')}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = _get(session, url)
    dest.write_bytes(resp.content)
    return dest


def run(config_path: str, out_root: str, verify: bool = True) -> None:
    config = load_config(config_path)
    companies = config.get("companies", [])
    session = make_session()
    out_root_path = Path(out_root)
    manifest: list[dict] = []

    for company in companies:
        ticker = company["ticker"]
        cik = str(company["cik"]).zfill(10)
        fye_month = int(company["fiscal_year_end_month"])
        # {fiscal_label: expected_form} declared in companies.yaml
        wanted = {p["fiscal"]: p["form"] for p in company["periods"]}
        remaining = set(wanted)

        log.info("== %s (CIK %s): seeking %d periods ==", ticker, cik, len(wanted))
        if verify:
            verify_cik(session, ticker, cik)

        try:
            submissions = _get(session, SUBMISSIONS_URL.format(cik=cik)).json()
        except requests.HTTPError as e:
            log.error("Failed to fetch submissions for %s: %s", ticker, e)
            continue

        out_dir = out_root_path / ticker
        for filing in iter_recent_filings(submissions):
            if not remaining:
                break
            if filing["form"] not in WANTED_FORMS or not filing["report_date"]:
                continue
            rpt = datetime.strptime(filing["report_date"], "%Y-%m-%d").date()
            label = derive_fiscal_label(rpt, fye_month)
            if label not in remaining or wanted[label] != filing["form"]:
                continue

            try:
                path = download_filing(session, cik, filing, out_dir, label)
            except requests.HTTPError as e:
                log.error("Download failed for %s %s: %s", ticker, label, e)
                continue

            remaining.discard(label)
            manifest.append({
                "ticker": ticker,
                "cik": cik,
                "fiscal_label": label,
                "form": filing["form"],
                "report_date": filing["report_date"],
                "filing_date": filing["filing_date"],
                "accession": filing["accession"],
                "primary_document": filing["primary_doc"],
                "source_url": ARCHIVE_URL.format(
                    cik_int=int(cik),
                    acc_nodash=filing["accession"].replace("-", ""),
                    doc=filing["primary_doc"],
                ),
                "local_path": str(path),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            })
            log.info("  saved %s %s -> %s", label, filing["form"], path.name)

        if remaining:
            log.warning("  %s: not found/unfiled -> %s", ticker, sorted(remaining))

    manifest_path = out_root_path / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Done. %d filings downloaded. Manifest: %s", len(manifest), manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEC 10-Q/10-K filings for QuarterLens.")
    parser.add_argument("--config", default="golden_dataset/companies.yaml")
    parser.add_argument("--out", default=os.environ.get("QUARTERLENS_RAW_DIR", "data/raw"))
    parser.add_argument("--no-verify", action="store_true", help="skip CIK verification")
    args = parser.parse_args()
    run(args.config, args.out, verify=not args.no_verify)


if __name__ == "__main__":
    main()