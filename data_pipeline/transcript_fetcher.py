"""Earnings-call transcript ingestion with a pluggable provider interface.

Why pluggable: transcript sources (FMP, etc.) are paid, key-gated, and prone to
lock-in — unlike free SEC EDGAR. Isolating the source behind TranscriptProvider
means swapping vendors touches one class, not the pipeline. A MockProvider lets
the loop/mapping/manifest logic run and be tested with no key and no network.

Fiscal mapping: FMP keys transcripts by the company's *fiscal* quarter numbering
(e.g. Apple's "Q1 FY2026" call -> year=2026, quarter=1), which is exactly what the
fiscal labels in companies.yaml already encode. So the label maps directly for FMP.
That assumption lives inside FMPProvider; a calendar-keyed provider would convert
using fiscal_year_end_month (helper stub noted below).

Output: data/raw/transcripts/{TICKER}/{fiscal_label}.json + a transcripts manifest,
parallel to the filings manifest produced by edgar_downloader.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("transcript_fetcher")

_LABEL_RE = re.compile(r"^FY(\d{4})-Q([1-4])$")


# --------------------------------------------------------------------------- #
# Data contracts
# --------------------------------------------------------------------------- #
@dataclass
class TranscriptRequest:
    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    fiscal_label: str
    fiscal_year_end_month: int  # for providers that need calendar conversion


@dataclass
class TranscriptResult:
    text: str
    call_date: Optional[str] = None      # ISO date of the call, if the source gives one
    metadata: dict = field(default_factory=dict)


def parse_fiscal_label(label: str) -> tuple[int, int]:
    """'FY2026-Q1' -> (2026, 1)."""
    m = _LABEL_RE.match(label)
    if not m:
        raise ValueError(f"Bad fiscal label: {label!r} (expected 'FY####-Q#')")
    return int(m.group(1)), int(m.group(2))


# --------------------------------------------------------------------------- #
# Provider interface + implementations
# --------------------------------------------------------------------------- #
class TranscriptProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, req: TranscriptRequest) -> Optional[TranscriptResult]:
        """Return a TranscriptResult, or None if this source has no transcript
        for the requested period. Must not raise on 'not found'."""


class FMPProvider(TranscriptProvider):
    name = "fmp"
    BASE = "https://financialmodelingprep.com/api/v3/earning_call_transcript"
    REQUEST_INTERVAL = 0.3  # be gentle with free-tier rate limits

    def __init__(self) -> None:
        self.api_key = os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise RuntimeError("FMP_API_KEY env var is required for FMPProvider.")
        self.session = requests.Session()

    def fetch(self, req: TranscriptRequest) -> Optional[TranscriptResult]:
        # FMP uses fiscal quarter numbering -> pass fiscal year/quarter directly.
        url = f"{self.BASE}/{req.ticker}"
        params = {"year": req.fiscal_year, "quarter": req.fiscal_quarter, "apikey": self.api_key}
        time.sleep(self.REQUEST_INTERVAL)
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        # FMP returns a list; empty means no transcript for that period.
        if not payload:
            return None
        item = payload[0] if isinstance(payload, list) else payload
        text = (item.get("content") or "").strip()
        if not text:
            return None
        return TranscriptResult(
            text=text,
            call_date=item.get("date"),
            metadata={k: item.get(k) for k in ("symbol", "year", "quarter") if k in item},
        )


class MockProvider(TranscriptProvider):
    """Offline provider for tests/CI — deterministic stub, no key, no network."""
    name = "mock"

    def fetch(self, req: TranscriptRequest) -> Optional[TranscriptResult]:
        text = (
            f"[MOCK TRANSCRIPT] {req.ticker} {req.fiscal_label}\n"
            f"Operator: Welcome to the {req.ticker} Q{req.fiscal_quarter} "
            f"FY{req.fiscal_year} earnings call. This is placeholder content."
        )
        return TranscriptResult(text=text, call_date=None, metadata={"mock": True})


_PROVIDERS: dict[str, type[TranscriptProvider]] = {
    "fmp": FMPProvider,
    "mock": MockProvider,
}


def get_provider(name: str) -> TranscriptProvider:
    key = name.lower()
    if key not in _PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}. Available: {sorted(_PROVIDERS)}")
    return _PROVIDERS[key]()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(config_path: str, out_root: str, provider_name: str) -> None:
    config = load_config(config_path)
    provider = get_provider(provider_name)
    log.info("Using transcript provider: %s", provider.name)

    out_root_path = Path(out_root)
    manifest: list[dict] = []

    for company in config.get("companies", []):
        ticker = company["ticker"]
        fye_month = int(company["fiscal_year_end_month"])
        out_dir = out_root_path / ticker

        for period in company["periods"]:
            label = period["fiscal"]
            fy, fq = parse_fiscal_label(label)
            req = TranscriptRequest(ticker, fy, fq, label, fye_month)

            try:
                result = provider.fetch(req)
            except requests.HTTPError as e:
                log.error("  %s %s: fetch error %s", ticker, label, e)
                continue

            if result is None:
                log.warning("  %s %s: no transcript available", ticker, label)
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"{label}.json"
            record = {
                "ticker": ticker,
                "fiscal_label": label,
                "fiscal_year": fy,
                "fiscal_quarter": fq,
                "provider": provider.name,
                "call_date": result.call_date,
                "char_count": len(result.text),
                "metadata": result.metadata,
                "text": result.text,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            dest.write_text(json.dumps(record, indent=2), encoding="utf-8")

            manifest.append({k: record[k] for k in (
                "ticker", "fiscal_label", "provider", "call_date", "char_count", "fetched_at"
            )} | {"local_path": str(dest)})
            log.info("  saved %s %s (%d chars) -> %s", ticker, label, len(result.text), dest.name)

    manifest_path = out_root_path / "transcripts_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Done. %d transcripts fetched. Manifest: %s", len(manifest), manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch earnings-call transcripts for QuarterLens.")
    parser.add_argument("--config", default="golden_dataset/companies.yaml")
    parser.add_argument("--out", default=os.environ.get(
        "QUARTERLENS_TRANSCRIPT_DIR", "data/raw/transcripts"))
    parser.add_argument("--provider", default=os.environ.get("TRANSCRIPT_PROVIDER", "fmp"),
                        help="transcript provider: fmp | mock")
    args = parser.parse_args()
    run(args.config, args.out, args.provider)


if __name__ == "__main__":
    main()