from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("document_parser")

# ---------------------------------------------------------------------------
# Form-aware canonical section maps
# key   : item_number_str  (e.g. "1A", "7")
# value : canonical section name emitted in output JSON
#
# Item 1 (10-Q) and Item 8 (10-K) are financial statement tables — excluded.
# ---------------------------------------------------------------------------

SECTION_MAP: dict[str, dict[str, str]] = {
    "10-Q": {
        "1":  "financial_statements",   # excluded
        "1A": "risk_factors",
        "1B": "unresolved_staff_comments",
        "2":  "mda",
        "3":  "quantitative_qualitative_disclosures",
        "4":  "controls_and_procedures",
        "5":  "other_information",
        "6":  "exhibits",
    },
    "10-K": {
        "1":   "business",
        "1A":  "risk_factors",
        "1B":  "unresolved_staff_comments",
        "1C":  "cybersecurity",
        "2":   "properties",
        "3":   "legal_proceedings",
        "4":   "mine_safety",
        "5":   "market_for_registrant_equity",
        "6":   "reserved",
        "7":   "mda",
        "7A":  "quantitative_qualitative_disclosures",
        "8":   "financial_statements",   # excluded
        "9":   "changes_in_disagreements_accountants",
        "9A":  "controls_and_procedures",
        "9B":  "other_information",
        "9C":  "disclosure_re_foreign_jurisdictions",
        "10":  "directors_executive_officers",
        "11":  "executive_compensation",
        "12":  "security_ownership",
        "13":  "certain_relationships",
        "14":  "principal_accountant_fees",
        "15":  "exhibits",
    },
}

# Items whose content is raw financial statement tables — excluded from corpus.
EXCLUDED_ITEMS_BY_FORM: dict[str, set[str]] = {
    "10-Q": {"1"},
    "10-K": {"8"},
}

# Canonical names with low/no narrative value — dropped silently.
ALWAYS_DROP: set[str] = {"exhibits", "reserved", "mine_safety"}

# ---------------------------------------------------------------------------
# iXBRL / noise stripping
# ---------------------------------------------------------------------------

_IXBRL_TAGS = {"ix:hidden", "ix:header", "ix:references", "ix:resources"}


def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove iXBRL hidden blocks, scripts, styles, and display:none elements."""
    for tag_name in _IXBRL_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()
    for el in soup.find_all(["script", "style"]):
        el.decompose()
    for el in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        el.decompose()


# ---------------------------------------------------------------------------
# TOC-anchor segmentation
# ---------------------------------------------------------------------------

def _collect_toc_anchors(soup: BeautifulSoup, form: str) -> dict[str, Tag]:
    """
    Scan <a href="#..."> links for Item references.
    Returns {item_number: target_element} for items in SECTION_MAP[form].
    """
    section_map = SECTION_MAP.get(form, {})
    item_pat = re.compile(r"item\s+(1[ABC]?|[2-9][ABC]?|1[0-5])\b", re.I)
    found: dict[str, Tag] = {}

    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        m = item_pat.search(text)
        if not m:
            continue
        item_num = m.group(1).upper()
        if item_num not in section_map:
            continue
        href = a["href"]
        if not href.startswith("#"):
            continue
        anchor_id = href[1:]
        target = soup.find(id=anchor_id) or soup.find(attrs={"name": anchor_id})
        if target:
            found[item_num] = target

    return found


def _text_between(start: Tag, end: Tag | None) -> str:
    """Collect text from DOM siblings between start and end (exclusive)."""
    parts: list[str] = []
    node = start.next_sibling
    while node is not None and node != end:
        if isinstance(node, Tag):
            parts.append(node.get_text(" ", strip=True))
        else:
            t = str(node).strip()
            if t:
                parts.append(t)
        node = node.next_sibling
    return " ".join(parts)


def _toc_segment(
    soup: BeautifulSoup, form: str, excluded: set[str]
) -> dict[str, str] | None:
    """
    TOC-anchor segmentation.
    Returns {canonical_name: text} or None if fewer than 3 anchors resolved.
    """
    anchors = _collect_toc_anchors(soup, form)
    if len(anchors) < 3:
        return None

    section_map = SECTION_MAP[form]
    key_order = list(section_map.keys())
    ordered = sorted(anchors.keys(), key=lambda k: key_order.index(k) if k in key_order else 999)

    sections: dict[str, str] = {}
    for i, item_num in enumerate(ordered):
        canonical = section_map[item_num]
        if item_num in excluded or canonical in ALWAYS_DROP:
            continue
        end_tag = anchors[ordered[i + 1]] if i + 1 < len(ordered) else None
        text = _text_between(anchors[item_num], end_tag).strip()
        if text:
            sections[canonical] = text

    return sections or None


# ---------------------------------------------------------------------------
# Heading-regex fallback segmentation
# ---------------------------------------------------------------------------

_HEADING_PAT = re.compile(r"^(?:item|part)\s+(\d{1,2}[A-Ca-c]?)[\.\s]", re.I)


def _is_heading(tag: Tag) -> bool:
    if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return True
    text = tag.get_text(" ", strip=True)
    if len(text) > 300:
        return False
    if tag.find(["b", "strong"]):
        return True
    if text.isupper() and len(text) > 3:
        return True
    return False


def _regex_segment(
    soup: BeautifulSoup, form: str, excluded: set[str]
) -> dict[str, str]:
    """
    Heading-regex fallback: scan block elements for Item heading patterns,
    collect text until the next matched heading.
    """
    section_map = SECTION_MAP.get(form, {})
    body = soup.find("body") or soup

    current_item: str | None = None
    current_parts: list[str] = []
    sections: dict[str, str] = {}

    def flush() -> None:
        if current_item and current_parts:
            canonical = section_map.get(current_item.upper())
            if canonical and canonical not in ALWAYS_DROP and current_item.upper() not in excluded:
                text = " ".join(current_parts).strip()
                if text:
                    sections[canonical] = text

    for el in body.find_all(True):
        if not _is_heading(el):
            continue
        text = el.get_text(" ", strip=True)
        m = _HEADING_PAT.match(text)
        if not m:
            if current_item:
                content = el.get_text(" ", strip=True)
                if content:
                    current_parts.append(content)
            continue

        item_raw = m.group(1).upper()
        if item_raw not in section_map:
            continue

        flush()
        current_item = item_raw
        current_parts = []

    flush()
    return sections


# ---------------------------------------------------------------------------
# Public parse entry point
# ---------------------------------------------------------------------------

def parse_filing(
    htm_path: Path,
    form: str,
    fiscal_label: str,
    report_date: str,
    cik: str,
    accession: str,
) -> list[dict]:
    """
    Parse one primary .htm filing into a list of section dicts.

    Output schema per section:
        ticker, cik, fiscal_label, report_date, form, accession, section, text
    """
    ticker = htm_path.parent.name

    soup = BeautifulSoup(htm_path.read_bytes(), "lxml")
    _strip_noise(soup)

    excluded = EXCLUDED_ITEMS_BY_FORM.get(form, set())

    sections = _toc_segment(soup, form, excluded)
    method = "toc"
    if sections is None:
        sections = _regex_segment(soup, form, excluded)
        method = "regex"

    log.info(
        "  %s %s: %d sections via %s", ticker, fiscal_label, len(sections), method
    )

    return [
        {
            "ticker": ticker,
            "cik": cik,
            "fiscal_label": fiscal_label,
            "report_date": report_date,
            "form": form,
            "accession": accession,
            "section": section_name,
            "text": text,
        }
        for section_name, text in sections.items()
    ]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run(manifest_path: str, out_root: str) -> None:
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    out_root_p = Path(out_root)
    parsed_manifest: list[dict] = []

    for entry in manifest:
        local_path = Path(entry["local_path"])
        if not local_path.exists():
            log.warning("Missing file, skipping: %s", local_path)
            continue

        log.info("Parsing %s %s (%s)", entry["ticker"], entry["fiscal_label"], entry["form"])

        try:
            sections = parse_filing(
                htm_path=local_path,
                form=entry["form"],
                fiscal_label=entry["fiscal_label"],
                report_date=entry["report_date"],
                cik=entry["cik"],
                accession=entry["accession"],
            )
        except Exception as e:
            log.error("Failed %s %s: %s", entry["ticker"], entry["fiscal_label"], e)
            continue

        out_dir = out_root_p / entry["ticker"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{entry['fiscal_label']}_{entry['form'].replace('-', '')}.json"
        out_file.write_text(
            json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        parsed_manifest.append({
            **{k: entry[k] for k in
               ("ticker", "cik", "fiscal_label", "form", "report_date", "accession")},
            "section_count": len(sections),
            "parsed_path": str(out_file),
        })

    parsed_manifest_path = out_root_p / "parsed_manifest.json"
    parsed_manifest_path.write_text(json.dumps(parsed_manifest, indent=2), encoding="utf-8")
    log.info(
        "Done. %d/%d filings parsed. Manifest: %s",
        len(parsed_manifest), len(manifest), parsed_manifest_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse SEC filings for QuarterLens AI.")
    parser.add_argument(
        "--manifest",
        default=os.environ.get("QUARTERLENS_RAW_DIR", "data/raw") + "/manifest.json",
    )
    parser.add_argument("--out", default="data/parsed")
    args = parser.parse_args()
    run(args.manifest, args.out)


if __name__ == "__main__":
    main()