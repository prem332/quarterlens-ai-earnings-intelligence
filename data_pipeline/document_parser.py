from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("document_parser")

# ---------------------------------------------------------------------------
# Form-aware canonical section maps
#
# 10-K: bare item key — Parts I-IV do NOT reset item numbering, no collision.
# 10-Q: (part, item) key — Part I and Part II both contain Items 1-4;
#        bare-item keying causes last-write-wins collision (BUG 1 fix).
#
# Financial statement tables (10-Q Part I Item 1, 10-K Item 8) → excluded.
# ---------------------------------------------------------------------------

SECTION_MAP_10K: dict[str, str] = {
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
}

SECTION_MAP_10Q: dict[tuple[str, str], str] = {
    ("I",  "1"):  "financial_statements",   # excluded
    ("I",  "2"):  "mda",
    ("I",  "3"):  "quantitative_qualitative_disclosures",
    ("I",  "4"):  "controls_and_procedures",
    ("II", "1"):  "legal_proceedings",
    ("II", "1A"): "risk_factors",
    ("II", "2"):  "unregistered_sales",
    ("II", "3"):  "defaults_senior_securities",
    ("II", "4"):  "mine_safety",
    ("II", "5"):  "other_information",
    ("II", "6"):  "exhibits",
}

_VALID_ITEMS_10Q: set[str] = {"1", "1A", "2", "3", "4", "5", "6"}

_EXCLUDED_CANONICAL: set[str] = {"financial_statements"}

# Dropped: boilerplate / low narrative value.
# legal_proceedings is intentionally KEPT — material for earnings analysis.
_ALWAYS_DROP: set[str] = {
    "exhibits",
    "reserved",
    "mine_safety",
    "unregistered_sales",
    "defaults_senior_securities",
}

_PART_NORM = {"I": "I", "II": "II", "III": "III", "IV": "IV",
              "1": "I", "2": "II", "3": "III", "4": "IV"}

# ---------------------------------------------------------------------------
# Title-text lookup table (Signal 3 — GOOGL / title-only TOC style)
#
# Some filers (e.g. GOOGL) use the section title as anchor text with opaque
# UUID hrefs — no "Item N" in text, no item slug in href. We match on lowered
# title substrings (longest/most-specific first to avoid prefix collisions).
#
# 10-Q maps title substring → (part, item)
# 10-K maps title substring → item_number_str
# ---------------------------------------------------------------------------

_TITLE_MAP_10Q: list[tuple[str, tuple[str, str]]] = [
    # Part markers
    ("part i",                                  ("_PART", "I")),
    ("part ii",                                 ("_PART", "II")),
    # Part I items (order: longest match first)
    ("management's discussion and analysis",    ("I",  "2")),
    ("quantitative and qualitative",            ("I",  "3")),
    ("controls and procedures",                 ("I",  "4")),
    ("financial statements",                    ("I",  "1")),   # excluded
    # Part II items
    ("risk factors",                            ("II", "1A")),
    ("legal proceedings",                       ("II", "1")),
    ("unregistered sales",                      ("II", "2")),
    ("other information",                       ("II", "5")),
    ("exhibits",                                ("II", "6")),
]

_TITLE_MAP_10K: list[tuple[str, str]] = [
    ("management's discussion and analysis",    "7"),
    ("quantitative and qualitative",            "7A"),
    ("controls and procedures",                 "9A"),
    ("financial statements",                    "8"),     # excluded
    ("risk factors",                            "1A"),
    ("unresolved staff comments",               "1B"),
    ("cybersecurity",                           "1C"),
    ("legal proceedings",                       "3"),
    ("properties",                              "2"),
    ("business",                                "1"),
    ("mine safety",                             "4"),
    ("market for registrant",                   "5"),
    ("reserved",                                "6"),
    ("changes in and disagreements",            "9"),
    ("other information",                       "9B"),
    ("disclosure regarding foreign",            "9C"),
    ("directors, executive officers",           "10"),
    ("executive compensation",                  "11"),
    ("security ownership",                      "12"),
    ("certain relationships",                   "13"),
    ("principal accountant fees",               "14"),
    ("exhibits",                                "15"),
]


def _normalize_title(text: str) -> str:
    """Lowercase and normalize Unicode quotes/dashes to ASCII equivalents."""
    return (
        text.lower()
        .replace("\u2019", "'")   # right single quotation mark → apostrophe
        .replace("\u2018", "'")   # left single quotation mark
        .replace("\u201c", '"')   # left double quotation mark
        .replace("\u201d", '"')   # right double quotation mark
        .replace("\u2013", "-")   # en-dash
        .replace("\u2014", "-")   # em-dash
    )


def _title_lookup_10q(text: str) -> tuple[str, str] | None:
    """Match lowered anchor text against 10-Q title map. Returns (part, item) or None."""
    t = _normalize_title(text)
    for substr, key in _TITLE_MAP_10Q:
        if substr in t:
            return key  # type: ignore[return-value]
    return None


def _title_lookup_10k(text: str) -> str | None:
    """Match lowered anchor text against 10-K title map. Returns item str or None."""
    t = _normalize_title(text)
    for substr, item in _TITLE_MAP_10K:
        if substr in t:
            return item
    return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _canonical(form: str, part: str, item: str) -> str | None:
    if form == "10-Q":
        return SECTION_MAP_10Q.get((part, item))
    return SECTION_MAP_10K.get(item)


def _dropped(canonical: str) -> bool:
    return canonical in _EXCLUDED_CANONICAL or canonical in _ALWAYS_DROP


def _ordinal(item: str) -> int:
    """Item ordinal for Part I→II reset detection: '1'→10, '1A'→11, '2'→20 …"""
    m = re.match(r"(\d+)([A-C]?)", item.upper())
    n = int(m.group(1))
    suf = m.group(2)
    return n * 10 + (ord(suf) - ord("A") + 1 if suf else 0)


# ---------------------------------------------------------------------------
# iXBRL / noise stripping
# ---------------------------------------------------------------------------

_IXBRL_TAGS = {"ix:hidden", "ix:header", "ix:references", "ix:resources"}


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag_name in _IXBRL_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()
    for el in soup.find_all(["script", "style"]):
        el.decompose()
    for el in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        el.decompose()


# ---------------------------------------------------------------------------
# TOC-anchor segmentation — three-signal item resolution
#
# Signal 1 (AAPL/NVDA/META style): "Item N" in anchor text
# Signal 2 (MSFT style):           item slug in href  (#item_2_managements…)
# Signal 3 (GOOGL style):          section title in anchor text, opaque UUID href
#
# Part context: explicit part marker (any signal) → ordinal-reset fallback.
# ---------------------------------------------------------------------------

_ITEM_TEXT_PAT = re.compile(r"item\s+(1[ABC]?|[2-9][ABC]?|1[0-5])\b", re.I)
_ITEM_SLUG_PAT = re.compile(r"item[_-](\d{1,2}[a-c]?)(?:[_-]|$)", re.I)
_PART_TEXT_PAT = re.compile(r"^\s*part\s+(i{1,3}|iv)\b", re.I)
_PART_SLUG_PAT = re.compile(r"part[_-](i{1,3}|iv|1|2|3|4)(?:[_-]|$)", re.I)


def _collect_toc_anchors(soup: BeautifulSoup, form: str) -> dict[tuple[str, str], Tag]:
    """
    Return {(part, item): target_element} for TOC item links in document order.
    For 10-K, part is always 'I' (placeholder — lookup uses bare item key).
    """
    valid_items = _VALID_ITEMS_10Q if form == "10-Q" else set(SECTION_MAP_10K)

    found: dict[tuple[str, str], Tag] = {}
    part = "I"
    prev_ord = 0

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("#"):
            continue
        text = a.get_text(" ", strip=True)
        aid = href[1:]

        # --- Part boundary (any signal) ---
        pm = _PART_TEXT_PAT.search(text) or _PART_SLUG_PAT.search(aid)
        if pm:
            part = _PART_NORM.get(pm.group(1).upper(), part)
            prev_ord = 0
            continue

        # --- Signal 1: "Item N" in anchor text ---
        item: str | None = None
        m = _ITEM_TEXT_PAT.search(text)
        if m:
            item = m.group(1).upper()

        # --- Signal 2: item slug in href ---
        if item is None:
            sm = _ITEM_SLUG_PAT.search(aid)
            if sm:
                item = sm.group(1).upper()

        # --- Signal 3: title-text lookup ---
        if item is None:
            if form == "10-Q":
                key = _title_lookup_10q(text)
                if key is not None:
                    if key[0] == "_PART":
                        # Part marker found via title table
                        part = _PART_NORM.get(key[1], part)
                        prev_ord = 0
                        continue
                    part_override, item = key
                    # Title lookup carries its own part — use it directly,
                    # bypassing ordinal-reset logic below.
                    target = soup.find(id=aid) or soup.find(attrs={"name": aid})
                    if target is not None and item in valid_items:
                        found[(part_override, item)] = target
                    continue
            else:
                item = _title_lookup_10k(text)

        if item is None or item not in valid_items:
            continue

        # --- Ordinal-reset Part I→II detection (10-Q, signals 1 & 2) ---
        if form == "10-Q":
            o = _ordinal(item)
            if o <= prev_ord and part == "I":
                part = "II"
            prev_ord = o

        target = soup.find(id=aid) or soup.find(attrs={"name": aid})
        if target is not None:
            found[(part, item)] = target

    return found


def _text_between(start: Tag, end: Tag | None) -> str:
    """
    Collect leaf text in document order between start and end (exclusive).
    Walks next_elements (full-tree) and collects only NavigableString leaves
    to avoid double-counting nested containers.
    """
    parts: list[str] = []
    for el in start.next_elements:
        if el is end:
            break
        if isinstance(el, NavigableString):
            s = str(el).strip()
            if s:
                parts.append(s)
    return " ".join(parts)


def _toc_segment(soup: BeautifulSoup, form: str) -> dict[str, str] | None:
    """TOC-anchor segmentation. Returns {canonical: text} or None if <3 anchors."""
    anchors = _collect_toc_anchors(soup, form)
    if len(anchors) < 3:
        return None

    pos = {id(t): i for i, t in enumerate(soup.find_all(True))}
    ordered = sorted(anchors.items(), key=lambda kv: pos.get(id(kv[1]), 1 << 30))

    sections: dict[str, str] = {}
    for i, ((part, item), tgt) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else None
        canonical = _canonical(form, part, item)
        if canonical is None or _dropped(canonical):
            continue
        text = _text_between(tgt, end).strip()
        if text and canonical not in sections:
            sections[canonical] = text

    return sections or None


# ---------------------------------------------------------------------------
# Heading-regex fallback segmentation (Part-aware for 10-Q)
# ---------------------------------------------------------------------------

_HEADING_PAT = re.compile(r"^(?:item|part)\s+(\d{1,2}[A-Ca-c]?)[\.\s]", re.I)
_PART_HEADING_PAT = re.compile(r"^\s*part\s+(i{1,3}|iv)\b", re.I)


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


def _regex_segment(soup: BeautifulSoup, form: str) -> dict[str, str]:
    """
    Heading-regex fallback. Part-aware for 10-Q; 10-K behavior unchanged.
    Collapses consecutive duplicate item headings (running-header suppression).
    """
    body = soup.find("body") or soup

    part = "I"
    prev_ord = 0
    current_key: tuple[str, str] | None = None
    current_parts: list[str] = []
    sections: dict[str, str] = {}

    def flush() -> None:
        if current_key and current_parts:
            canonical = _canonical(form, current_key[0], current_key[1])
            if canonical and not _dropped(canonical):
                text = " ".join(current_parts).strip()
                if text and canonical not in sections:
                    sections[canonical] = text

    for el in body.find_all(True):
        if not _is_heading(el):
            continue
        text = el.get_text(" ", strip=True)

        pm = _PART_HEADING_PAT.match(text)
        if pm:
            part = _PART_NORM.get(pm.group(1).upper(), part)
            prev_ord = 0
            continue

        m = _HEADING_PAT.match(text)
        if not m:
            if current_key:
                content = el.get_text(" ", strip=True)
                if content:
                    current_parts.append(content)
            continue

        item = m.group(1).upper()
        valid = _VALID_ITEMS_10Q if form == "10-Q" else set(SECTION_MAP_10K)
        if item not in valid:
            continue

        if form == "10-Q":
            o = _ordinal(item)
            if o <= prev_ord and part == "I":
                part = "II"
            prev_ord = o

        new_key = (part, item)
        if new_key == current_key:   # running-header suppression
            continue

        flush()
        current_key = new_key
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

    sections = _toc_segment(soup, form)
    method = "toc"
    if sections is None:
        sections = _regex_segment(soup, form)
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