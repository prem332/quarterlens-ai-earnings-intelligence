"""
finetuning/prepare_dataset.py

Knowledge distillation dataset generation for report_agent fine-tuning.

Teacher: gpt-5.4-mini generates ideal analyst responses.
Student: gpt-4o-mini will be fine-tuned on these outputs (separate step).

Outputs:
  finetuning/training.jsonl   — 120 examples (80%), UTF-8 BOM
  finetuning/validation.jsonl — 30 examples (20%), UTF-8 BOM

Usage:
  python -m finetuning.prepare_dataset
  python -m finetuning.prepare_dataset --dry-run   # skip API calls, check structure only
"""

import json
import random
import time
import argparse
import logging
from pathlib import Path

from tools.search_documents import search_documents
from azure_clients.openai_client import openai_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Output paths ──────────────────────────────────────────────────────────────

_OUT_DIR = Path(__file__).parent
_TRAIN_PATH = _OUT_DIR / "training.jsonl"
_VAL_PATH = _OUT_DIR / "validation.jsonl"

# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_A = (
    "You are a financial analyst assistant. For simple factual questions about earnings filings, "
    "give a direct, concise answer in 1-2 sentences. Always cite the source with [FILING] or "
    "[TRANSCRIPT]. Never add unrequested information. Never hallucinate facts not in the evidence."
)

_SYSTEM_B = """\
You are a senior equity research analyst writing earnings intelligence briefings for institutional investors.

TONE:
- Professional, direct, assertive — not hedged or generic
- Use active voice: "Revenue grew 10%" not "Revenue was reported to have grown"
- No filler phrases: never use "it is worth noting", "importantly", "it should be mentioned"
- No financial advice, no buy/sell/hold recommendations

STRUCTURE (always follow this exact order):
## Executive Summary (2-3 sentences max — the single most important takeaway)
## Key Financial Metrics (bullet points, one metric per line)
## Guidance & Language Shifts (what changed vs prior quarter, be specific)
## Risk Factor Changes (what's new or dropped)
## Sentiment Overview (FinBERT-based, cite specific passages)
## Source Citations (list every [FILING] and [TRANSCRIPT] reference used)

FORMATTING RULES:
- Never write paragraphs longer than 3 sentences
- Use bullet points for lists of 3+ items
- Every number must have a unit: "$94.0B" not "94036"
- Every factual claim tagged: [FILING] or [TRANSCRIPT]
- No hallucinated facts — only state what's in the evidence
- Dates always in format: Q3 FY2025, not "third quarter of fiscal year 2025"

LENGTH: 600-800 words total. No more, no less."""

# ── Company/quarter matrix ────────────────────────────────────────────────────

_COMBOS: list[tuple[str, str]] = [
    ("AAPL", "FY2025-Q2"), ("AAPL", "FY2025-Q3"), ("AAPL", "FY2025-Q4"),
    ("AAPL", "FY2026-Q1"), ("AAPL", "FY2026-Q2"),
    ("MSFT", "FY2025-Q3"), ("MSFT", "FY2025-Q4"), ("MSFT", "FY2026-Q1"),
    ("MSFT", "FY2026-Q2"), ("MSFT", "FY2026-Q3"),
    ("NVDA", "FY2026-Q1"), ("NVDA", "FY2026-Q2"), ("NVDA", "FY2026-Q3"),
    ("NVDA", "FY2026-Q4"), ("NVDA", "FY2027-Q1"),
    ("GOOGL", "FY2025-Q1"), ("GOOGL", "FY2025-Q2"), ("GOOGL", "FY2025-Q3"),
    ("GOOGL", "FY2025-Q4"), ("GOOGL", "FY2026-Q1"),
    ("META", "FY2025-Q1"), ("META", "FY2025-Q2"), ("META", "FY2025-Q3"),
    ("META", "FY2025-Q4"), ("META", "FY2026-Q1"),
]

# ── Query templates ───────────────────────────────────────────────────────────

# Prompt A — 2 simple queries per combo → 50 examples
_QUERIES_A: list[str] = [
    "What was {company} revenue in {quarter}?",
    "What was {company} EPS in {quarter}?",
]

# Prompt B — 4 full analysis queries per combo → 100 examples
_QUERIES_B: list[str] = [
    "Analyze {company} earnings for {quarter} — provide a full intelligence briefing",
    "What are the key risk factors and changes for {company} in {quarter}?",
    "How did {company} management guidance language shift in {quarter} compared to prior quarter?",
    "Summarize {company} financial performance and sentiment in {quarter}",
]


# ── Evidence builders ─────────────────────────────────────────────────────────

def _fetch_evidence(query: str, company: str, quarter: str) -> list[dict]:
    """
    Retrieve top-5 chunks via full retrieval chain (MMR + rerank).
    Returns empty list on failure — caller skips the example.
    """
    try:
        result = search_documents(
            query=query,
            company=company,
            quarter=quarter,
            top=5,
            mmr=True,
            rerank=True,
            rerank_top_k=5,
            use_cache=True,
        )
        return result.get("results", [])
    except Exception as exc:
        logger.warning("search_documents failed [%s %s]: %s", company, quarter, exc)
        return []


def _build_user_A(company: str, quarter: str, chunks: list[dict], question: str) -> str:
    """User message for Prompt A: evidence + question."""
    chunk_lines = "\n".join(
        f"[{c.get('doc_type', 'FILING').upper()}] {c.get('content', '')[:400]}"
        for c in chunks
    )
    return (
        f"COMPANY: {company}\nQUARTER: {quarter}\n\n"
        f"EVIDENCE:\n{chunk_lines}\n\n"
        f"Question: {question}"
    )


def _build_user_B(company: str, quarter: str, chunks: list[dict], question: str) -> str:
    """
    User message for Prompt B: mirrors _build_draft_prompt() structure from report_agent.py.
    Language shift, sentiment, and numeric sections use placeholder text when data is
    not available outside the full pipeline — model is instructed to use evidence only.
    """
    chunk_text = "\n\n".join(
        f"[{c.get('doc_type', 'FILING').upper()}] {c.get('content', '')}"
        for c in chunks
    )
    return (
        f"COMPANY: {company}\nQUARTER: {quarter}\n\n"
        f"=== RETRIEVED EVIDENCE ===\n{chunk_text}\n\n"
        f"=== LANGUAGE SHIFT ANALYSIS ===\n"
        f"Use evidence above to identify quarter-over-quarter language shifts.\n\n"
        f"=== SENTIMENT ANALYSIS ===\n"
        f"Use evidence above to assess tone (positive/negative/neutral) in key passages.\n\n"
        f"=== NUMERIC VALIDATION ===\n"
        f"Use evidence above to identify and cite financial figures with units.\n\n"
        f"{question}"
    )


# ── Teacher LLM call ──────────────────────────────────────────────────────────

def _call_teacher(system: str, user: str, dry_run: bool) -> str:
    """
    Call gpt-5.4-mini (primary deployment) synchronously.
    Returns empty string on failure.
    """
    if dry_run:
        return "[DRY RUN — no API call made]"
    try:
        response = openai_client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("Teacher LLM call failed: %s", exc)
        return ""


# ── JSONL record builder ──────────────────────────────────────────────────────

def _make_record(system: str, user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _estimate_tokens(record: dict) -> int:
    """Rough token estimate: 1 token ≈ 4 chars."""
    total_chars = sum(len(m["content"]) for m in record["messages"])
    return total_chars // 4


# ── Main generation loop ──────────────────────────────────────────────────────

def generate(dry_run: bool = False) -> None:
    examples: list[dict] = []
    skipped: list[str] = []
    total_A = 0
    total_B = 0

    logger.info("Starting dataset generation — dry_run=%s", dry_run)
    logger.info("Target: 50 Prompt-A + 100 Prompt-B = 150 examples")

    # ── Prompt A examples ─────────────────────────────────────────────────
    logger.info("=== Generating Prompt A examples (simple queries) ===")
    for company, quarter in _COMBOS:
        for template in _QUERIES_A:
            question = template.format(company=company, quarter=quarter)
            chunks = _fetch_evidence(question, company, quarter) if not dry_run else []

            if not chunks and not dry_run:
                label = f"{company}/{quarter} | {template[:40]}"
                logger.warning("SKIP (empty retrieval): %s", label)
                skipped.append(label)
                continue

            user = _build_user_A(company, quarter, chunks, question)
            assistant = _call_teacher(_SYSTEM_A, user, dry_run)

            if not assistant and not dry_run:
                label = f"{company}/{quarter} | {template[:40]}"
                logger.warning("SKIP (empty LLM response): %s", label)
                skipped.append(label)
                continue

            record = _make_record(_SYSTEM_A, user, assistant)
            if not _validate_record(record):
                skipped.append(f"{company}/{quarter} | invalid JSON structure")
                continue

            examples.append(record)
            total_A += 1
            logger.info("[A %d/50] %s %s — %d chunks", total_A, company, quarter, len(chunks))

            if not dry_run:
                time.sleep(2)

    # ── Prompt B examples ─────────────────────────────────────────────────
    logger.info("=== Generating Prompt B examples (full analysis) ===")
    for company, quarter in _COMBOS:
        for template in _QUERIES_B:
            question = template.format(company=company, quarter=quarter)
            chunks = _fetch_evidence(question, company, quarter) if not dry_run else []

            if not chunks and not dry_run:
                label = f"{company}/{quarter} | {template[:40]}"
                logger.warning("SKIP (empty retrieval): %s", label)
                skipped.append(label)
                continue

            user = _build_user_B(company, quarter, chunks, question)
            assistant = _call_teacher(_SYSTEM_B, user, dry_run)

            if not assistant and not dry_run:
                label = f"{company}/{quarter} | {template[:40]}"
                logger.warning("SKIP (empty LLM response): %s", label)
                skipped.append(label)
                continue

            record = _make_record(_SYSTEM_B, user, assistant)
            if not _validate_record(record):
                skipped.append(f"{company}/{quarter} | invalid JSON structure")
                continue

            examples.append(record)
            total_B += 1
            logger.info("[B %d/100] %s %s — %d chunks", total_B, company, quarter, len(chunks))

            if not dry_run:
                time.sleep(2)

    # ── Shuffle + split ───────────────────────────────────────────────────
    random.seed(42)
    random.shuffle(examples)

    n_train = int(len(examples) * 0.8)
    train_examples = examples[:n_train]
    val_examples = examples[n_train:]

    # ── Write outputs ─────────────────────────────────────────────────────
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(_TRAIN_PATH, train_examples)
    _write_jsonl(_VAL_PATH, val_examples)

    # ── Summary ───────────────────────────────────────────────────────────
    token_counts = [_estimate_tokens(r) for r in examples]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Prompt A examples generated : {total_A} / 50")
    print(f"  Prompt B examples generated : {total_B} / 100")
    print(f"  Total examples              : {len(examples)}")
    print(f"  Training set                : {len(train_examples)}")
    print(f"  Validation set              : {len(val_examples)}")
    print(f"  Skipped (empty/error)       : {len(skipped)}")
    print(f"  Avg tokens per example (est): {avg_tokens:.0f}")
    print(f"  Output: {_TRAIN_PATH}")
    print(f"  Output: {_VAL_PATH}")
    if skipped:
        print(f"\n  Skipped examples:")
        for s in skipped:
            print(f"    - {s}")
    print("=" * 60)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_record(record: dict) -> bool:
    """Confirm record serialises to valid JSON and has required structure."""
    try:
        serialised = json.dumps(record, ensure_ascii=False)
        parsed = json.loads(serialised)
        msgs = parsed.get("messages", [])
        if len(msgs) != 3:
            return False
        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            return False
        if not all(isinstance(m.get("content"), str) and m["content"] for m in msgs):
            return False
        return True
    except (ValueError, TypeError):
        return False


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSONL with UTF-8 BOM encoding."""
    with open(path, "w", encoding="utf-8-sig") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuarterLens knowledge distillation dataset generator")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate structure without making API calls",
    )
    args = parser.parse_args()
    generate(dry_run=args.dry_run)