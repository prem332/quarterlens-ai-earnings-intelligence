"""
Smoke test for Phase 3 structured ComparisonFinding changes.
Tests:
  1. ComparisonFinding TypedDict accepts new optional fields
  2. _to_finding() maps all fields correctly (with and without structured fields)
  3. _build_draft_prompt() renders structured facts vs narrative fallback
  4. _build_chunk_text() returns same string for both draft and verify
  5. _ranked_context() preserves input order

Run from repo root:
    python test_phase3_changes.py
"""

import sys
import os
sys.path.insert(0, '.')

print("=" * 60)
print("Phase 3 Smoke Tests")
print("=" * 60)

failures = []

# ── Test 1: ComparisonFinding with new fields ─────────────────────────────
print("\n[1] ComparisonFinding TypedDict — new optional fields...")
try:
    from graph.state import ComparisonFinding, RetrievalResult, GraphState

    # With structured fields
    f1 = ComparisonFinding(
        topic="Gross Margin",
        current_language="Gross margin expanded to 46.5%",
        prior_language={"FY2025-Q2": "Gross margin was 45.2%"},
        shift_detected=True,
        shift_description="Margin expanded 1.3pp",
        metric="gross_margin",
        current_value="46.5%",
        prior_value="45.2%",
        delta="+1.3pp",
        source_section="mda",
    )
    assert f1["metric"] == "gross_margin"
    assert f1["delta"] == "+1.3pp"

    # Without structured fields (backward compat)
    f2 = ComparisonFinding(
        topic="Revenue Guidance",
        current_language="We expect continued growth",
        prior_language={"FY2025-Q2": "We maintained our outlook"},
        shift_detected=False,
        shift_description=None,
        metric=None,
        current_value=None,
        prior_value=None,
        delta=None,
        source_section=None,
    )
    assert f2["metric"] is None

    print("  ✓ ComparisonFinding accepts structured and null fields")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    failures.append("ComparisonFinding TypedDict")

# ── Test 2: _to_finding() mapping ─────────────────────────────────────────
print("\n[2] comparison_agent._to_finding() field mapping...")
try:
    from agents.comparison_agent import _to_finding

    # Full structured response from LLM
    raw_structured = {
        "topic": "Gross Margin",
        "current_language": "Gross margin expanded to 46.5%",
        "prior_language": {"FY2025-Q2": "Gross margin was 45.2%"},
        "shift_detected": True,
        "shift_description": "Margin expanded 1.3pp YoY",
        "metric": "gross_margin",
        "current_value": "46.5%",
        "prior_value": "45.2%",
        "delta": "+1.3pp",
        "source_section": "mda",
    }
    f = _to_finding(raw_structured)
    assert f["metric"] == "gross_margin"
    assert f["current_value"] == "46.5%"
    assert f["delta"] == "+1.3pp"
    assert f["source_section"] == "mda"
    print("  ✓ Structured fields mapped correctly")

    # Old-style response without new fields (backward compat)
    raw_old = {
        "topic": "Revenue",
        "current_language": "Revenue grew strongly",
        "prior_language": {"FY2025-Q2": "Revenue was solid"},
        "shift_detected": False,
        "shift_description": None,
    }
    f2 = _to_finding(raw_old)
    assert f2["metric"] is None
    assert f2["delta"] is None
    print("  ✓ Missing structured fields default to None (backward compat)")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    failures.append("_to_finding() mapping")

# ── Test 3: _ranked_context() preserves order ─────────────────────────────
print("\n[3] comparison_agent._ranked_context() order preservation...")
try:
    from agents.comparison_agent import _ranked_context

    chunks = [
        {"content": "First chunk — highest rerank score"},
        {"content": "Second chunk — medium score"},
        {"content": "Third chunk — lower score"},
    ]
    result = _ranked_context(chunks, max_chars=10000)
    parts = result.split("\n\n")
    assert parts[0] == "First chunk — highest rerank score"
    assert parts[1] == "Second chunk — medium score"
    assert parts[2] == "Third chunk — lower score"
    print("  ✓ Input order preserved")

    # Test max_chars truncation
    result_short = _ranked_context(chunks, max_chars=50)
    assert "Third chunk" not in result_short
    print("  ✓ max_chars truncation works")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    failures.append("_ranked_context() order")

# ── Test 4: _build_chunk_text() consistency ───────────────────────────────
print("\n[4] report_agent._build_chunk_text() draft/verify consistency...")
try:
    from agents.report_agent import _build_chunk_text

    mock_state = {
        "company": "AAPL",
        "quarter": "FY2025-Q3",
        "query": "analyze earnings",
        "retrieval_results": [
            {"doc_type": "10-Q", "content": "Revenue was $94B for the quarter.", "company": "AAPL", "quarter": "FY2025-Q3", "fiscal_label": "FY2025-Q3", "score": 0.9, "accession": "acc1", "section": "mda", "chunk_id": "c1"},
            {"doc_type": "transcript", "content": "CEO: We are very pleased with results.", "company": "AAPL", "quarter": "FY2025-Q3", "fiscal_label": "FY2025-Q3", "score": 0.8, "accession": "acc2", "section": "transcript_part_0", "chunk_id": "c2"},
        ],
        "comparison_findings": [],
        "sentiment_scores": [],
        "numeric_validations": [],
        "model_tier": "primary",
        "report_model_tier": "primary",
    }

    chunk_text_1 = _build_chunk_text(mock_state)
    chunk_text_2 = _build_chunk_text(mock_state)

    assert chunk_text_1 == chunk_text_2, "chunk_text must be deterministic"
    assert "[10-Q]" in chunk_text_1
    assert "[TRANSCRIPT]" in chunk_text_1
    assert "Revenue was $94B" in chunk_text_1
    print("  ✓ chunk_text is deterministic and identical for draft/verify")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    failures.append("_build_chunk_text() consistency")

# ── Test 5: _build_draft_prompt() structured vs narrative findings ─────────
print("\n[5] report_agent._build_draft_prompt() structured fact rendering...")
try:
    from agents.report_agent import _build_draft_prompt
    from graph.state import ComparisonFinding

    mock_state_structured = {
        "company": "AAPL",
        "quarter": "FY2025-Q3",
        "retrieval_results": [],
        "comparison_findings": [
            ComparisonFinding(
                topic="Gross Margin", current_language="46.5%",
                prior_language={}, shift_detected=True,
                shift_description="Expanded", metric="gross_margin",
                current_value="46.5%", prior_value="45.2%",
                delta="+1.3pp", source_section="mda",
            )
        ],
        "sentiment_scores": [],
        "numeric_validations": [],
    }

    prompt_structured = _build_draft_prompt(mock_state_structured, chunk_text="test evidence")
    assert "gross_margin" in prompt_structured
    assert "current=46.5%" in prompt_structured
    assert "delta=+1.3pp" in prompt_structured
    assert "[mda]" in prompt_structured
    print("  ✓ Structured findings rendered as fact table")

    # Narrative fallback
    mock_state_narrative = {
        "company": "AAPL",
        "quarter": "FY2025-Q3",
        "retrieval_results": [],
        "comparison_findings": [
            ComparisonFinding(
                topic="Revenue", current_language="grew",
                prior_language={}, shift_detected=False,
                shift_description="no change", metric=None,
                current_value=None, prior_value=None,
                delta=None, source_section=None,
            )
        ],
        "sentiment_scores": [],
        "numeric_validations": [],
    }

    prompt_narrative = _build_draft_prompt(mock_state_narrative, chunk_text="test evidence")
    assert "no change" in prompt_narrative
    print("  ✓ Narrative fallback renders correctly when metric=None")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    failures.append("_build_draft_prompt() rendering")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if failures:
    print(f"FAILED: {len(failures)} test(s)")
    for f in failures:
        print(f"  ✗ {f}")
    sys.exit(1)
else:
    print(f"ALL TESTS PASSED (5/5)")
    print("Safe to run eval.")