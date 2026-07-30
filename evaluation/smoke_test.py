"""
evaluation/smoke_test.py

Deploy-gate smoke test — deliberately NOT the RAGAS/LLM-judge eval suite.
Runs exactly one real claim through the full compiled_graph pipeline (real
Azure OpenAI calls, real AI Search retrieval) and asserts it completes
without error, in a bounded time, with a non-empty report. That's it: no
faithfulness/context_precision/judge scoring, no golden_dataset claims file.

Why not the real eval suite here: run_baseline_eval.py's RAGAS + LLM-judge
scoring calls the LLM many times per claim (per CLAUDE.md's own "Running
Evaluations" section) — fine for a deliberate, budget-tracked, manual
session action, not something to re-spend on every push to main. This
script costs roughly one report_agent run (draft + verify + bull/bear ~
4 LLM calls total), not dozens.

Exit code 0 = pipeline is healthy enough to deploy. Non-zero = don't deploy.
"""

import asyncio
import sys
import time
from pathlib import Path

# Run directly with `python evaluation/smoke_test.py` (as deploy.yml does),
# not `python -m`, so the repo root isn't on sys.path automatically -- same
# gap run_baseline_eval.py already works around this same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.build_graph import compiled_graph
from graph.state import GraphState

_COMPANY = "MSFT"
_QUARTER = "FY2026-Q2"
_QUERY = "Summarize key earnings findings and verify management claims."
_MAX_SECONDS = 240  # generous — see NewAnalysis.jsx's own ~110s real-run baseline


async def _run() -> None:
    initial_state: GraphState = {
        "company": _COMPANY,
        "quarter": _QUARTER,
        "query": _QUERY,
        "comparison_quarters": [],
        "retrieval_results": [],
        "transcript_retrieval_results": [],
        "comparison_findings": [],
        "sentiment_scores": [],
        "numeric_validations": [],
        "report": "",
        "decision_log_entries": [],
        "model_tier": "primary",
        "error": None,
    }

    start = time.time()
    result = await compiled_graph.ainvoke(initial_state)
    elapsed = time.time() - start

    if result.get("error"):
        print(f"SMOKE TEST FAILED: pipeline set error={result['error']!r}")
        sys.exit(1)

    report = result.get("report") or ""
    if not report.strip():
        print("SMOKE TEST FAILED: report is empty")
        sys.exit(1)

    if elapsed > _MAX_SECONDS:
        print(f"SMOKE TEST FAILED: took {elapsed:.1f}s, budget is {_MAX_SECONDS}s")
        sys.exit(1)

    retrieval_count = len(result.get("retrieval_results") or [])
    print(
        f"SMOKE TEST PASSED: {elapsed:.1f}s, "
        f"{retrieval_count} chunks retrieved, {len(report)} char report"
    )


if __name__ == "__main__":
    asyncio.run(_run())
