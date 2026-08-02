"""
evaluation/eval_gate_check.py

Deploy quality gate: runs a small stratified sample of the golden dataset
through the real pipeline (same run_eval() that backs the baseline/
production evals) and fails the build if any of the 7 locked headline
metrics drops below a regression-guard floor.

NOT a target gate. Two of the 7 metrics (llm_judge, context_recall) sit
below CLAUDE.md's own stated target today and are shipping anyway,
deliberately (see "Metric Targets — LOCKED" / Known Issues). Floors here
guard against a REGRESSION from the current locked baseline
(production-full-eval-75), not against falling short of a target nobody
has hit yet — the point is to catch a broken prompt or a retrieval
regression before it deploys, not to freeze all future deploys until
llm_judge reaches 4.5.

Runs GATE_MAX_CLAIMS claims (default 10) stratified across claim types —
deliberately smaller than a full baseline run, since this executes on
every push to main via eval_gate.yml -> deploy.yml, not as a one-off
budgeted session action (same cost reasoning as smoke_test.py's
docstring, one level up: smoke_test.py spends ~4 LLM calls per push,
this spends roughly 10 claims x (pipeline + ~5 RAGAS + 1 judge) calls).

Locked to CONTEXT_PRECISION_K=2 / CONTEXT_PRECISION_CHUNK_CHARS=0 to match
the headline production-full-eval-75 measurement window (0.8219) rather
than run_baseline_eval.py's own default k=5/chunk=300 window (0.51-0.56,
per CLAUDE.md) — comparing the default window's output against a floor
set for the k=2 window would either be meaninglessly loose or falsely
fail every run.

Exit code 0 = metrics within floor, safe to deploy. Non-zero = do not
deploy; prints which metric(s) failed and by how much.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must be set before importing run_baseline_eval — it reads these two at
# module import time (its own CONTEXT_PRECISION_K/_CHUNK_CHARS globals).
os.environ.setdefault("CONTEXT_PRECISION_K", "2")
os.environ.setdefault("CONTEXT_PRECISION_CHUNK_CHARS", "0")

from evaluation.run_baseline_eval import run_eval  # noqa: E402

_CLAIMS_DIR = Path(__file__).resolve().parent.parent / "golden_dataset" / "claims"
_MAX_CLAIMS = int(os.environ.get("GATE_MAX_CLAIMS", "10"))

# Regression-guard floors, not target thresholds — see module docstring.
# Baseline values in the comments are production-full-eval-75 (CLAUDE.md).
_FLOORS: dict[str, float] = {
    "ragas_faithfulness": 0.80,             # baseline 0.9353
    "ragas_answer_relevancy": 0.80,         # baseline 0.9616
    "ragas_context_precision": 0.65,        # baseline 0.8219 (k=2/full-chunk)
    "ragas_context_recall_excl_oos": 0.65,  # baseline 0.8918
    "precision_at_5": 0.55,                 # baseline 0.7222 (target 0.60)
    "recall_at_5": 0.90,                    # baseline 1.0000, historically stable
    "llm_judge_mean": 3.30,                 # baseline 3.9863 (target 4.5, not yet met)
}


async def _main() -> None:
    print(
        f"Eval gate: running {_MAX_CLAIMS} stratified claims "
        f"(CONTEXT_PRECISION_K={os.environ['CONTEXT_PRECISION_K']}, "
        f"CONTEXT_PRECISION_CHUNK_CHARS={os.environ['CONTEXT_PRECISION_CHUNK_CHARS']})\n"
    )

    metrics = await run_eval(_CLAIMS_DIR, max_claims=_MAX_CLAIMS, run_name="eval-gate")

    print("\n" + "=" * 55)
    print("EVAL GATE — floor check")
    print("=" * 55)
    failures: list[str] = []
    for name, floor in _FLOORS.items():
        val = metrics.get(name)
        if val is None:
            failures.append(f"{name}: MISSING from metrics output")
            print(f"  {name}: MISSING")
            continue
        ok = val >= floor
        print(f"  {name}: {val:.4f} (floor {floor:.4f}) [{'PASS' if ok else 'FAIL'}]")
        if not ok:
            failures.append(f"{name}: {val:.4f} < floor {floor:.4f}")
    print("=" * 55)

    if failures:
        print(f"\nEVAL GATE FAILED — {len(failures)} metric(s) below floor:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\nEVAL GATE PASSED — all metrics within floor, safe to deploy.")


if __name__ == "__main__":
    asyncio.run(_main())
