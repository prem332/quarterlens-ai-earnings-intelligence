import json
import uuid
import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from opentelemetry import trace as otel_trace

from agents.debate_crew import build_evidence_summary, run_debate
from api.guardrails import check_query
from api.schemas.requests import AnalysisRequest
from api.schemas.responses import AnalysisResponse, RunStatusResponse
from api.schemas.shared import RunStatus
from api.blob_helpers import blob_exists, download_blob, upload_blob
from azure_clients.redis_client import get_run_cached, set_run_cached
from graph.build_graph import compiled_graph
from graph.state import GraphState

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

CONTAINER = "raw-documents"
REPORTS_PREFIX = "reports"

# run_id -> queue of report_agent draft/verify events for the SSE endpoint
# below. In-memory, single-process — fine for this dev deployment, would need
# a shared store (Redis pub/sub) behind more than one uvicorn worker. Entries
# are created in run_analysis, written to by report_agent (via the
# stream_queue key threaded through GraphState — see report_agent.py), and
# torn down by the SSE generator once it reads the terminal event, or by
# run_analysis on early failure if no one ever connected.
_stream_queues: dict[str, asyncio.Queue] = {}

# Vendor-neutral tracer, shared by both observability backends. Phoenix's
# phoenix.otel.register() (observability/phoenix_setup.py) sets the global
# OTEL tracer provider, so get_tracer() here resolves to it automatically.
# Langfuse's SDK (v4+, fully OTEL-based under the hood -- confirmed via its
# own exported OtelSpanData/LangfuseOtelSpanAttributes types) uses the same
# ambient OTEL context to parent its own spans. One span wrapping the whole
# pipeline call therefore groups every LLM/embedding call from BOTH
# instrumentors under one real per-run trace in both dashboards, instead of
# each individual OpenAI call showing up as its own disconnected root trace
# (confirmed missing before this: zero calls to langfuse's own trace()/
# get_langfuse_client() existed anywhere in agents/, api/, or graph/).
_tracer = otel_trace.get_tracer(__name__)


def _to_fiscal_label(quarter: str) -> str:
    """Convert 'Q2_2025' → 'FY2025-Q2' to match index fiscal_label format."""
    if quarter.startswith("FY"):
        return quarter  # already in correct format
    # expect Q{n}_{yyyy}
    parts = quarter.split("_")
    if len(parts) == 2 and parts[0].startswith("Q") and parts[1].isdigit():
        return f"FY{parts[1]}-{parts[0]}"
    return quarter  # passthrough if unrecognised


def _blob_path(run_id: str) -> str:
    return f"{REPORTS_PREFIX}/{run_id}.json"




def _serialize(run_id: str, req: AnalysisRequest, status: RunStatus, result: dict = None, error: str = None) -> bytes:
    # Store the CANONICAL fiscal label, not the raw request value. The pipeline
    # runs on _to_fiscal_label(req.quarter), so recording req.quarter verbatim
    # let a legacy-format request ("Q2_2025") be analysed as FY2025-Q2 while the
    # report header still read "Q2_2025" — the stored doc disagreed with the
    # analysis it describes.
    doc = {
        "run_id": run_id,
        "company": req.company,
        "quarter": _to_fiscal_label(req.quarter),
        "comparison_quarters": [_to_fiscal_label(q) for q in req.comparison_quarters],
        "query": req.query,
        "status": status,
        "created_at": result.get("created_at") if result else datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat() if status in (RunStatus.COMPLETED, RunStatus.FAILED) else None,
        "error": error,
        "report": result.get("report") if result else None,
        "retrieval_results": result.get("retrieval_results", []) if result else [],
        "comparison_findings": result.get("comparison_findings", []) if result else [],
        "sentiment_scores": result.get("sentiment_scores", []) if result else [],
        "numeric_validations": result.get("numeric_validations", []) if result else [],
        "decision_log_entries": result.get("decision_log_entries", []) if result else [],
    }
    return json.dumps(doc).encode()


async def _run_pipeline(run_id: str, req: AnalysisRequest, created_at: str):
    queue: asyncio.Queue = asyncio.Queue()
    _stream_queues[run_id] = queue
    quarter = _to_fiscal_label(req.quarter)
    comp_quarters = [_to_fiscal_label(q) for q in req.comparison_quarters]
    try:
        # ── Full-run cache ────────────────────────────────────────────────
        # An identical request re-runs the entire 5-agent pipeline otherwise.
        # Serving it from cache turns a ~30s analysis into a sub-second one.
        # Only completed runs with a real report are ever stored, so a cache
        # hit can't resurrect a rate-limited or errored run.
        if not req.no_cache:
            cached = await asyncio.to_thread(
                get_run_cached, req.query, req.company, quarter, comp_quarters
            )
            if cached:
                doc = {
                    **cached,
                    "run_id": run_id,
                    "created_at": created_at,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "status": RunStatus.COMPLETED,
                }
                await upload_blob(CONTAINER, _blob_path(run_id), json.dumps(doc).encode())
                # The browser is waiting on the SSE stream; hand it the report
                # directly since no draft tokens will be generated this time.
                await queue.put({"type": "final", "report": doc.get("report") or ""})
                return

        # Mark running
        doc = json.loads(await download_blob(CONTAINER, _blob_path(run_id)))
        doc["status"] = RunStatus.RUNNING
        await upload_blob(CONTAINER, _blob_path(run_id), json.dumps(doc).encode())

        state = GraphState(
            company=req.company,
            quarter=_to_fiscal_label(req.quarter),
            query=req.query,
            comparison_quarters=[_to_fiscal_label(q) for q in req.comparison_quarters],
        )
        state["stream_queue"] = queue  # not in GraphState's TypedDict — see report_agent.py

        with _tracer.start_as_current_span(
            "analysis_pipeline_run",
            attributes={
                "run_id": run_id,
                "company": req.company,
                "quarter": _to_fiscal_label(req.quarter),
                "comparison_quarters": ",".join(_to_fiscal_label(q) for q in req.comparison_quarters),
                "query": req.query[:200],
            },
        ):
            result: dict = await compiled_graph.ainvoke(state)
        result["created_at"] = created_at

        # An agent that set state["error"] did NOT succeed, even though
        # ainvoke() returned normally. Without this the run was stored as
        # COMPLETED with error=None and an empty report, so the UI showed a
        # blank page and the user had no way to tell a rate-limited run from
        # a genuine "no data" answer.
        agent_error = result.get("error")
        status = RunStatus.FAILED if agent_error else RunStatus.COMPLETED

        payload = _serialize(run_id, req, status, result=result, error=agent_error)
        await upload_blob(CONTAINER, _blob_path(run_id), payload, overwrite=True)

        # Cache only a genuine success. A failed or empty-report run must never
        # be served to a later request.
        if not agent_error and (result.get("report") or "").strip():
            await asyncio.to_thread(
                set_run_cached, req.query, req.company, quarter, comp_quarters,
                json.loads(payload),
            )
    except Exception as exc:
        doc = json.loads(await download_blob(CONTAINER, _blob_path(run_id)))
        doc["status"] = RunStatus.FAILED
        doc["error"] = str(exc)
        doc["completed_at"] = datetime.now(timezone.utc).isoformat()
        await upload_blob(CONTAINER, _blob_path(run_id), json.dumps(doc).encode())
    finally:
        # Always signal stream end, success or failure — otherwise a browser
        # connected to /stream would hang on queue.get() forever. report_agent
        # already pushes {"type": "final", ...} on its own success path; this
        # sentinel is the one thing guaranteed to arrive on every path,
        # including a failure before report_agent ever ran.
        await queue.put(None)


@router.post("/run", response_model=RunStatusResponse, status_code=202)
async def run_analysis(req: AnalysisRequest):
    # Input guardrails run here — before the run is even registered, let
    # alone before compiled_graph.ainvoke() reaches an Azure OpenAI call.
    # A rejected query never spends a token and is never sent to the LLM.
    verdict = check_query(req.query)
    if not verdict.allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Query rejected by input guardrails ({verdict.category}): {verdict.reason}",
        )

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    await upload_blob(
        CONTAINER,
        _blob_path(run_id),
        _serialize(run_id, req, RunStatus.PENDING),
        overwrite=True,
    )

    asyncio.create_task(_run_pipeline(run_id, req, created_at))

    return RunStatusResponse(
        run_id=run_id,
        status=RunStatus.PENDING,
        company=req.company,
        quarter=_to_fiscal_label(req.quarter),
        created_at=datetime.fromisoformat(created_at),
    )


@router.get("/{run_id}/stream")
async def stream_analysis(run_id: str):
    """
    Server-Sent Events feed of report_agent's draft/verify progress.

    Events (each a JSON object in the SSE `data:` field):
      {"type": "draft_token", "text": "..."} — one text delta from the
          streaming draft call, in order. Concatenate to reconstruct the
          in-progress draft.
      {"type": "draft_reset"} — rare: a 429 forced a mid-draft retry: discard
          whatever draft_token text has been accumulated so far and restart.
      {"type": "verifying"} — draft finished, the verify pass has started.
          No more draft_token events will follow.
      {"type": "final", "report": "..."} — the verified report text. This is
          what actually gets stored/exported, and may differ from the drafted
          text the browser was just shown (verify can delete or rewrite
          unsupported sentences).
      {"type": "done"} — stream is over (terminal event either way — sent
          whether the run succeeded or failed; check GET /{run_id}/status for
          the actual outcome).
    """
    queue = _stream_queues.get(run_id)
    if queue is None:
        raise HTTPException(
            status_code=404,
            detail="No active stream for this run — it may already be finished, or never started",
        )

    async def event_gen():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    yield 'data: {"type": "done"}\n\n'
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            _stream_queues.pop(run_id, None)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/{run_id}/debate")
async def run_debate_endpoint(run_id: str):
    """
    CrewAI bull/bear analyst debate over a completed run's evidence.

    Deliberately on demand rather than part of the pipeline: it costs ~11s and
    two extra LLM calls, and it's supplementary colour rather than part of the
    verified answer. Users who want the investment-debate view ask for it;
    everyone else gets their report ~11s sooner.
    """
    if not await blob_exists(CONTAINER, _blob_path(run_id)):
        raise HTTPException(status_code=404, detail="Run not found")

    doc = json.loads(await download_blob(CONTAINER, _blob_path(run_id)))
    if doc.get("status") != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Run is '{doc.get('status')}' — a debate needs a completed run's evidence",
        )

    evidence = build_evidence_summary(doc)
    if not evidence.strip():
        raise HTTPException(status_code=422, detail="Run has no retrieved evidence to debate")

    result = await run_debate(evidence)
    if not result["bull"] and not result["bear"]:
        raise HTTPException(status_code=502, detail="Both analyst crews failed — try again")
    return result


@router.get("/{run_id}/status", response_model=RunStatusResponse)
async def get_status(run_id: str):
    if not await blob_exists(CONTAINER, _blob_path(run_id)):
        raise HTTPException(status_code=404, detail="Run not found")
    doc = json.loads(await download_blob(CONTAINER, _blob_path(run_id)))
    return RunStatusResponse(
        run_id=doc["run_id"],
        status=doc["status"],
        company=doc["company"],
        quarter=doc["quarter"],
        created_at=datetime.fromisoformat(doc["created_at"]),
        completed_at=datetime.fromisoformat(doc["completed_at"]) if doc.get("completed_at") else None,
        error=doc.get("error"),
    )


@router.get("/{run_id}", response_model=AnalysisResponse)
async def get_analysis(run_id: str):
    if not await blob_exists(CONTAINER, _blob_path(run_id)):
        raise HTTPException(status_code=404, detail="Run not found")
    doc = json.loads(await download_blob(CONTAINER, _blob_path(run_id)))
    doc["created_at"] = datetime.fromisoformat(doc["created_at"])
    if doc.get("completed_at"):
        doc["completed_at"] = datetime.fromisoformat(doc["completed_at"])
    return AnalysisResponse(**doc)