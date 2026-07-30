import json
import uuid
import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas.requests import AnalysisRequest
from api.schemas.responses import AnalysisResponse, RunStatusResponse
from api.schemas.shared import RunStatus
from azure_clients.blob_client import BlobClient
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


def _blob() -> BlobClient:
    return BlobClient()


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
    blob = _blob()
    queue: asyncio.Queue = asyncio.Queue()
    _stream_queues[run_id] = queue
    try:
        # Mark running
        doc = json.loads(blob.download_blob(CONTAINER, _blob_path(run_id)))
        doc["status"] = RunStatus.RUNNING
        blob.upload_blob(CONTAINER, _blob_path(run_id), json.dumps(doc).encode(), overwrite=True)

        state = GraphState(
            company=req.company,
            quarter=_to_fiscal_label(req.quarter),
            query=req.query,
            comparison_quarters=[_to_fiscal_label(q) for q in req.comparison_quarters],
        )
        state["stream_queue"] = queue  # not in GraphState's TypedDict — see report_agent.py
        result: dict = await compiled_graph.ainvoke(state)
        result["created_at"] = created_at

        blob.upload_blob(
            CONTAINER,
            _blob_path(run_id),
            _serialize(run_id, req, RunStatus.COMPLETED, result=result),
            overwrite=True,
        )
    except Exception as exc:
        doc = json.loads(blob.download_blob(CONTAINER, _blob_path(run_id)))
        doc["status"] = RunStatus.FAILED
        doc["error"] = str(exc)
        doc["completed_at"] = datetime.now(timezone.utc).isoformat()
        blob.upload_blob(CONTAINER, _blob_path(run_id), json.dumps(doc).encode(), overwrite=True)
    finally:
        # Always signal stream end, success or failure — otherwise a browser
        # connected to /stream would hang on queue.get() forever. report_agent
        # already pushes {"type": "final", ...} on its own success path; this
        # sentinel is the one thing guaranteed to arrive on every path,
        # including a failure before report_agent ever ran.
        await queue.put(None)


@router.post("/run", response_model=RunStatusResponse, status_code=202)
async def run_analysis(req: AnalysisRequest):
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    blob = _blob()
    blob.upload_blob(
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


@router.get("/{run_id}/status", response_model=RunStatusResponse)
async def get_status(run_id: str):
    blob = _blob()
    if not blob.blob_exists(CONTAINER, _blob_path(run_id)):
        raise HTTPException(status_code=404, detail="Run not found")
    doc = json.loads(blob.download_blob(CONTAINER, _blob_path(run_id)))
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
    blob = _blob()
    if not blob.blob_exists(CONTAINER, _blob_path(run_id)):
        raise HTTPException(status_code=404, detail="Run not found")
    doc = json.loads(blob.download_blob(CONTAINER, _blob_path(run_id)))
    doc["created_at"] = datetime.fromisoformat(doc["created_at"])
    if doc.get("completed_at"):
        doc["completed_at"] = datetime.fromisoformat(doc["completed_at"])
    return AnalysisResponse(**doc)