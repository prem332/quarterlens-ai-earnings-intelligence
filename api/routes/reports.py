import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query

from api.schemas.responses import AnalysisResponse, ReportSummary
from api.schemas.shared import Company
from azure_clients.blob_client import BlobClient

router = APIRouter(prefix="/api/reports", tags=["reports"])

CONTAINER = "raw-documents"
REPORTS_PREFIX = "reports"


def _blob() -> BlobClient:
    return BlobClient()


@router.get("", response_model=list[ReportSummary])
async def list_reports(
    company: Company | None = None,
    quarter: str | None = None,
    limit: int = Query(default=20, le=100),
    offset: int = 0,
):
    blob = _blob()
    paths = blob.list_blobs(CONTAINER, prefix=REPORTS_PREFIX)
    results = []
    for path in paths:
        try:
            doc = json.loads(blob.download_blob(CONTAINER, path))
            if doc.get("status") != "completed":
                continue
            if company and doc.get("company") != company:
                continue
            if quarter and doc.get("quarter") != quarter:
                continue
            results.append(doc)
        except Exception:
            continue

    # Sort newest first
    results.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    results = results[offset: offset + limit]

    return [
        ReportSummary(
            run_id=d["run_id"],
            company=d["company"],
            quarter=d["quarter"],
            status=d["status"],
            created_at=datetime.fromisoformat(d["created_at"]),
            report_snippet=(d.get("report") or "")[:200] or None,
        )
        for d in results
    ]


@router.get("/{run_id}", response_model=AnalysisResponse)
async def get_report(run_id: str):
    blob = _blob()
    path = f"{REPORTS_PREFIX}/{run_id}.json"
    if not blob.blob_exists(CONTAINER, path):
        raise HTTPException(status_code=404, detail="Report not found")
    doc = json.loads(blob.download_blob(CONTAINER, path))
    doc["created_at"] = datetime.fromisoformat(doc["created_at"])
    if doc.get("completed_at"):
        doc["completed_at"] = datetime.fromisoformat(doc["completed_at"])
    return AnalysisResponse(**doc)


@router.delete("/{run_id}", status_code=204)
async def delete_report(run_id: str):
    # BlobClient has no delete method — mark as deleted via status flag
    blob = _blob()
    path = f"{REPORTS_PREFIX}/{run_id}.json"
    if not blob.blob_exists(CONTAINER, path):
        raise HTTPException(status_code=404, detail="Report not found")
    doc = json.loads(blob.download_blob(CONTAINER, path))
    doc["status"] = "deleted"
    blob.upload_blob(CONTAINER, path, json.dumps(doc).encode(), overwrite=True)