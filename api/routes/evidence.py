import json
from fastapi import APIRouter, HTTPException

from api.schemas.responses import ClaimEvidence
from azure_clients.blob_client import BlobClient

router = APIRouter(prefix="/api/evidence", tags=["evidence"])

CONTAINER = "raw-documents"
REPORTS_PREFIX = "reports"


def _blob() -> BlobClient:
    return BlobClient()


def _load(run_id: str) -> dict:
    blob = _blob()
    path = f"{REPORTS_PREFIX}/{run_id}.json"
    if not blob.blob_exists(CONTAINER, path):
        raise HTTPException(status_code=404, detail="Run not found")
    return json.loads(blob.download_blob(CONTAINER, path))


def _extract_claims(doc: dict) -> list[ClaimEvidence]:
    claims = []
    for entry in doc.get("decision_log_entries", []):
        if entry.get("type") == "claim":
            claims.append(ClaimEvidence(
                claim_id=entry["claim_id"],
                claim_text=entry["claim_text"],
                source_section=entry.get("section", ""),
                source_paragraph=entry.get("source_text", ""),
                confidence=entry.get("confidence", 0.0),
                doc_type=entry.get("doc_type", ""),
                quarter=entry.get("quarter", doc["quarter"]),
            ))
    return claims


@router.get("/{run_id}/claims", response_model=list[ClaimEvidence])
async def list_claims(run_id: str):
    return _extract_claims(_load(run_id))


@router.get("/{run_id}/claims/{claim_id}", response_model=ClaimEvidence)
async def get_claim(run_id: str, claim_id: str):
    doc = _load(run_id)
    for c in _extract_claims(doc):
        if c.claim_id == claim_id:
            return c
    raise HTTPException(status_code=404, detail="Claim not found")