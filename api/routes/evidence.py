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


def _supporting_chunk(doc: dict, fiscal_label: str) -> dict:
    """
    Best available retrieved chunk backing a claim from `fiscal_label`.

    numeric_validations don't carry a chunk reference, so this scopes by the
    filing coordinate the validation actually used and prefers a filing chunk
    (the claim is checked *against the filing*). Returns {} when nothing
    matches, rather than guessing.
    """
    results = doc.get("retrieval_results") or []
    scoped = [c for c in results if c.get("fiscal_label") == fiscal_label] or results
    filings = [c for c in scoped if c.get("doc_type", "").lower() != "transcript"]
    return (filings or scoped or [{}])[0]


def _extract_claims(doc: dict) -> list[ClaimEvidence]:
    """
    Claim-level evidence for the run.

    Sourced from numeric_validations — the pipeline's actual claim-verification
    output, where each entry is a verbatim executive statement checked against
    the filed figures. (This previously read decision_log_entries looking for
    entry["type"] == "claim"; DecisionLogEntry has no `type` field and no agent
    ever emitted one, so it always returned an empty list.)

    confidence is the verification verdict, not a model probability: 1.0 when
    the claim matched the filed value, 0.0 when it did not.
    """
    claims: list[ClaimEvidence] = []
    for i, v in enumerate(doc.get("numeric_validations") or []):
        claim_text = v.get("claim") or ""
        if not claim_text:
            continue
        fiscal_label = v.get("source_fiscal_label") or doc.get("quarter", "")
        chunk = _supporting_chunk(doc, fiscal_label)

        detail = (
            f"Metric: {v.get('metric') or 'n/a'}. "
            f"Stated: {v.get('claimed_value')}. "
            f"Filing-derived: {v.get('calculated_value')}. "
            f"Verdict: {'match' if v.get('match') else 'mismatch'}."
        )
        source_text = chunk.get("content") or chunk.get("parent_content") or ""

        claims.append(ClaimEvidence(
            claim_id=f"nv-{i}",
            claim_text=claim_text,
            source_section=chunk.get("section", ""),
            source_paragraph=f"{detail}\n\n{source_text}".strip(),
            confidence=1.0 if v.get("match") else 0.0,
            doc_type=chunk.get("doc_type") or "transcript",
            quarter=fiscal_label,
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