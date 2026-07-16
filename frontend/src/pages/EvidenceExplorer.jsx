import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "../api/client";

export default function EvidenceExplorer() {
  const { runId, claimId } = useParams();
  const navigate = useNavigate();

  const [claims, setClaims]         = useState([]);
  const [selected, setSelected]     = useState(null);
  const [loading, setLoading]       = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    api.listClaims(runId)
      .then(setClaims)
      .catch(() => setClaims([]))
      .finally(() => setLoading(false));
  }, [runId]);

  useEffect(() => {
    if (claimId) {
      setDetailLoading(true);
      api.getClaim(runId, claimId)
        .then(setSelected)
        .catch(() => setSelected(null))
        .finally(() => setDetailLoading(false));
    }
  }, [claimId]);

  function selectClaim(claim) {
    navigate(`/report/${runId}/evidence/${claim.claim_id}`);
    setSelected(claim);
  }

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
        <button className="btn btn-ghost" style={{ padding: "4px 10px", fontSize: 12 }}
          onClick={() => navigate(`/report/${runId}`)}>
          ← Report
        </button>
        <h1 className="page-title" style={{ margin: 0 }}>Evidence Explorer</h1>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", gap: 16, alignItems: "start" }}>
        {/* Claim list */}
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
            <p style={{ fontSize: 11, fontFamily: "var(--mono)", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              {loading ? "Loading…" : `${claims.length} claims`}
            </p>
          </div>
          {loading ? (
            <p className="dim" style={{ padding: 16 }}><span className="spinner" /></p>
          ) : claims.length === 0 ? (
            <p className="dim" style={{ padding: 16, fontSize: 13 }}>No claims found for this run.</p>
          ) : (
            <div style={{ overflowY: "auto", maxHeight: "calc(100vh - 200px)" }}>
              {claims.map(c => (
                <div
                  key={c.claim_id}
                  onClick={() => selectClaim(c)}
                  style={{
                    padding: "12px 16px",
                    borderBottom: "1px solid var(--border)",
                    cursor: "pointer",
                    background: selected?.claim_id === c.claim_id ? "var(--accent-dim)" : "transparent",
                    transition: "background 0.1s",
                  }}
                >
                  <p style={{ fontSize: 13, lineHeight: 1.5, marginBottom: 6 }}>{c.claim_text}</p>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    <span className="badge" style={{ background: "var(--bg)", color: "var(--text-dim)" }}>
                      {c.doc_type}
                    </span>
                    <span className="mono dim" style={{ fontSize: 11, lineHeight: "18px" }}>
                      conf: {(c.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Detail panel */}
        <div className="card">
          {!selected && !detailLoading ? (
            <p className="dim" style={{ fontSize: 13 }}>Select a claim to see its source.</p>
          ) : detailLoading ? (
            <p className="dim"><span className="spinner" /></p>
          ) : selected ? (
            <>
              <p style={{ fontWeight: 500, fontSize: 15, marginBottom: 16, lineHeight: 1.5 }}>{selected.claim_text}</p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 20 }}>
                {[
                  { label: "Section",    value: selected.source_section },
                  { label: "Doc type",   value: selected.doc_type },
                  { label: "Confidence", value: `${(selected.confidence * 100).toFixed(0)}%` },
                ].map(({ label, value }) => (
                  <div key={label}>
                    <p style={{ fontSize: 11, fontFamily: "var(--mono)", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 4 }}>{label}</p>
                    <p className="mono" style={{ fontSize: 13 }}>{value || "—"}</p>
                  </div>
                ))}
              </div>
              <div style={{ background: "var(--bg)", borderRadius: "var(--radius)", padding: "14px 16px", borderLeft: "3px solid var(--accent)" }}>
                <p style={{ fontSize: 11, fontFamily: "var(--mono)", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 8 }}>Source paragraph</p>
                <p style={{ fontSize: 13, lineHeight: 1.75, color: "var(--text)" }}>{selected.source_paragraph || "(Source text not captured)"}</p>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
