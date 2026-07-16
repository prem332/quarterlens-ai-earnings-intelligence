import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "../api/client";

function Section({ title, children }) {
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <p style={{ fontSize: 11, fontFamily: "var(--mono)", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 14 }}>
        {title}
      </p>
      {children}
    </div>
  );
}

export default function AnalysisReport() {
  const { runId } = useParams();
  const navigate  = useNavigate();
  const [data, setData]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState(null);

  useEffect(() => {
    api.getReport(runId)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId]);

  async function download(type) {
    const res = type === "pdf" ? await api.exportPdf(runId) : await api.exportDocx(runId);
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `quarterlens_${runId}.${type}`; a.click();
    URL.revokeObjectURL(url);
  }

  if (loading) return <p className="dim"><span className="spinner" /> Loading…</p>;
  if (error)   return <p className="error-msg">{error}</p>;
  if (!data)   return null;

  const passRate = data.numeric_validations.length > 0
    ? (data.numeric_validations.filter(v => v.verified).length / data.numeric_validations.length * 100).toFixed(0)
    : null;

  return (
    <div>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
        <div>
          <h1 className="page-title" style={{ marginBottom: 4 }}>
            <span className="mono">{data.company}</span> · {data.quarter}
          </h1>
          <p className="dim mono" style={{ fontSize: 12 }}>run/{runId.slice(0, 8)}</p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-ghost" onClick={() => navigate(`/report/${runId}/evidence`)}>
            Evidence Explorer
          </button>
          <button className="btn btn-ghost" onClick={() => download("pdf")}>Export PDF</button>
          <button className="btn btn-ghost" onClick={() => download("docx")}>Export DOCX</button>
        </div>
      </div>

      {/* Stats bar */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 20 }}>
        {[
          { label: "Retrieval chunks",    value: data.retrieval_results.length },
          { label: "Numeric checks",      value: data.numeric_validations.length },
          { label: "Pass rate",           value: passRate != null ? `${passRate}%` : "—" },
          { label: "Sentiment signals",   value: data.sentiment_scores.length },
        ].map(({ label, value }) => (
          <div key={label} className="card" style={{ textAlign: "center" }}>
            <p style={{ fontSize: 22, fontWeight: 600, fontFamily: "var(--mono)", color: "var(--text-hi)" }}>{value}</p>
            <p style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 4 }}>{label}</p>
          </div>
        ))}
      </div>

      {/* Report text */}
      {data.report && (
        <Section title="Executive Summary">
          <p style={{ fontSize: 14, lineHeight: 1.75, whiteSpace: "pre-wrap" }}>{data.report}</p>
        </Section>
      )}

      {/* Numeric validations */}
      {data.numeric_validations.length > 0 && (
        <Section title="Numeric Validation">
          <table>
            <thead><tr><th></th><th>Claim</th><th>Filed</th><th>Stated</th><th>Δ%</th></tr></thead>
            <tbody>
              {data.numeric_validations.map((v, i) => (
                <tr key={i}>
                  <td style={{ width: 20 }}>
                    {v.verified ? <span className="check">✓</span> : <span className="cross">✗</span>}
                  </td>
                  <td>{v.claim}</td>
                  <td className="mono">{v.filed_value ?? "—"}</td>
                  <td className="mono">{v.stated_value ?? "—"}</td>
                  <td className="mono" style={{ color: v.delta_pct != null && Math.abs(v.delta_pct) > 1 ? "var(--red)" : "var(--text-dim)" }}>
                    {v.delta_pct != null ? `${v.delta_pct > 0 ? "+" : ""}${v.delta_pct.toFixed(1)}%` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      )}

      {/* Comparison findings */}
      {data.comparison_findings.length > 0 && (
        <Section title="Guidance & Language Changes">
          {data.comparison_findings.map((f, i) => (
            <div key={i} style={{ padding: "12px 0", borderBottom: "1px solid var(--border)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontWeight: 500 }}>{f.topic}</span>
                {f.shift_detected
                  ? <span className="badge badge-failed">shift detected</span>
                  : <span className="badge badge-completed">no shift</span>}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, fontSize: 13 }}>
                <div>
                  <p className="dim" style={{ fontSize: 11, fontFamily: "var(--mono)", marginBottom: 4 }}>CURRENT</p>
                  <p style={{ lineHeight: 1.5 }}>{f.current}</p>
                </div>
                <div>
                  <p className="dim" style={{ fontSize: 11, fontFamily: "var(--mono)", marginBottom: 4 }}>{f.quarter}</p>
                  <p style={{ lineHeight: 1.5 }}>{f.prior}</p>
                </div>
              </div>
            </div>
          ))}
        </Section>
      )}

      {/* Sentiment */}
      {data.sentiment_scores.length > 0 && (
        <Section title="Sentiment Signals (FinBERT)">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {data.sentiment_scores.map((s, i) => (
              <div key={i} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                <span
                  className="badge"
                  style={{
                    background: s.label === "positive" ? "#0d2e20" : s.label === "negative" ? "#2e0d0d" : "#1a1f2e",
                    color: s.label === "positive" ? "var(--green)" : s.label === "negative" ? "var(--red)" : "var(--text-dim)",
                    flexShrink: 0,
                    marginTop: 2,
                  }}
                >
                  {s.label}
                </span>
                <p style={{ fontSize: 13, lineHeight: 1.5 }}>"{s.excerpt}"</p>
                <span className="mono dim" style={{ fontSize: 11, flexShrink: 0, marginTop: 3 }}>{(s.score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}
