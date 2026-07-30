import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import { api } from "../api/client";

// report_agent tags every retained sentence [FILING] / [TRANSCRIPT] so verify
// can check provenance — real, load-bearing data, not decoration. Stripped
// here for display only; the underlying text (and the export endpoints,
// which read the same field) is untouched.
function stripCitationTags(text) {
  return (text || "").replace(/\s*\[(FILING|TRANSCRIPT)\]/g, "");
}

// collapsible=true renders as a toggle (closed by default) instead of always-open.
// Executive Summary and Numeric Validation are the primary read — always shown.
// Guidance & Language Changes and Sentiment are the underlying evidence for that
// summary, useful to verify but noisy to show inline every time, so they start
// collapsed and expand on click.
function Section({ title, count, children, collapsible = false, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div
        style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          cursor: collapsible ? "pointer" : "default",
          marginBottom: open ? 14 : 0,
        }}
        onClick={collapsible ? () => setOpen(o => !o) : undefined}
      >
        <p style={{ fontSize: 11, fontFamily: "var(--mono)", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
          {title}{count != null ? ` (${count})` : ""}
        </p>
        {collapsible && (
          <span className="dim mono" style={{ fontSize: 11 }}>{open ? "▲ collapse" : "▼ expand"}</span>
        )}
      </div>
      {open && children}
    </div>
  );
}

export default function AnalysisReport() {
  const { runId } = useParams();
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
    ? (data.numeric_validations.filter(v => v.match).length / data.numeric_validations.length * 100).toFixed(0)
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
          <div className="markdown-body">
            <ReactMarkdown>{stripCitationTags(data.report)}</ReactMarkdown>
          </div>
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
                    {v.match ? <span className="check">✓</span> : <span className="cross">✗</span>}
                  </td>
                  <td>{stripCitationTags(v.claim)}</td>
                  <td className="mono">{v.calculated_value ?? "—"}</td>
                  <td className="mono">{v.claimed_value ?? "—"}</td>
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
        <Section title="Guidance & Language Changes" count={data.comparison_findings.length}
                 collapsible defaultOpen={false}>
          {data.comparison_findings.map((f, i) => (
            <div key={i} style={{ padding: "12px 0", borderBottom: "1px solid var(--border)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontWeight: 500 }}>{f.topic}</span>
                {f.shift_detected
                  ? <span className="badge badge-failed">shift detected</span>
                  : <span className="badge badge-completed">no shift</span>}
              </div>
              {f.shift_description && (
                <p className="dim" style={{ fontSize: 13, marginBottom: 10 }}>{stripCitationTags(f.shift_description)}</p>
              )}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, fontSize: 13 }}>
                <div>
                  <p className="dim" style={{ fontSize: 11, fontFamily: "var(--mono)", marginBottom: 4 }}>CURRENT</p>
                  <p style={{ lineHeight: 1.5 }}>{stripCitationTags(f.current_language)}</p>
                </div>
                <div>
                  {Object.entries(f.prior_language || {}).map(([fiscalLabel, excerpt]) => (
                    <div key={fiscalLabel} style={{ marginBottom: 8 }}>
                      <p className="dim" style={{ fontSize: 11, fontFamily: "var(--mono)", marginBottom: 4 }}>{fiscalLabel}</p>
                      <p style={{ lineHeight: 1.5 }}>{stripCitationTags(excerpt)}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </Section>
      )}

      {/* Sentiment */}
      {data.sentiment_scores.length > 0 && (
        <Section title="Sentiment Signals (FinBERT)" count={data.sentiment_scores.length}
                 collapsible defaultOpen={false}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {data.sentiment_scores.map((s, i) => (
              <div key={i} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                <span
                  className="badge"
                  style={{
                    background: s.label === "positive" ? "#0d2e20" : s.label === "negative" ? "#2e0d0d" : "#2a2510",
                    color: s.label === "positive" ? "var(--green)" : s.label === "negative" ? "var(--red)" : "var(--yellow)",
                    flexShrink: 0,
                    marginTop: 2,
                  }}
                >
                  {s.label}
                </span>
                <p style={{ fontSize: 13, lineHeight: 1.5 }}>"{stripCitationTags(s.passage)}"</p>
                <span className="mono dim" style={{ fontSize: 11, flexShrink: 0, marginTop: 3 }}>{(s.score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}
