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

      {/* Report text */}
      {data.report && (
        <Section title="Executive Summary">
          <div className="markdown-body">
            <ReactMarkdown>{stripCitationTags(data.report)}</ReactMarkdown>
          </div>
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
    </div>
  );
}
