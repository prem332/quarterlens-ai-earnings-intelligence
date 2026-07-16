import { useState, useEffect, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api/client";

const COMPANIES = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"];
const QUARTERS = ["FY2026-Q2","FY2026-Q1","FY2025-Q4","FY2025-Q3","FY2025-Q2","FY2025-Q1","FY2027-Q1","FY2026-Q4","FY2026-Q3"];

const AGENTS = ["retrieval","comparison","sentiment","numeric_validation","report"];

function AgentProgress({ runId, onDone }) {
  const [statuses, setStatuses] = useState({});
  const [pollStatus, setPollStatus] = useState("running");
  const timerRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const s = await api.getStatus(runId);
        if (cancelled) return;
        setPollStatus(s.status);
        if (s.status === "completed" || s.status === "failed") {
          clearInterval(timerRef.current);
          onDone(s.status, s.error);
        }
        // Simulate per-agent progress from overall status
        if (s.status === "running") {
          setStatuses(prev => {
            const next = { ...prev };
            const done = Object.keys(prev).filter(k => prev[k] === "done").length;
            const idx = Math.min(done, AGENTS.length - 1);
            if (!next[AGENTS[idx]]) next[AGENTS[idx]] = "running";
            return next;
          });
        }
        if (s.status === "completed") {
          const all = {};
          AGENTS.forEach(a => all[a] = "done");
          setStatuses(all);
        }
      } catch { /* transient — keep polling */ }
    }
    poll();
    timerRef.current = setInterval(poll, 2000);
    return () => { cancelled = true; clearInterval(timerRef.current); };
  }, [runId]);

  return (
    <div className="card" style={{ marginTop: 24 }}>
      <p style={{ fontSize: 12, fontFamily: "var(--mono)", color: "var(--text-dim)", marginBottom: 16, textTransform: "uppercase", letterSpacing: "0.04em" }}>
        Pipeline — {pollStatus}
      </p>
      {AGENTS.map(a => {
        const st = statuses[a];
        return (
          <div key={a} style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 0", borderBottom: "1px solid var(--border)" }}>
            <span style={{ width: 16, textAlign: "center" }}>
              {st === "done" ? <span className="check">✓</span>
               : st === "running" ? <span className="spinner" />
               : <span style={{ color: "var(--border-hi)" }}>·</span>}
            </span>
            <span className="mono" style={{ fontSize: 13, color: st ? "var(--text-hi)" : "var(--text-dim)" }}>
              {a.replace("_", " ")}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function NewAnalysis() {
  const navigate = useNavigate();
  const [params] = useSearchParams();

  const [company, setCompany]   = useState(params.get("company") || "AAPL");
  const [quarter, setQuarter]   = useState("Q2_2025");
  const [compQ, setCompQ]       = useState(["Q1_2025"]);
  const [query, setQuery]       = useState("");
  const [runId, setRunId]       = useState(null);
  const [error, setError]       = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const toggleCompQ = (q) => setCompQ(prev =>
    prev.includes(q) ? prev.filter(x => x !== q) : prev.length < 3 ? [...prev, q] : prev
  );

  async function submit() {
    setError(null);
    setSubmitting(true);
    try {
      const res = await api.runAnalysis({
        company,
        quarter,
        comparison_quarters: compQ,
        query: query.trim() || undefined,
      });
      setRunId(res.run_id);
    } catch (e) {
      setError(e.message);
      setSubmitting(false);
    }
  }

  function onDone(status, err) {
    if (status === "completed") {
      navigate(`/report/${runId}`);
    } else {
      setError(err || "Pipeline failed");
      setRunId(null);
      setSubmitting(false);
    }
  }

  return (
    <div style={{ maxWidth: 600 }}>
      <h1 className="page-title">New Analysis</h1>

      {!runId ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <div className="field">
              <label>Company</label>
              <select value={company} onChange={e => setCompany(e.target.value)}>
                {COMPANIES.map(c => <option key={c}>{c}</option>)}
              </select>
            </div>
            <div className="field">
              <label>Quarter</label>
              <select value={quarter} onChange={e => setQuarter(e.target.value)}>
                {QUARTERS.map(q => <option key={q}>{q}</option>)}
              </select>
            </div>
          </div>

          <div className="field">
            <label>Compare against (up to 3)</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {QUARTERS.filter(q => q !== quarter).map(q => (
                <button
                  key={q}
                  className={`btn ${compQ.includes(q) ? "btn-primary" : "btn-ghost"}`}
                  style={{ padding: "4px 10px", fontSize: 12, fontFamily: "var(--mono)" }}
                  onClick={() => toggleCompQ(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          <div className="field">
            <label>Query (optional)</label>
            <textarea
              rows={3}
              placeholder="Focus the analysis, e.g. 'Verify revenue growth claims and flag guidance changes'"
              value={query}
              onChange={e => setQuery(e.target.value)}
              style={{ resize: "vertical" }}
            />
          </div>

          {error && <p className="error-msg">{error}</p>}

          <button className="btn btn-primary" onClick={submit} disabled={submitting}>
            {submitting ? <><span className="spinner" /> Starting…</> : "Run Analysis"}
          </button>
        </div>
      ) : (
        <AgentProgress runId={runId} onDone={onDone} />
      )}
    </div>
  );
}
