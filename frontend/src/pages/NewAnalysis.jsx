import { useState, useEffect, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api/client";

const COMPANIES = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"];
const QUARTERS = ["FY2026-Q2","FY2026-Q1","FY2025-Q4","FY2025-Q3","FY2025-Q2","FY2025-Q1","FY2027-Q1","FY2026-Q4","FY2026-Q3"];

const AGENTS = ["retrieval","comparison","sentiment","numeric_validation","report"];

// The backend only ever reports overall pending/running/completed/failed — it
// has no per-agent telemetry to poll, so the "current stage" shown here is an
// elapsed-time ESTIMATE, not a live signal. It's deliberately framed as an
// estimate in the UI rather than implying real tracking.
//
// Stage durations below are NOT an even split — they're weighted using the
// real decision_log_entries.latency_ms from an actual completed run
// (retrieval 28.1s / comparison 3.9s / sentiment 8.4s / numeric 2.5s /
// report 58.2s, ~110s total). report_agent (CrewAI bull/bear debate + draft +
// verify — 4 sequential LLM calls) dominates; numeric_validation is
// consistently fast. This is a single real measurement, not an average across
// many runs, so treat it as a much better guess than an even split, not a
// precise profile — retrieval's share is also somewhat inflated whenever it's
// the first request since a server restart (one-time model-load cost).
const STAGE_SECONDS = { retrieval: 28, comparison: 4, sentiment: 8, numeric_validation: 3, report: 58 };
const ESTIMATED_TOTAL_SECONDS = Object.values(STAGE_SECONDS).reduce((a, b) => a + b, 0);
const STAGE_CUMULATIVE = AGENTS.reduce((acc, a) => {
  const prev = acc.length ? acc[acc.length - 1] : 0;
  acc.push(prev + STAGE_SECONDS[a]);
  return acc;
}, []);

function AgentProgress({ runId, onDone }) {
  const [pollStatus, setPollStatus] = useState("running");
  const [elapsed, setElapsed] = useState(0);
  const pollTimerRef = useRef(null);
  const tickTimerRef = useRef(null);
  const startRef = useRef(Date.now());

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const s = await api.getStatus(runId);
        if (cancelled) return;
        setPollStatus(s.status);
        if (s.status === "completed" || s.status === "failed") {
          clearInterval(pollTimerRef.current);
          clearInterval(tickTimerRef.current);
          onDone(s.status, s.error);
        }
      } catch { /* transient — keep polling */ }
    }
    poll();
    pollTimerRef.current = setInterval(poll, 2000);
    tickTimerRef.current = setInterval(
      () => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)),
      1000
    );
    return () => {
      cancelled = true;
      clearInterval(pollTimerRef.current);
      clearInterval(tickTimerRef.current);
    };
  }, [runId]);

  // First stage whose cumulative threshold hasn't been reached yet; once
  // elapsed exceeds every threshold, stay on the last stage (report).
  const firstUnreached = STAGE_CUMULATIVE.findIndex(threshold => elapsed < threshold);
  const estimatedIdx = firstUnreached === -1 ? AGENTS.length - 1 : firstUnreached;

  return (
    <div className="card" style={{ marginTop: 24 }}>
      <p style={{ fontSize: 12, fontFamily: "var(--mono)", color: "var(--text-dim)", marginBottom: 16, textTransform: "uppercase", letterSpacing: "0.04em" }}>
        Pipeline — {pollStatus} ({elapsed}s elapsed — stages below are an estimate, not live tracking)
      </p>
      {AGENTS.map((a, i) => {
        const st = pollStatus === "completed" ? "done"
          : i < estimatedIdx ? "done"
          : i === estimatedIdx ? "running"
          : null;
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
