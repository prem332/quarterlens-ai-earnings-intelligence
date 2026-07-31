import { useState, useEffect, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api/client";

const COMPANIES = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"];

// Quarters that actually exist in the AI Search index, PER COMPANY. Fiscal
// years don't line up across these five, so the union is 9 quarters while each
// company only has 4-5 of them.
//
// This used to be one flat 9-quarter list shown for every company, which let
// you pick a company/quarter pair with zero indexed documents — retrieval then
// returned 0 chunks and the pipeline produced a confident-looking report built
// on nothing. NVDA has no FY2025-* data at all, so "NVDA vs FY2025-Q4" was a
// silently empty comparison, not a real one.
//
// Static because the dataset is fixed (25 filings + 25 transcripts) and a
// lookup endpoint would add a round trip to page load. Regenerate with:
//   ai_search.filter_search("ticker eq '<TICKER>'") -> distinct fiscal_label
const COMPANY_QUARTERS = {
  AAPL:  ["FY2026-Q2", "FY2026-Q1", "FY2025-Q4", "FY2025-Q3", "FY2025-Q2"],
  MSFT:  ["FY2026-Q3", "FY2026-Q2", "FY2026-Q1", "FY2025-Q4", "FY2025-Q3"],
  NVDA:  ["FY2027-Q1", "FY2026-Q4", "FY2026-Q3", "FY2026-Q2", "FY2026-Q1"],
  GOOGL: ["FY2026-Q1", "FY2025-Q4", "FY2025-Q3", "FY2025-Q2", "FY2025-Q1"],
  META:  ["FY2026-Q1", "FY2025-Q4", "FY2025-Q3", "FY2025-Q2"],
};

const AGENTS = ["retrieval","comparison","sentiment","numeric_validation","report"];

// The backend only ever reports overall pending/running/completed/failed — it
// has no per-agent telemetry to poll, so the "current stage" shown here is an
// elapsed-time ESTIMATE, not a live signal. It's deliberately framed as an
// estimate in the UI rather than implying real tracking.
//
// Stage durations below are NOT an even split — they're weighted from real
// measurements. Re-measured 2026-07-31 by profiling each retrieval sub-step
// directly and by reading per-node timing logs off a live production run:
//
//   retrieval  ~3s warm / ~8s cold  (was listed here as 28s — badly wrong;
//              the old figure came from one early run whose retrieval share
//              was inflated by the one-time cross-encoder model load, which
//              is now paid at server startup instead, and by the uncached
//              chunk re-embedding that has since been fixed)
//   comparison ~4s, sentiment ~8s, numeric_validation ~3s
//   report     ~58s — genuinely dominates (CrewAI bull/bear debate + draft +
//              verify = 4-5 sequential LLM calls)
//
// These remain an ESTIMATE driving a progress animation, not live tracking:
// the backend reports only overall pending/running/completed. Keeping them
// roughly honest matters because a wrong retrieval figure makes the UI look
// stuck on retrieval while the real time is being spent in report_agent.
const STAGE_SECONDS = { retrieval: 5, comparison: 4, sentiment: 8, numeric_validation: 3, report: 58 };
const ESTIMATED_TOTAL_SECONDS = Object.values(STAGE_SECONDS).reduce((a, b) => a + b, 0);
const STAGE_CUMULATIVE = AGENTS.reduce((acc, a) => {
  const prev = acc.length ? acc[acc.length - 1] : 0;
  acc.push(prev + STAGE_SECONDS[a]);
  return acc;
}, []);

function AgentProgress({ runId, onDone }) {
  const [pollStatus, setPollStatus] = useState("running");
  const [elapsed, setElapsed] = useState(0);
  const [draftText, setDraftText] = useState("");
  const [draftPhase, setDraftPhase] = useState(null); // null | "drafting" | "verifying"
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

  // Live token stream for report_agent's draft/verify pass — see
  // api/routes/analysis.py's /stream endpoint. This is genuinely the only
  // part of the pipeline that CAN stream: retrieval/comparison/sentiment/
  // numeric all run and finish before report_agent even starts, so no tokens
  // exist to show until then regardless of how this connects. EventSource
  // retries automatically on its own if it connects before the run's queue
  // is registered (a race of a few ms, not something to special-case here).
  //
  // draft_token text is UNVERIFIED — report_agent's verify pass can still
  // delete or rewrite sentences after drafting finishes. "final" event
  // replaces the streamed text with what was actually verified, so what's
  // shown here can visibly change once verification completes; that's
  // expected, not a bug, and is why the phase label changes to "verifying"
  // rather than silently continuing to look done.
  useEffect(() => {
    const es = new EventSource(`/api/analysis/${runId}/stream`);
    es.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "draft_token") {
        setDraftPhase("drafting");
        setDraftText(prev => prev + msg.text);
      } else if (msg.type === "draft_reset") {
        setDraftText("");
      } else if (msg.type === "verifying") {
        setDraftPhase("verifying");
      } else if (msg.type === "final") {
        setDraftText(msg.report);
      } else if (msg.type === "done") {
        es.close();
      }
    };
    es.onerror = () => { /* browser retries automatically; nothing to do */ };
    return () => es.close();
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

      {draftPhase && (
        <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid var(--border)" }}>
          <p style={{ fontSize: 11, fontFamily: "var(--mono)", color: "var(--text-dim)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.04em" }}>
            {draftPhase === "drafting" ? "Drafting… (unverified)" : "Verifying citations…"}
          </p>
          <div
            className="mono"
            style={{
              fontSize: 12, lineHeight: 1.6, color: "var(--text-dim)",
              maxHeight: 220, overflowY: "auto", whiteSpace: "pre-wrap",
              background: "var(--bg)", border: "1px solid var(--border)",
              borderRadius: "var(--radius)", padding: 12,
            }}
          >
            {draftText}
          </div>
        </div>
      )}
    </div>
  );
}

export default function NewAnalysis() {
  const navigate = useNavigate();
  const [params] = useSearchParams();

  // Defaults are derived from COMPANY_QUARTERS, never hardcoded. The previous
  // literals ("Q2_2025" / ["Q1_2025"]) matched no <option>, so the select
  // rendered its first option while state still held the stale legacy string —
  // the form showed one quarter and the backend analysed a different one.
  const initialCompany = COMPANIES.includes(params.get("company")) ? params.get("company") : "AAPL";

  const [company, setCompany]   = useState(initialCompany);
  const [quarter, setQuarter]   = useState(COMPANY_QUARTERS[initialCompany][0]);
  const [compQ, setCompQ]       = useState(COMPANY_QUARTERS[initialCompany].slice(1, 2));
  const [query, setQuery]       = useState("");
  const [runId, setRunId]       = useState(null);
  const [error, setError]       = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const quarters = COMPANY_QUARTERS[company];

  // Switching company must reset both selections — the previous company's
  // quarters are frequently not valid for the new one.
  function changeCompany(next) {
    setCompany(next);
    setQuarter(COMPANY_QUARTERS[next][0]);
    setCompQ(COMPANY_QUARTERS[next].slice(1, 2));
  }

  // Keep the primary quarter out of the comparison set.
  function changeQuarter(next) {
    setQuarter(next);
    setCompQ(prev => prev.filter(q => q !== next));
  }

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
              <select value={company} onChange={e => changeCompany(e.target.value)}>
                {COMPANIES.map(c => <option key={c}>{c}</option>)}
              </select>
            </div>
            <div className="field">
              <label>Quarter</label>
              <select value={quarter} onChange={e => changeQuarter(e.target.value)}>
                {quarters.map(q => <option key={q}>{q}</option>)}
              </select>
            </div>
          </div>

          <div className="field">
            <label>Compare against (up to 3)</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {quarters.filter(q => q !== quarter).map(q => (
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
