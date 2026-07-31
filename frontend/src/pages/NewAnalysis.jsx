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

// Backend node names -> the labels above. supervisor_init/finalize are real
// nodes but carry no user-meaningful work (both measured at ~0-1s), so they
// aren't shown as stages.
const NODE_TO_AGENT = {
  retrieval_agent: "retrieval",
  comparison_agent: "comparison",
  sentiment_agent: "sentiment",
  numeric_validation_agent: "numeric_validation",
  report_agent: "report",
};

// Stage progress is now LIVE, not estimated. graph/build_graph.py's _traced
// wrapper emits stage_start/stage_end on the same SSE stream the draft tokens
// use, so each row below reflects what the pipeline is actually doing and
// reports the real per-stage duration when it finishes.
//
// This replaces a hardcoded elapsed-time animation, which was actively
// misleading: it ticked stages off on a fixed schedule regardless of reality,
// so a run whose numeric_validation genuinely took 53s still showed that stage
// completing after its guessed 3s. Anyone timing the UI with a stopwatch was
// measuring the animation, not the pipeline.

function AgentProgress({ runId, onDone }) {
  const [pollStatus, setPollStatus] = useState("running");
  const [elapsed, setElapsed] = useState(0);
  const [draftText, setDraftText] = useState("");
  const [draftPhase, setDraftPhase] = useState(null); // null | "drafting" | "verifying"
  const [activeStage, setActiveStage] = useState(null);   // stage currently running
  const [doneStages, setDoneStages] = useState({});       // stage -> real seconds taken
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
      if (msg.type === "stage_start") {
        const a = NODE_TO_AGENT[msg.stage];
        if (a) setActiveStage(a);
      } else if (msg.type === "stage_end" || msg.type === "stage_error") {
        const a = NODE_TO_AGENT[msg.stage];
        if (a) {
          setDoneStages(prev => ({ ...prev, [a]: msg.seconds }));
          // comparison and sentiment run in parallel, so a stage_end does not
          // imply the next stage has begun — clear only if this was the one
          // being shown as active.
          setActiveStage(prev => (prev === a ? null : prev));
        }
      } else if (msg.type === "draft_token") {
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

  return (
    <div className="card" style={{ marginTop: 24 }}>
      <p style={{ fontSize: 12, fontFamily: "var(--mono)", color: "var(--text-dim)", marginBottom: 16, textTransform: "uppercase", letterSpacing: "0.04em" }}>
        Pipeline — {pollStatus} ({elapsed}s elapsed — live stage tracking)
      </p>
      {AGENTS.map((a) => {
        const seconds = doneStages[a];
        const st = seconds !== undefined ? "done"
          : activeStage === a ? "running"
          : pollStatus === "completed" ? "done"
          : null;
        return (
          <div key={a} style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 0", borderBottom: "1px solid var(--border)" }}>
            <span style={{ width: 16, textAlign: "center" }}>
              {st === "done" ? <span className="check">✓</span>
               : st === "running" ? <span className="spinner" />
               : <span style={{ color: "var(--border-hi)" }}>·</span>}
            </span>
            <span className="mono" style={{ fontSize: 13, color: st ? "var(--text-hi)" : "var(--text-dim)", flex: 1 }}>
              {a.replace("_", " ")}
            </span>
            {seconds !== undefined && (
              <span className="mono" style={{ fontSize: 12, color: "var(--text-dim)" }}>
                {seconds}s
              </span>
            )}
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
  // No comparison quarter selected by default. Picking one runs
  // comparison_agent, which costs a full LLM call -- measured at ~3.8s, the
  // longest branch of the parallel group and therefore the whole group's
  // duration. That is worth paying when the question is actually comparative
  // ("how did X change vs last quarter"), but the previous default pre-selected
  // a quarter, so every run paid it even for a plain "what was net income this
  // quarter". Users who want a comparison still click one.
  //
  // UI default only: the evaluation harness supplies its own comparison_quarters
  // from the claim files, so this does not touch any locked metric.
  const [compQ, setCompQ]       = useState([]);
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
    setCompQ([]);   // clear, don't re-select — see the useState default above
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
