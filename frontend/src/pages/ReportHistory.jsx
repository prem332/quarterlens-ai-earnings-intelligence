import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";

const COMPANIES = ["", "AAPL", "MSFT", "NVDA", "GOOGL", "META"];

export default function ReportHistory() {
  const navigate = useNavigate();
  const [reports, setReports]     = useState([]);
  const [loading, setLoading]     = useState(true);
  const [company, setCompany]     = useState("");
  const [quarter, setQuarter]     = useState("");
  const [deleting, setDeleting]   = useState(null);

  function load() {
    setLoading(true);
    api.listReports({ company: company || undefined, quarter: quarter || undefined, limit: 50 })
      .then(setReports)
      .catch(() => setReports([]))
      .finally(() => setLoading(false));
  }

  useEffect(() => { load(); }, [company, quarter]);

  async function remove(runId) {
    setDeleting(runId);
    try { await api.deleteReport(runId); setReports(r => r.filter(x => x.run_id !== runId)); }
    catch { /* ignore */ }
    finally { setDeleting(null); }
  }

  return (
    <div>
      <h1 className="page-title">Report History</h1>

      <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
        <select value={company} onChange={e => setCompany(e.target.value)} style={{ width: 130 }}>
          {COMPANIES.map(c => <option key={c} value={c}>{c || "All companies"}</option>)}
        </select>
        <input
          placeholder="Quarter, e.g. Q2_2025"
          value={quarter}
          onChange={e => setQuarter(e.target.value)}
          style={{ width: 180 }}
        />
      </div>

      <div className="card">
        {loading ? (
          <p className="dim"><span className="spinner" /> Loading…</p>
        ) : reports.length === 0 ? (
          <p className="dim">No reports match these filters.</p>
        ) : (
          <table>
            <thead>
              <tr><th>Company</th><th>Quarter</th><th>Status</th><th>Date</th><th>Summary</th><th></th></tr>
            </thead>
            <tbody>
              {reports.map(r => (
                <tr key={r.run_id}>
                  <td className="mono">{r.company}</td>
                  <td>{r.quarter}</td>
                  <td><span className={`badge badge-${r.status}`}>{r.status}</span></td>
                  <td className="dim">{new Date(r.created_at).toLocaleDateString()}</td>
                  <td style={{ maxWidth: 280 }}>
                    <p style={{ fontSize: 12, color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {r.report_snippet || "—"}
                    </p>
                  </td>
                  <td>
                    <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
                      <button className="btn btn-ghost" style={{ padding: "4px 10px", fontSize: 12 }}
                        onClick={() => navigate(`/report/${r.run_id}`)}>
                        View
                      </button>
                      <button className="btn btn-danger" style={{ padding: "4px 10px", fontSize: 12 }}
                        disabled={deleting === r.run_id}
                        onClick={() => remove(r.run_id)}>
                        {deleting === r.run_id ? "…" : "Delete"}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
