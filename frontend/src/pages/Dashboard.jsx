import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";

const COMPANIES = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"];

function StatusBadge({ status }) {
  return <span className={`badge badge-${status}`}>{status}</span>;
}

export default function Dashboard() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    api.listReports({ limit: 5 })
      .then(setReports)
      .catch(() => setReports([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
        <div>
          <h1 className="page-title" style={{ marginBottom: 4 }}>Dashboard</h1>
          <p className="dim" style={{ fontSize: 13 }}>5 companies · 5 quarters · SEC EDGAR cross-verification</p>
        </div>
        <button className="btn btn-primary" onClick={() => navigate("/new")}>
          + New Analysis
        </button>
      </div>

      {/* Quick-launch grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 32 }}>
        {COMPANIES.map(c => (
          <button
            key={c}
            className="card btn"
            style={{ justifyContent: "center", fontFamily: "var(--mono)", fontSize: 14, cursor: "pointer" }}
            onClick={() => navigate(`/new?company=${c}`)}
          >
            {c}
          </button>
        ))}
      </div>

      {/* Recent analyses */}
      <div className="card">
        <h2 style={{ fontSize: 13, fontFamily: "var(--mono)", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 16 }}>
          Recent Analyses
        </h2>
        {loading ? (
          <p className="dim"><span className="spinner" /></p>
        ) : reports.length === 0 ? (
          <p className="dim" style={{ fontSize: 13 }}>No analyses yet. Run your first one above.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Company</th><th>Quarter</th><th>Status</th><th>Created</th><th></th>
              </tr>
            </thead>
            <tbody>
              {reports.map(r => (
                <tr key={r.run_id} style={{ cursor: "pointer" }} onClick={() => navigate(`/report/${r.run_id}`)}>
                  <td className="mono">{r.company}</td>
                  <td>{r.quarter}</td>
                  <td><StatusBadge status={r.status} /></td>
                  <td className="dim">{new Date(r.created_at).toLocaleDateString()}</td>
                  <td style={{ textAlign: "right" }}>
                    <span style={{ color: "var(--accent)", fontSize: 13 }}>View →</span>
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
