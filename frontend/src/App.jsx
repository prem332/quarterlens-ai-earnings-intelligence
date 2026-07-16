import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import NewAnalysis from "./pages/NewAnalysis";
import AnalysisReport from "./pages/AnalysisReport";
import ReportHistory from "./pages/ReportHistory";
import EvidenceExplorer from "./pages/EvidenceExplorer";

const NAV = [
  { to: "/",          label: "Dashboard"  },
  { to: "/new",       label: "New Analysis" },
  { to: "/history",   label: "Report History" },
];

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-logo">Quarter<span>Lens</span> AI</div>
        <nav>
          {NAV.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) => isActive ? "active" : ""}
            >
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <main className="main">
        <Routes>
          <Route path="/"                              element={<Dashboard />} />
          <Route path="/new"                           element={<NewAnalysis />} />
          <Route path="/history"                       element={<ReportHistory />} />
          <Route path="/report/:runId"                 element={<AnalysisReport />} />
          <Route path="/report/:runId/evidence"        element={<EvidenceExplorer />} />
          <Route path="/report/:runId/evidence/:claimId" element={<EvidenceExplorer />} />
          <Route path="*"                              element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
