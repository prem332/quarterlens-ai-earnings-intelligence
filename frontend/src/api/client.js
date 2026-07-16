const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  // Analysis
  runAnalysis: (body) => request("/analysis/run", { method: "POST", body: JSON.stringify(body) }),
  getStatus: (runId) => request(`/analysis/${runId}/status`),
  getAnalysis: (runId) => request(`/analysis/${runId}`),

  // Reports
  listReports: (params = {}) => {
    const qs = new URLSearchParams(
      Object.fromEntries(Object.entries(params).filter(([, v]) => v != null))
    ).toString();
    return request(`/reports${qs ? `?${qs}` : ""}`);
  },
  getReport: (runId) => request(`/reports/${runId}`),
  deleteReport: (runId) => request(`/reports/${runId}`, { method: "DELETE" }),

  // Evidence
  listClaims: (runId) => request(`/evidence/${runId}/claims`),
  getClaim: (runId, claimId) => request(`/evidence/${runId}/claims/${claimId}`),

  // Export
  exportPdf: (runId) => fetch(`${BASE}/export/${runId}/pdf`, { method: "POST" }),
  exportDocx: (runId) => fetch(`${BASE}/export/${runId}/docx`, { method: "POST" }),
};