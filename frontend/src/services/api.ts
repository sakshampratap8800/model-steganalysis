import axios from "axios";
import type {
  ScanRecord,
  ScanReport,
  ScanStatusResponse,
  CreateScanResponse,
  PaginatedScans,
} from "../types";

// ─── Axios instance ───────────────────────────────────────────────────────────
const api = axios.create({
  baseURL: "/api",
  timeout: 60_000,
  headers: {
    "Content-Type": "application/json",
  },
});

// ─── Response interceptor for error normalisation ─────────────────────────────
api.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    if (axios.isAxiosError(error)) {
      const msg =
        (error.response?.data as { detail?: string })?.detail ??
        error.message ??
        "Unknown API error";
      return Promise.reject(new Error(msg));
    }
    return Promise.reject(error);
  }
);

// ─── Scans API ────────────────────────────────────────────────────────────────
export const scansApi = {
  /**
   * POST /api/scans — Upload a model file and start a scan.
   * Returns the new scan record with queued status.
   */
  create: async (file: File): Promise<CreateScanResponse> => {
    const form = new FormData();
    form.append("file", file);
    const response = await api.post<CreateScanResponse>("/scans", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  },

  /**
   * GET /api/scans/{scan_id}/status — Lightweight polling endpoint.
   */
  getStatus: async (scanId: string): Promise<ScanStatusResponse> => {
    const response = await api.get<ScanStatusResponse>(
      `/scans/${scanId}/status`
    );
    return response.data;
  },

  /**
   * GET /api/scans/{scan_id}/report — Full scan report (only valid when complete).
   */
  getReport: async (scanId: string): Promise<ScanReport> => {
    const response = await api.get<ScanReport>(`/scans/${scanId}/report`);
    return response.data;
  },

  /**
   * GET /api/scans/{scan_id} — Full scan record with profile and report fields.
   */
  getScan: async (scanId: string): Promise<ScanRecord> => {
    const response = await api.get<ScanRecord>(`/scans/${scanId}`);
    return response.data;
  },

  /**
   * GET /api/scans — Paginated list of scans.
   */
  list: async (
    skip = 0,
    limit = 50
  ): Promise<PaginatedScans> => {
    const response = await api.get<PaginatedScans>("/scans", {
      params: { skip, limit },
    });
    return response.data;
  },
};

export default api;
