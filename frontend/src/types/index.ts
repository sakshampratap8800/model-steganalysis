// ─── Scan Status ─────────────────────────────────────────────────────────────
export type ScanStatus = "queued" | "running" | "complete" | "failed";

// ─── Model Profile ────────────────────────────────────────────────────────────
export interface TensorProfile {
  name: string;
  shape: number[];
  dtype: string;
  parameter_count: number;
  min_val: number;
  max_val: number;
  mean: number;
  std_dev: number;
  variance: number;
  median: number;
  sparsity: number;
}

export interface ModelProfile {
  scan_id: string;
  filename: string;
  sha256: string;
  format: string;
  architecture: string;
  architecture_hint: string;
  dtype: string;
  tensor_count: number;
  total_parameter_count: number;
  tensors: TensorProfile[];
}

// ─── Verdict ─────────────────────────────────────────────────────────────────
export type VerdictLevel = "BENIGN" | "SUSPICIOUS" | "HIGH_RISK" | "UNKNOWN";

// ─── Evidence Families ────────────────────────────────────────────────────────
// Family values: "bit_representation" | "trojan_signature" | "structural" |
//                "reference_deviation" | "behavioral" | "model_xray"
export interface EvidenceFamilyScore {
  family: string;
  score: number;     // 0–100
  level: "Low" | "Medium" | "High" | "Unavailable";
  description: string;
  available: boolean;
}

// ─── Flagged Layers ───────────────────────────────────────────────────────────
export interface FlaggedLayer {
  name: string;
  anomaly_score: number;
  primary_evidence: string[];
  shape: number[];
  parameter_count: number;
}

// ─── Scan Report ──────────────────────────────────────────────────────────────
export interface ScanReport {
  scan_id: string;
  model_hash: string;
  filename: string;
  architecture: string;
  format: string;
  dtype: string;
  risk_score: number;         // 0–100
  verdict: VerdictLevel;
  status_message: string;
  evidence_families: EvidenceFamilyScore[];
  flagged_layers: FlaggedLayer[];
  explanation: string;
  behavioral_analysis_available: boolean;
  scan_duration_ms: number;
  scanner_version: string;
  model_profile: ModelProfile | null;
}

// ─── Scan Record (list view) ─────────────────────────────────────────────────
export interface ScanRecord {
  id: string;           // matches backend ScanListItem.id
  filename: string;
  sha256: string;
  file_size_bytes: number;
  architecture: string | null;
  format: string | null;
  status: ScanStatus;
  created_at: string;
  completed_at: string | null;
  verdict: VerdictLevel | null;
  overall_score: number | null;  // matches backend ScanListItem.overall_score
}

// ─── API Response wrappers ───────────────────────────────────────────────────
export interface ScanStatusResponse {
  scan_id: string;
  status: ScanStatus;
  progress_pct: number;          // 0-100, from backend ScanStatusResponse
  progress_message?: string;     // optional alias for UI display
  message: string;               // matches backend ScanStatusResponse.message
  created_at?: string;
  completed_at?: string | null;
}

export interface CreateScanResponse {
  scan_id: string;
  status: ScanStatus;
  message: string;
}

export interface PaginatedScans {
  items: ScanRecord[];
  total: number;
  skip: number;
  limit: number;
}
