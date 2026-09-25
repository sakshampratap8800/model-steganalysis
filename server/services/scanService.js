const { v4: uuidv4 } = require('uuid');
const path = require('path');
const config = require('../config');
const db = require('../db');
const LocalStorage = require('../storage/localStorage');
const { runScan } = require('./scannerBridge');

const storage = new LocalStorage();

// ─── Helpers ──────────────────────────────────────────────────────────────────

function statusToProgress(status) {
  switch (status) {
    case 'queued':   return { progress_pct: 0,   message: 'Scan is queued and waiting to start.' };
    case 'running':  return { progress_pct: 50,  message: 'Scan is in progress.' };
    case 'complete': return { progress_pct: 100, message: 'Scan completed successfully.' };
    case 'failed':   return { progress_pct: 0,   message: 'Scan failed.' };
    default:         return { progress_pct: 0,   message: 'Unknown status.' };
  }
}

/**
 * Map raw C++ report JSON to the ScanReport shape the React frontend expects.
 * This is the translation layer between C++ output and TypeScript types.
 */
function buildScanReport(scanId, filename, reportJson) {
  const report = typeof reportJson === 'string' ? JSON.parse(reportJson) : reportJson;
  const profile = report.profile || {};
  const globalRisk = report.global_risk || {};
  const tensors = report.tensor_stats || [];

  // Map verdict: C++ uses CLEAN/SUSPICIOUS/MALICIOUS, frontend expects BENIGN/SUSPICIOUS/HIGH_RISK
  const verdictMap = { CLEAN: 'BENIGN', SUSPICIOUS: 'SUSPICIOUS', MALICIOUS: 'HIGH_RISK' };
  const verdict = verdictMap[globalRisk.verdict] || 'UNKNOWN';

  // Build evidence families from the normalized evidence vector
  const ev = globalRisk.normalized_evidence_vector || {};
  const evidenceFamilies = [
    {
      family: 'bit_representation',
      score: Math.round((ev.bit_entropy_deviation || 0) * 100),
      level: (ev.bit_entropy_deviation || 0) > 0.7 ? 'High' : (ev.bit_entropy_deviation || 0) > 0.3 ? 'Medium' : 'Low',
      description: 'IEEE-754 mantissa bit-level entropy analysis. Detects LSB steganography attacks (FMLA, HMLA, HBLA).',
      available: true,
    },
    {
      family: 'structural',
      score: Math.round((ev.structural_deviation || 0) * 100),
      level: (ev.structural_deviation || 0) > 0.7 ? 'High' : (ev.structural_deviation || 0) > 0.3 ? 'Medium' : 'Low',
      description: 'Project-proposed structural evidence using Gini coefficients and Spearman rank correlation across weight channels.',
      available: true,
    },
    {
      family: 'trojan_signature',
      score: Math.round((ev.signature_deviation || 0) * 100),
      level: (ev.signature_deviation || 0) > 0.7 ? 'High' : (ev.signature_deviation || 0) > 0.3 ? 'Medium' : 'Low',
      description: 'Dixon Q-test on final classification layer row means to detect Trojan backdoors.',
      available: !!report.trojan_signature,
    },
    {
      family: 'reference_deviation',
      score: Math.round((ev.statistical_deviation || 0) * 100),
      level: (ev.statistical_deviation || 0) > 0.7 ? 'High' : (ev.statistical_deviation || 0) > 0.3 ? 'Medium' : 'Low',
      description: 'Statistical deviation from a known-clean baseline model of the same architecture.',
      available: true,
    },
    {
      family: 'model_xray',
      score: 0,
      level: 'Unavailable',
      description: 'Grayscale-Fourpart (GF) Few-Shot detection. Requires pre-trained GF detector model.',
      available: false,
    },
    {
      family: 'behavioral',
      score: 0,
      level: 'Unavailable',
      description: 'Implicit feature analysis via softmax probe set. Requires classification model + probe images.',
      available: false,
    },
  ];

  // Build flagged layers from tensor_stats with high risk scores
  const flaggedLayers = tensors
    .filter((t) => t.risk_score > 0.35)
    .sort((a, b) => b.risk_score - a.risk_score)
    .slice(0, 10)
    .map((t) => ({
      name: t.name,
      anomaly_score: t.risk_score,
      primary_evidence: t.triggers || [],
      shape: profile.tensors?.find((p) => p.name === t.name)?.shape || [],
      parameter_count: profile.tensors?.find((p) => p.name === t.name)?.parameter_count || 0,
    }));

  // Build model profile
  const modelProfile = {
    scan_id: scanId,
    filename,
    sha256: profile.sha256 || '',
    format: profile.format || 'onnx',
    architecture: profile.architecture || 'unknown',
    architecture_hint: profile.architecture_hint || profile.architecture || 'unknown',
    dtype: profile.dtype || 'float32',
    tensor_count: profile.tensor_count || tensors.length,
    total_parameter_count: profile.total_parameter_count || 0,
    tensors: (profile.tensors || []).map((t) => ({
      name: t.name,
      shape: t.shape || [],
      dtype: t.dtype || 'float32',
      parameter_count: t.parameter_count || 0,
      min_val: t.stats?.min_val ?? 0,
      max_val: t.stats?.max_val ?? 0,
      mean: t.stats?.mean ?? 0,
      std_dev: t.stats?.std_dev ?? 0,
      variance: t.stats?.variance ?? 0,
      median: t.stats?.median ?? 0,
      sparsity: t.stats?.sparsity ?? 0,
    })),
  };

  const riskScore = Math.round((globalRisk.risk_score || 0) * 100);
  const triggers = globalRisk.triggers || [];

  return {
    scan_id: scanId,
    model_hash: profile.sha256 || '',
    filename,
    architecture: profile.architecture || 'unknown',
    format: profile.format || 'onnx',
    dtype: profile.dtype || 'float32',
    risk_score: riskScore,
    verdict,
    status_message: `Analysis complete. Risk score: ${riskScore}/100`,
    evidence_families: evidenceFamilies,
    flagged_layers: flaggedLayers,
    explanation: triggers.length > 0
      ? triggers.slice(0, 3).join(' ')
      : 'No significant anomalies detected in this model.',
    behavioral_analysis_available: false,
    scan_duration_ms: report.scan_duration_ms || 0,
    scanner_version: '1.0.0',
    model_profile: modelProfile,
  };
}

// ─── Service functions ────────────────────────────────────────────────────────

/**
 * Create a new scan: save file, write DB record, kick off background job.
 */
async function createScan(filename, fileBuffer) {
  const ext = path.extname(filename).toLowerCase();
  if (!config.ALLOWED_EXTENSIONS.has(ext)) {
    const err = new Error(`Unsupported file type '${ext}'. Allowed: ${[...config.ALLOWED_EXTENSIONS].join(', ')}`);
    err.statusCode = 400;
    throw err;
  }

  const scanId = uuidv4();
  const sha256 = LocalStorage.sha256(fileBuffer);
  const modelPath = storage.saveUpload(scanId, filename, fileBuffer);

  db.createScan({
    id: scanId,
    filename: path.basename(filename),
    sha256,
    file_size_bytes: fileBuffer.length,
    format: ext.replace('.', ''),
  });

  // Fire-and-forget background scan
  runScanBackground(scanId, modelPath).catch(() => {});

  return {
    scan_id: scanId,
    status: 'queued',
    message: 'Scan queued successfully.',
  };
}

async function runScanBackground(scanId, modelPath) {
  const scanDir = storage.getScanDir(scanId);
  db.updateStatus(scanId, 'running');

  try {
    const report = await runScan(scanId, modelPath, scanDir);
    db.saveReport(scanId, report);
    storage.cleanupModel(scanId, path.basename(modelPath));
  } catch (err) {
    db.updateStatus(scanId, 'failed', err.message);
  }
}

function getScan(scanId) {
  const row = db.getScan(scanId);
  if (!row) return null;

  return {
    id: row.id,
    filename: row.filename,
    sha256: row.sha256,
    file_size_bytes: row.file_size_bytes,
    architecture: row.architecture || null,
    format: row.format || null,
    status: row.status,
    created_at: row.created_at,
    completed_at: row.completed_at || null,
    verdict: row.verdict || null,
    overall_score: row.overall_score ?? null,
  };
}

function getScanStatus(scanId) {
  const row = db.getScan(scanId);
  if (!row) return null;

  const { progress_pct, message } = statusToProgress(row.status);
  return {
    scan_id: scanId,
    status: row.status,
    progress_pct,
    message: row.status === 'failed' ? `Scan failed: ${row.error_message || ''}` : message,
    created_at: row.created_at,
    completed_at: row.completed_at || null,
  };
}

function getScanReport(scanId) {
  const row = db.getScan(scanId);
  if (!row) return null;
  if (row.status !== 'complete' || !row.report_json) return null;
  return buildScanReport(scanId, row.filename, row.report_json);
}

function listScans(limit = 50, skip = 0) {
  const { items, total } = db.listScans(limit, skip);
  return {
    items: items.map((row) => ({
      id: row.id,
      filename: row.filename,
      sha256: row.sha256,
      file_size_bytes: row.file_size_bytes,
      architecture: row.architecture || null,
      format: row.format || null,
      status: row.status,
      created_at: row.created_at,
      completed_at: row.completed_at || null,
      verdict: row.verdict || null,
      overall_score: row.overall_score ?? null,
    })),
    total,
    skip,
    limit,
  };
}

module.exports = { createScan, getScan, getScanStatus, getScanReport, listScans };
