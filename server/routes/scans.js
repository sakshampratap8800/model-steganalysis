const express = require('express');
const multer = require('multer');
const config = require('../config');
const {
  createScan,
  getScan,
  getScanStatus,
  getScanReport,
  listScans,
} = require('../services/scanService');

const router = express.Router();

// multer: store upload in memory (max 2 GB), we write to disk ourselves
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: config.MAX_FILE_SIZE_MB * 1024 * 1024 },
});

// ─── POST /api/scans ─────────────────────────────────────────────────────────
// Upload a model file and start a scan.
router.post('/', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ detail: 'No file uploaded. Send a multipart/form-data request with field name "file".' });
  }

  try {
    const result = await createScan(req.file.originalname, req.file.buffer);
    return res.status(202).json(result);
  } catch (err) {
    return res.status(err.statusCode || 500).json({ detail: err.message });
  }
});

// ─── GET /api/scans ──────────────────────────────────────────────────────────
// Paginated list of all scans.
router.get('/', (req, res) => {
  const limit = Math.min(parseInt(req.query.limit) || 50, 100);
  const skip  = parseInt(req.query.skip) || 0;

  try {
    const result = listScans(limit, skip);
    return res.json(result);
  } catch (err) {
    return res.status(500).json({ detail: err.message });
  }
});

// ─── GET /api/scans/:id ──────────────────────────────────────────────────────
// Get a single scan record (lightweight, used for list rows).
router.get('/:id', (req, res) => {
  const scan = getScan(req.params.id);
  if (!scan) return res.status(404).json({ detail: `Scan '${req.params.id}' not found.` });
  return res.json(scan);
});

// ─── GET /api/scans/:id/status ───────────────────────────────────────────────
// Lightweight polling endpoint — React frontend polls this every 2s while running.
router.get('/:id/status', (req, res) => {
  const status = getScanStatus(req.params.id);
  if (!status) return res.status(404).json({ detail: `Scan '${req.params.id}' not found.` });
  return res.json(status);
});

// ─── GET /api/scans/:id/report ───────────────────────────────────────────────
// Full scan report — only available when status === 'complete'.
router.get('/:id/report', (req, res) => {
  const scan = getScan(req.params.id);
  if (!scan) return res.status(404).json({ detail: `Scan '${req.params.id}' not found.` });

  if (scan.status !== 'complete') {
    return res.status(409).json({
      detail: `Scan is not complete yet (status: ${scan.status}). Poll /status first.`,
    });
  }

  const report = getScanReport(req.params.id);
  if (!report) {
    return res.status(500).json({ detail: 'Scan is complete but report data is missing.' });
  }

  return res.json(report);
});

module.exports = router;
