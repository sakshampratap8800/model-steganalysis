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

// multer: store upload in memory, we write to disk ourselves
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: config.MAX_FILE_SIZE_MB * 1024 * 1024 },
});

// ─── POST /api/scans ─────────────────────────────────────────────────────────
router.post('/', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ detail: 'No file uploaded. Use field name "file".' });
  }
  try {
    const result = await createScan(req.file.originalname, req.file.buffer);
    return res.status(202).json(result);
  } catch (err) {
    return res.status(err.statusCode || 500).json({ detail: err.message });
  }
});

// ─── GET /api/scans ──────────────────────────────────────────────────────────
router.get('/', async (req, res) => {
  const limit = Math.min(parseInt(req.query.limit) || 50, 100);
  const skip  = parseInt(req.query.skip) || 0;
  try {
    const result = await listScans(limit, skip);
    return res.json(result);
  } catch (err) {
    return res.status(500).json({ detail: err.message });
  }
});

// ─── GET /api/scans/:id ──────────────────────────────────────────────────────
router.get('/:id', async (req, res) => {
  try {
    const scan = await getScan(req.params.id);
    if (!scan) return res.status(404).json({ detail: `Scan '${req.params.id}' not found.` });
    return res.json(scan);
  } catch (err) {
    return res.status(500).json({ detail: err.message });
  }
});

// ─── GET /api/scans/:id/status ───────────────────────────────────────────────
router.get('/:id/status', async (req, res) => {
  try {
    const status = await getScanStatus(req.params.id);
    if (!status) return res.status(404).json({ detail: `Scan '${req.params.id}' not found.` });
    return res.json(status);
  } catch (err) {
    return res.status(500).json({ detail: err.message });
  }
});

// ─── GET /api/scans/:id/report ───────────────────────────────────────────────
router.get('/:id/report', async (req, res) => {
  try {
    const scan = await getScan(req.params.id);
    if (!scan) return res.status(404).json({ detail: `Scan '${req.params.id}' not found.` });

    if (scan.status !== 'complete') {
      return res.status(409).json({
        detail: `Scan is not complete yet (status: ${scan.status}).`,
      });
    }

    const report = await getScanReport(req.params.id);
    if (!report) {
      return res.status(500).json({ detail: 'Scan is complete but report data is missing.' });
    }
    return res.json(report);
  } catch (err) {
    return res.status(500).json({ detail: err.message });
  }
});

module.exports = router;
