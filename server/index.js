const express = require('express');
const cors = require('cors');
const path = require('path');
const config = require('./config');
const scansRouter = require('./routes/scans');

const app = express();

// ─── Middleware ───────────────────────────────────────────────────────────────

// Allow React dev server (port 3000 / 5173) to call this API
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:5173'],
  methods: ['GET', 'POST', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ─── Routes ───────────────────────────────────────────────────────────────────

// Health check — useful during demo to confirm server is alive
app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '1.0.0', timestamp: new Date().toISOString() });
});

// All scan endpoints under /api/scans
app.use('/api/scans', scansRouter);

// ─── 404 handler ─────────────────────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({ detail: `Route ${req.method} ${req.path} not found.` });
});

// ─── Global error handler ─────────────────────────────────────────────────────
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  console.error('[Server Error]', err);
  res.status(500).json({ detail: err.message || 'Internal server error.' });
});

// ─── Start ────────────────────────────────────────────────────────────────────
app.listen(config.PORT, () => {
  console.log(`\n🛡  Model Steganalysis API`);
  console.log(`   Running on http://localhost:${config.PORT}`);
  console.log(`   Health: http://localhost:${config.PORT}/health`);
  console.log(`   Scans:  http://localhost:${config.PORT}/api/scans\n`);
});

module.exports = app;
