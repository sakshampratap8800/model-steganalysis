const initSqlJs = require('sql.js');
const fs = require('fs');
const path = require('path');
const config = require('./config');

// sql.js runs SQLite compiled to WebAssembly — no native C++ build required.
// We persist the database to disk manually (read on startup, write on each change).

let db = null;
const DB_PATH = config.DB_PATH;

async function getDb() {
  if (db) return db;

  const SQL = await initSqlJs();

  // Load existing database from disk, or create a fresh one
  if (fs.existsSync(DB_PATH)) {
    const fileBuffer = fs.readFileSync(DB_PATH);
    db = new SQL.Database(fileBuffer);
  } else {
    fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
    db = new SQL.Database();
  }

  // Create table if it doesn't exist
  db.run(`
    CREATE TABLE IF NOT EXISTS scans (
      id            TEXT PRIMARY KEY,
      filename      TEXT NOT NULL,
      sha256        TEXT NOT NULL,
      file_size_bytes INTEGER NOT NULL,
      architecture  TEXT,
      format        TEXT,
      status        TEXT NOT NULL DEFAULT 'queued',
      created_at    TEXT NOT NULL,
      completed_at  TEXT,
      verdict       TEXT,
      overall_score REAL,
      report_json   TEXT,
      error_message TEXT
    )
  `);

  persist();
  return db;
}

// Write the in-memory SQLite database back to disk
function persist() {
  if (!db) return;
  const data = db.export();
  fs.writeFileSync(DB_PATH, Buffer.from(data));
}

// ─── Helper: run a query and return all rows as objects ───────────────────────
function queryAll(sql, params = []) {
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const rows = [];
  while (stmt.step()) {
    rows.push(stmt.getAsObject());
  }
  stmt.free();
  return rows;
}

function queryOne(sql, params = []) {
  const rows = queryAll(sql, params);
  return rows[0] || null;
}

// ─── Public API ───────────────────────────────────────────────────────────────

async function createScan({ id, filename, sha256, file_size_bytes, format }) {
  const d = await getDb();
  d.run(
    `INSERT INTO scans (id, filename, sha256, file_size_bytes, format, status, created_at)
     VALUES (?, ?, ?, ?, ?, 'queued', ?)`,
    [id, filename, sha256, file_size_bytes, format || 'onnx', new Date().toISOString()]
  );
  persist();
}

async function getScan(id) {
  await getDb();
  return queryOne('SELECT * FROM scans WHERE id = ?', [id]);
}

async function listScans(limit = 50, skip = 0) {
  await getDb();
  const items = queryAll('SELECT * FROM scans ORDER BY created_at DESC LIMIT ? OFFSET ?', [limit, skip]);
  const totalRow = queryOne('SELECT COUNT(*) as total FROM scans');
  return { items, total: totalRow ? totalRow.total : 0 };
}

async function updateStatus(id, status, errorMessage = null) {
  await getDb();
  db.run('UPDATE scans SET status = ?, error_message = ? WHERE id = ?', [status, errorMessage, id]);
  persist();
}

async function saveReport(id, report) {
  await getDb();
  const verdict       = report?.global_risk?.verdict ?? 'UNKNOWN';
  const overall_score = report?.global_risk?.risk_score ?? null;
  const architecture  = report?.profile?.architecture ?? null;

  db.run(
    `UPDATE scans
     SET status = 'complete', completed_at = ?, verdict = ?, overall_score = ?,
         architecture = ?, report_json = ?
     WHERE id = ?`,
    [new Date().toISOString(), verdict, overall_score, architecture, JSON.stringify(report), id]
  );
  persist();
}

module.exports = { createScan, getScan, listScans, updateStatus, saveReport };
