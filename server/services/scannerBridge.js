const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const config = require('../config');

/**
 * Run a child process and capture its stdout/stderr.
 * Rejects on non-zero exit code or timeout.
 */
function runProcess(binary, args, timeoutMs = config.SCAN_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    const proc = spawn(binary, args);
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (d) => { stdout += d.toString(); });
    proc.stderr.on('data', (d) => { stderr += d.toString(); });

    const timer = setTimeout(() => {
      proc.kill();
      reject(new Error(`Process timed out after ${timeoutMs / 1000}s: ${binary}`));
    }, timeoutMs);

    proc.on('close', (code) => {
      clearTimeout(timer);
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(`[${path.basename(binary)}] exited ${code}\nstderr: ${stderr}`));
      }
    });

    proc.on('error', (err) => {
      clearTimeout(timer);
      reject(new Error(`Failed to start ${binary}: ${err.message}`));
    });
  });
}

/**
 * Full scan pipeline:
 *   Step 1 — onnx_extract.exe : ONNX file → meta.json + weights.bin
 *   Step 2 — scanner_cli.exe  : meta.json + weights.bin → report.json
 *
 * Returns the parsed report object.
 */
async function runScan(scanId, modelPath, scanDir) {
  const metaPath   = path.join(scanDir, 'meta.json');
  const reportPath = path.join(scanDir, 'report.json');

  // ── Step 1: Extract tensors ──────────────────────────────────────────────
  if (!fs.existsSync(config.ONNX_EXTRACTOR)) {
    throw new Error(
      `ONNX extractor not found at ${config.ONNX_EXTRACTOR}.\n` +
      `Run: cmake --build build --config Debug`
    );
  }

  await runProcess(config.ONNX_EXTRACTOR, [modelPath, '--output-dir', scanDir, '--scan-id', scanId]);

  if (!fs.existsSync(metaPath)) {
    throw new Error('ONNX extractor ran but meta.json was not produced.');
  }

  // ── Step 2: Run C++ scanner ──────────────────────────────────────────────
  if (!fs.existsSync(config.SCANNER_CLI)) {
    throw new Error(
      `Scanner CLI not found at ${config.SCANNER_CLI}.\n` +
      `Run: cmake --build build --config Debug`
    );
  }

  const scannerArgs = [metaPath, '--output', reportPath];
  if (fs.existsSync(config.BASELINES_PATH)) {
    scannerArgs.push('--baseline', config.BASELINES_PATH);
  }

  await runProcess(config.SCANNER_CLI, scannerArgs);

  if (!fs.existsSync(reportPath)) {
    throw new Error('scanner_cli ran but report.json was not produced.');
  }

  return JSON.parse(fs.readFileSync(reportPath, 'utf-8'));
}

module.exports = { runScan };
