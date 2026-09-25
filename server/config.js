const path = require('path');

const config = {
  // Server
  PORT: process.env.PORT || 8000,

  // Paths
  DATA_DIR: path.resolve(__dirname, '../data/scans'),
  DB_PATH:  path.resolve(__dirname, '../data/scans.db'),

  // C++ binaries (built by CMake)
  SCANNER_CLI: path.resolve(
    __dirname,
    process.platform === 'win32'
      ? '../build/core/Debug/scanner_cli.exe'
      : '../build/core/scanner_cli'
  ),
  ONNX_EXTRACTOR: path.resolve(
    __dirname,
    process.platform === 'win32'
      ? '../build/core/Debug/onnx_extract.exe'
      : '../build/core/onnx_extract'
  ),

  // Baselines (optional, used by the C++ fusion layer)
  BASELINES_PATH: path.resolve(__dirname, '../data/baselines.json'),

  // Upload limits
  MAX_FILE_SIZE_MB: 2048, // 2 GB

  // Scan timeout (ms) — kills the process if it hangs
  SCAN_TIMEOUT_MS: 300_000, // 5 minutes

  // Allowed file extensions
  ALLOWED_EXTENSIONS: new Set(['.onnx', '.pt', '.pth', '.bin', '.pb']),
};

module.exports = config;
