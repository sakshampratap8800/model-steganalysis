const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const config = require('../config');

class LocalStorage {
  constructor(baseDir = config.DATA_DIR) {
    this.baseDir = baseDir;
    fs.mkdirSync(baseDir, { recursive: true });
  }

  getScanDir(scanId) {
    return path.join(this.baseDir, scanId);
  }

  /**
   * Save an uploaded file buffer to disk inside the scan's directory.
   * Returns the full path where the file was written.
   */
  saveUpload(scanId, filename, buffer) {
    const scanDir = this.getScanDir(scanId);
    fs.mkdirSync(scanDir, { recursive: true });
    const dest = path.join(scanDir, path.basename(filename));
    fs.writeFileSync(dest, buffer);
    return dest;
  }

  /**
   * Compute SHA-256 of a buffer.
   */
  static sha256(buffer) {
    return crypto.createHash('sha256').update(buffer).digest('hex');
  }

  /**
   * Delete the uploaded model file after scanning (save disk space).
   * Keeps meta.json, weights.bin, and report.json.
   */
  cleanupModel(scanId, filename) {
    const modelPath = path.join(this.getScanDir(scanId), path.basename(filename));
    if (fs.existsSync(modelPath)) {
      fs.unlinkSync(modelPath);
    }
  }
}

module.exports = LocalStorage;
