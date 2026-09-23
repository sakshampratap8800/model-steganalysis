"""
File-system storage for uploaded model files and scan artifacts.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional


class LocalStorage:
    """
    Manages the directory layout for scan uploads and artifacts.

    Layout:
      <base_dir>/
        <scan_id>/
          model.<ext>       <- uploaded model file
          meta.json         <- written by extract_tensors.py
          weights.bin       <- written by extract_tensors.py
          report.json       <- written by scanner_cli (optional)
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Directory access
    # ------------------------------------------------------------------

    def get_scan_dir(self, scan_id: str) -> Path:
        """Return (and create) the per-scan working directory."""
        d = self.base_dir / scan_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def save_upload(self, scan_id: str, filename: str, data: bytes) -> Path:
        """
        Persist raw bytes of the uploaded model file.

        The file is stored with its original extension so downstream tools
        can identify the format.  Returns the absolute path of the saved file.
        """
        scan_dir = self.get_scan_dir(scan_id)
        suffix = Path(filename).suffix or ".bin"
        dest = scan_dir / f"model{suffix}"
        dest.write_bytes(data)
        return dest

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_model_path(self, scan_id: str) -> Optional[Path]:
        """
        Return the path to the uploaded model file, or None if not found.

        Searches for any file named 'model.*' in the scan directory.
        """
        scan_dir = self.base_dir / scan_id
        if not scan_dir.exists():
            return None
        candidates = list(scan_dir.glob("model.*"))
        return candidates[0] if candidates else None

    def get_meta_path(self, scan_id: str) -> Path:
        return self.get_scan_dir(scan_id) / "meta.json"

    def get_weights_path(self, scan_id: str) -> Path:
        return self.get_scan_dir(scan_id) / "weights.bin"

    def get_report_path(self, scan_id: str) -> Path:
        return self.get_scan_dir(scan_id) / "report.json"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_scan(self, scan_id: str) -> None:
        """
        Delete the uploaded model file to free space, but keep meta.json,
        weights.bin, and report.json for audit / re-analysis purposes.
        """
        model_path = self.get_model_path(scan_id)
        if model_path and model_path.exists():
            model_path.unlink()

    def delete_scan(self, scan_id: str) -> None:
        """Delete the entire scan directory (use with caution)."""
        scan_dir = self.base_dir / scan_id
        if scan_dir.exists():
            shutil.rmtree(scan_dir)
