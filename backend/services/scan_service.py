"""
ScanService: high-level scan lifecycle management.

Handles upload validation, DB record creation, background task dispatch,
and report retrieval.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config import config
from models.scan import (
    CreateScanResponse,
    ScanListItem,
    ScanRecord,
    ScanReport,
    ScanStatus,
    ScanStatusResponse,
    ModelProfileResponse,
    TensorProfileResponse,
    EvidenceFamilyScore,
    FlaggedLayer,
)
from storage.database import (
    create_scan as db_create_scan,
    get_scan as db_get_scan,
    list_scans as db_list_scans,
    update_scan_report as db_update_scan_report,
    update_scan_status as db_update_scan_status,
)
from storage.local_storage import LocalStorage
from services.scanner_bridge import ScannerBridge, ScannerError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _row_to_record(row) -> ScanRecord:
    """Convert a SQLAlchemy ScanRow to a ScanRecord pydantic model."""
    report_obj: Optional[ScanReport] = None
    if row.report_json:
        try:
            report_data = json.loads(row.report_json)
            report_obj = _dict_to_report(report_data)
        except Exception:
            pass  # malformed report — return without it

    return ScanRecord(
        id=row.id,
        filename=row.filename,
        sha256=row.sha256,
        file_size_bytes=row.file_size_bytes,
        architecture=row.architecture,
        format=row.format,
        status=ScanStatus(row.status),
        created_at=_ensure_tz(row.created_at),
        completed_at=_ensure_tz(row.completed_at) if row.completed_at else None,
        report=report_obj,
        error_message=row.error_message,
    )


def _row_to_list_item(row) -> ScanListItem:
    overall_score: Optional[float] = None
    verdict: Optional[str] = None
    if row.report_json:
        try:
            d = json.loads(row.report_json)
            overall_score = d.get("overall_score")
            verdict = d.get("verdict")
        except Exception:
            pass
    return ScanListItem(
        id=row.id,
        filename=row.filename,
        sha256=row.sha256,
        file_size_bytes=row.file_size_bytes,
        architecture=row.architecture,
        status=ScanStatus(row.status),
        created_at=_ensure_tz(row.created_at),
        completed_at=_ensure_tz(row.completed_at) if row.completed_at else None,
        overall_score=overall_score,
        verdict=verdict,
    )


def _ensure_tz(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _dict_to_report(d: dict) -> ScanReport:
    """Deserialize a report dict (from JSON) into a ScanReport pydantic model."""
    mp = d.get("model_profile", {})
    model_profile = ModelProfileResponse(
        architecture=mp.get("architecture", "unknown"),
        format=mp.get("format", "onnx"),
        total_tensors=mp.get("total_tensors", 0),
        total_params=mp.get("total_params", 0),
        sha256=mp.get("sha256", ""),
        file_size_bytes=mp.get("file_size_bytes", 0),
        opset_version=mp.get("opset_version"),
    )
    tensor_profiles = [
        TensorProfileResponse(**tp) for tp in d.get("tensor_profiles", [])
    ]
    evidence_scores = [
        EvidenceFamilyScore(**ev) for ev in d.get("evidence_scores", [])
    ]
    flagged_layers = [
        FlaggedLayer(**fl) for fl in d.get("flagged_layers", [])
    ]
    generated_at = d.get("generated_at", datetime.now(timezone.utc).isoformat())
    if isinstance(generated_at, str):
        generated_at = datetime.fromisoformat(generated_at)

    return ScanReport(
        scan_id=d.get("scan_id", ""),
        model_profile=model_profile,
        tensor_profiles=tensor_profiles,
        evidence_scores=evidence_scores,
        flagged_layers=flagged_layers,
        overall_score=d.get("overall_score", 0.0),
        verdict=d.get("verdict", "clean"),
        verdict_confidence=d.get("verdict_confidence", 0.0),
        summary=d.get("summary", ""),
        generated_at=generated_at,
        scanner_version=d.get("scanner_version"),
        raw_scanner_output=d.get("raw_scanner_output"),
    )


# ---------------------------------------------------------------------------
# Allowed file extensions
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS = {".onnx", ".pt", ".pth", ".bin", ".pb"}


def _validate_upload(filename: str, data: bytes) -> None:
    """Raise ValueError if the upload is invalid."""
    if len(data) == 0:
        raise ValueError("Uploaded file is empty.")

    if len(data) > config.MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"File exceeds maximum upload size of "
            f"{config.MAX_UPLOAD_SIZE_BYTES // (1024**2)} MB."
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
        )


# ---------------------------------------------------------------------------
# ScanService
# ---------------------------------------------------------------------------

class ScanService:
    """High-level service for the scan lifecycle."""

    def __init__(
        self,
        storage: Optional[LocalStorage] = None,
        bridge: Optional[ScannerBridge] = None,
    ) -> None:
        self.storage = storage or LocalStorage(config.DATA_DIR)
        self.bridge = bridge or ScannerBridge()

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_scan(self, filename: str, file) -> CreateScanResponse:
        """
        Validate, persist, and queue a new scan.

        Returns CreateScanResponse immediately; scanning runs in background.
        """
        safe_filename = Path(filename).name
        suffix = Path(safe_filename).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
            )

        scan_id = str(uuid.uuid4())
        scan_dir = self.storage.get_scan_dir(scan_id)
        scan_dir.mkdir(parents=True, exist_ok=True)
        model_path = scan_dir / safe_filename

        # Stream directly to disk to prevent OOM
        hasher = hashlib.sha256()
        file_size = 0
        
        try:
            with open(model_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    hasher.update(chunk)
                    f.write(chunk)
                    file_size += len(chunk)
                    if file_size > config.MAX_UPLOAD_SIZE_BYTES:
                        raise ValueError(f"File exceeds maximum upload size")
        except Exception:
            import shutil
            if model_path.exists():
                model_path.unlink()
            if scan_dir.exists():
                shutil.rmtree(scan_dir, ignore_errors=True)
            raise
                    
        sha = hasher.hexdigest()

        # Create DB record
        db_create_scan(
            scan_id=scan_id,
            filename=safe_filename,
            sha256=sha,
            file_size_bytes=file_size,
        )

        # Launch background scan (fire-and-forget)
        asyncio.create_task(self._run_scan_background(scan_id, model_path))

        return CreateScanResponse(
            scan_id=scan_id,
            status=ScanStatus.QUEUED,
            filename=safe_filename,
            sha256=sha,
            file_size_bytes=file_size,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_scan(self, scan_id: str) -> ScanRecord:
        row = db_get_scan(scan_id)
        if row is None:
            raise KeyError(f"Scan '{scan_id}' not found.")
        return _row_to_record(row)

    async def get_scan_status(self, scan_id: str) -> ScanStatusResponse:
        row = db_get_scan(scan_id)
        if row is None:
            raise KeyError(f"Scan '{scan_id}' not found.")

        status = ScanStatus(row.status)
        progress_pct, message = _status_progress(status, row.error_message)

        return ScanStatusResponse(
            scan_id=scan_id,
            status=status,
            progress_pct=progress_pct,
            message=message,
        )

    async def get_scan_report(self, scan_id: str) -> ScanReport:
        row = db_get_scan(scan_id)
        if row is None:
            raise KeyError(f"Scan '{scan_id}' not found.")
        if row.status != "complete":
            raise ValueError(f"Scan '{scan_id}' is not complete (status={row.status}).")
        if not row.report_json:
            raise ValueError(f"Scan '{scan_id}' has no report data.")
        return _dict_to_report(json.loads(row.report_json))

    async def list_scans(self, skip: int = 0, limit: int = 20) -> tuple:
        rows, total = db_list_scans(skip=skip, limit=limit)
        items = [_row_to_list_item(r) for r in rows]
        return items, total

    # ------------------------------------------------------------------
    # Background scan task
    # ------------------------------------------------------------------

    async def _run_scan_background(self, scan_id: str, model_path: Path) -> None:
        """
        Background coroutine that drives the scan pipeline.

        Updates the DB record at each stage transition.
        """
        scan_dir = self.storage.get_scan_dir(scan_id)

        # Transition -> RUNNING
        db_update_scan_status(scan_id, "running")

        try:
            report = await self.bridge.run_scan(
                scan_id=scan_id,
                model_path=model_path,
                scan_dir=scan_dir,
            )

            # Persist architecture and format alongside the report
            arch = report.get("model_profile", {}).get("architecture")
            fmt = report.get("model_profile", {}).get("format", "onnx")
            # Update arch/format before saving report
            if arch or fmt:
                db_update_scan_status(scan_id, "running", architecture=arch, format_str=fmt)

            db_update_scan_report(scan_id, report)

        except asyncio.TimeoutError as exc:
            db_update_scan_status(
                scan_id, "failed", error_message=f"Scan timed out: {exc}"
            )
        except ScannerError as exc:
            db_update_scan_status(
                scan_id, "failed", error_message=str(exc)
            )
        except Exception as exc:
            db_update_scan_status(
                scan_id, "failed", error_message=f"Unexpected error: {exc}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_progress(status: ScanStatus, error_message: Optional[str]) -> tuple[int, str]:
    if status == ScanStatus.QUEUED:
        return 0, "Scan is queued and waiting to start."
    if status == ScanStatus.RUNNING:
        return 50, "Scan is in progress."
    if status == ScanStatus.COMPLETE:
        return 100, "Scan completed successfully."
    if status == ScanStatus.FAILED:
        msg = f"Scan failed. {error_message or ''}".strip()
        return 0, msg
    return 0, "Unknown status."
