"""
FastAPI router for scan endpoints.

Endpoints:
  POST   /api/scans               - Upload model, create scan
  GET    /api/scans               - List scans (paginated)
  GET    /api/scans/{scan_id}     - Get scan record
  GET    /api/scans/{scan_id}/status  - Polling status
  GET    /api/scans/{scan_id}/report  - Full report (404 if not complete)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from config import config
from models.scan import (
    CreateScanResponse,
    PaginatedScans,
    ScanRecord,
    ScanReport,
    ScanStatusResponse,
)
from services.scan_service import ScanService

router = APIRouter(tags=["scans"])

# ---------------------------------------------------------------------------
# Dependency: shared ScanService instance
# ---------------------------------------------------------------------------

_scan_service: ScanService | None = None


def get_scan_service() -> ScanService:
    global _scan_service
    if _scan_service is None:
        _scan_service = ScanService()
    return _scan_service


ServiceDep = Annotated[ScanService, Depends(get_scan_service)]


# ---------------------------------------------------------------------------
# POST /scans  — upload model & queue scan
# ---------------------------------------------------------------------------

@router.post(
    "/scans",
    response_model=CreateScanResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a model file and start a scan",
)
async def create_scan(
    service: ServiceDep,
    file: UploadFile = File(..., description="Model file (.onnx, .pt, .pth, .bin, .pb)"),
) -> CreateScanResponse:
    """
    Accept a multipart-encoded model upload, validate it, and queue a scan.
    Returns immediately with the scan_id; polling /status for progress.
    """
    if file.size is not None and file.size > config.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of "
                   f"{config.MAX_UPLOAD_SIZE_BYTES // (1024 ** 2)} MB.",
        )

    filename = file.filename or "upload.bin"

    try:
        result = await service.create_scan(filename=filename, file=file)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return result


# ---------------------------------------------------------------------------
# GET /scans  — paginated list
# ---------------------------------------------------------------------------

@router.get(
    "/scans",
    response_model=PaginatedScans,
    summary="List all scans",
)
async def list_scans(
    service: ServiceDep,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max records to return"),
) -> PaginatedScans:
    items, total = await service.list_scans(skip=skip, limit=limit)
    return PaginatedScans(items=items, total=total, skip=skip, limit=limit)


# ---------------------------------------------------------------------------
# GET /scans/{scan_id}  — full record
# ---------------------------------------------------------------------------

@router.get(
    "/scans/{scan_id}",
    response_model=ScanRecord,
    summary="Get a scan record",
)
async def get_scan(scan_id: str, service: ServiceDep) -> ScanRecord:
    try:
        return await service.get_scan(scan_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


# ---------------------------------------------------------------------------
# GET /scans/{scan_id}/status  — lightweight polling
# ---------------------------------------------------------------------------

@router.get(
    "/scans/{scan_id}/status",
    response_model=ScanStatusResponse,
    summary="Poll scan status",
)
async def get_scan_status(scan_id: str, service: ServiceDep) -> ScanStatusResponse:
    try:
        return await service.get_scan_status(scan_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


# ---------------------------------------------------------------------------
# GET /scans/{scan_id}/report  — full analysis report
# ---------------------------------------------------------------------------

@router.get(
    "/scans/{scan_id}/report",
    response_model=ScanReport,
    summary="Get the full analysis report (only available when status=complete)",
)
async def get_scan_report(scan_id: str, service: ServiceDep) -> ScanReport:
    try:
        return await service.get_scan_report(scan_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
