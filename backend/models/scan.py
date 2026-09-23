"""
Pydantic models for the Scan API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class ScanStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Sub-models used in reports
# ---------------------------------------------------------------------------

class TensorProfileResponse(BaseModel):
    """Statistics for a single weight tensor."""
    name: str
    dims: List[int]
    n_params: int
    original_dtype: str
    # Basic stats
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    # LSB analysis
    lsb_entropy: Optional[float] = None          # Shannon entropy of LSB distribution
    lsb_chi_square_p: Optional[float] = None     # chi-square p-value for uniformity
    lsb_uniformity_score: Optional[float] = None # 0-1, 1 = perfectly uniform
    # Bit-plane metrics
    bit_plane_entropies: Optional[List[float]] = None  # entropy per bit plane [0..31]


class ModelProfileResponse(BaseModel):
    """Aggregate statistics across all tensors in the model."""
    architecture: str
    format: str
    total_tensors: int
    total_params: int
    sha256: str
    file_size_bytes: int
    opset_version: Optional[int] = None


class EvidenceFamilyScore(BaseModel):
    """Score for one evidence family (e.g., LSB, entropy, chi-square)."""
    family: str          # e.g. "lsb_statistics", "chi_square", "trojan_signature"
    score: float         # 0.0 - 1.0
    confidence: float    # 0.0 - 1.0
    description: str
    details: Optional[Dict[str, Any]] = None


class FlaggedLayer(BaseModel):
    """A layer flagged as suspicious with reason and score."""
    layer_name: str
    reason: str
    score: float         # 0.0 - 1.0
    dims: List[int]
    n_params: int
    details: Optional[Dict[str, Any]] = None


class ScanReport(BaseModel):
    """Full analysis report returned after a completed scan."""
    scan_id: str
    model_profile: ModelProfileResponse
    tensor_profiles: List[TensorProfileResponse]
    evidence_scores: List[EvidenceFamilyScore]
    flagged_layers: List[FlaggedLayer]
    overall_score: float            # 0.0 - 1.0 stego likelihood
    verdict: str                    # "clean" | "suspicious" | "malicious"
    verdict_confidence: float       # 0.0 - 1.0
    summary: str                    # human-readable one-paragraph summary
    generated_at: datetime
    scanner_version: Optional[str] = None
    raw_scanner_output: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Scan record (DB-backed)
# ---------------------------------------------------------------------------

class ScanRecord(BaseModel):
    """Full scan record including status and optional report."""
    id: str
    filename: str
    sha256: str
    file_size_bytes: int
    architecture: Optional[str] = None
    format: Optional[str] = None
    status: ScanStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    report: Optional[ScanReport] = None
    error_message: Optional[str] = None


class ScanListItem(BaseModel):
    """Lightweight scan listing item."""
    id: str
    filename: str
    sha256: str
    file_size_bytes: int
    architecture: Optional[str] = None
    status: ScanStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    overall_score: Optional[float] = None
    verdict: Optional[str] = None


# ---------------------------------------------------------------------------
# API request / response wrappers
# ---------------------------------------------------------------------------

class CreateScanResponse(BaseModel):
    """Response from POST /api/scans."""
    scan_id: str
    status: ScanStatus
    filename: str
    sha256: str
    file_size_bytes: int
    message: str = "Scan queued successfully"


class ScanStatusResponse(BaseModel):
    """Response from GET /api/scans/{id}/status."""
    scan_id: str
    status: ScanStatus
    progress_pct: int   # 0-100
    message: str


class PaginatedScans(BaseModel):
    """Response from GET /api/scans."""
    items: List[ScanListItem]
    total: int
    skip: int
    limit: int
