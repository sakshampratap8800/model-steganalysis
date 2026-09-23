"""
Pytest tests for the Model Steganalysis API.

Uses httpx.AsyncClient with the FastAPI app in-process (no real server needed).
"""

from __future__ import annotations

import io
import struct
from typing import Generator

import numpy as np
import onnx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from onnx import TensorProto, helper, numpy_helper


# ---------------------------------------------------------------------------
# Minimal ONNX model fixture
# ---------------------------------------------------------------------------

def _build_tiny_onnx() -> bytes:
    """
    Build a minimal valid ONNX model in memory and return its bytes.

    Model: single Conv node
      Input:  X  (1, 1, 8, 8) float32
      Weight: W  (1, 1, 3, 3) float32  <- initializer (weight tensor)
      Output: Y  (1, 1, 6, 6) float32
    """
    # Weight initializer
    W_data = np.random.randn(1, 1, 3, 3).astype(np.float32)
    W_init = numpy_helper.from_array(W_data, name="conv_weight")

    # Graph I/O
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 6, 6])

    # Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "conv_weight"],
        outputs=["Y"],
        kernel_shape=[3, 3],
    )

    graph = helper.make_graph(
        nodes=[conv_node],
        name="tiny_graph",
        inputs=[X],
        outputs=[Y],
        initializer=[W_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    return model.SerializeToString()


@pytest.fixture(scope="session")
def tiny_onnx_bytes() -> bytes:
    return _build_tiny_onnx()


# ---------------------------------------------------------------------------
# App fixture — import after env is set
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    import sys, os
    # Ensure backend package root is on path when running from repo root
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    os.environ.setdefault("DATABASE_URL", "sqlite:///./data/test_scans.db")
    os.environ.setdefault("DATA_DIR", "./data/test_scans")
    from main import app as _app
    return _app


# ---------------------------------------------------------------------------
# Sync client for simple tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client(app) -> Generator:
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# 2. POST /api/scans with valid ONNX file
# ---------------------------------------------------------------------------

def test_create_scan_valid_onnx(client, tiny_onnx_bytes):
    resp = client.post(
        "/api/scans",
        files={"file": ("model.onnx", io.BytesIO(tiny_onnx_bytes), "application/octet-stream")},
    )
    assert resp.status_code in (200, 202), resp.text
    body = resp.json()
    assert "scan_id" in body
    assert len(body["scan_id"]) == 36  # UUID format
    assert body["status"] in ("queued", "running")
    return body["scan_id"]


# ---------------------------------------------------------------------------
# 3. GET /api/scans/{id}/status
# ---------------------------------------------------------------------------

def test_scan_status(client, tiny_onnx_bytes):
    # Create a scan first
    create_resp = client.post(
        "/api/scans",
        files={"file": ("model.onnx", io.BytesIO(tiny_onnx_bytes), "application/octet-stream")},
    )
    assert create_resp.status_code in (200, 202)
    scan_id = create_resp.json()["scan_id"]

    status_resp = client.get(f"/api/scans/{scan_id}/status")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["scan_id"] == scan_id
    assert body["status"] in ("queued", "running", "complete", "failed")
    assert 0 <= body["progress_pct"] <= 100
    assert isinstance(body["message"], str)


# ---------------------------------------------------------------------------
# 4. GET /api/scans — list
# ---------------------------------------------------------------------------

def test_list_scans(client, tiny_onnx_bytes):
    # Ensure at least one scan exists
    client.post(
        "/api/scans",
        files={"file": ("model.onnx", io.BytesIO(tiny_onnx_bytes), "application/octet-stream")},
    )

    resp = client.get("/api/scans")
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert "total" in body
    assert isinstance(body["items"], list)
    assert body["total"] >= 1


# ---------------------------------------------------------------------------
# 5. POST /api/scans with invalid file type — should return 400
# ---------------------------------------------------------------------------

def test_create_scan_invalid_type(client):
    resp = client.post(
        "/api/scans",
        files={"file": ("script.exe", io.BytesIO(b"\x00\x01\x02\x03bad"), "application/octet-stream")},
    )
    assert resp.status_code == 400
    detail = resp.json().get("detail", "")
    assert "Unsupported" in detail or "invalid" in detail.lower() or "type" in detail.lower()


# ---------------------------------------------------------------------------
# 6. POST /api/scans with empty file — should return 400
# ---------------------------------------------------------------------------

def test_create_scan_empty_file(client):
    resp = client.post(
        "/api/scans",
        files={"file": ("empty.onnx", io.BytesIO(b""), "application/octet-stream")},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 7. GET /api/scans/{non-existent-id} — 404
# ---------------------------------------------------------------------------

def test_get_scan_not_found(client):
    resp = client.get("/api/scans/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 8. GET /api/scans/{id}/report for queued scan — 404
# ---------------------------------------------------------------------------

def test_get_report_not_complete(client, tiny_onnx_bytes):
    create_resp = client.post(
        "/api/scans",
        files={"file": ("model.onnx", io.BytesIO(tiny_onnx_bytes), "application/octet-stream")},
    )
    scan_id = create_resp.json()["scan_id"]

    # Report should not be available yet (scan is queued/running immediately after creation)
    # We check that when status is queued, report returns 404
    status_resp = client.get(f"/api/scans/{scan_id}/status")
    if status_resp.json()["status"] in ("queued", "running"):
        report_resp = client.get(f"/api/scans/{scan_id}/report")
        assert report_resp.status_code == 404


# ---------------------------------------------------------------------------
# 9. Async test: full lifecycle with AsyncClient
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_create_and_retrieve(app, tiny_onnx_bytes):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create
        create_resp = await ac.post(
            "/api/scans",
            files={"file": ("model.onnx", io.BytesIO(tiny_onnx_bytes), "application/octet-stream")},
        )
        assert create_resp.status_code in (200, 202)
        scan_id = create_resp.json()["scan_id"]

        # Retrieve record
        get_resp = await ac.get(f"/api/scans/{scan_id}")
        assert get_resp.status_code == 200
        record = get_resp.json()
        assert record["id"] == scan_id
        assert record["filename"] == "model.onnx"
        assert record["status"] in ("queued", "running", "complete", "failed")
