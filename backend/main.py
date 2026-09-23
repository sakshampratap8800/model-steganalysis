"""
FastAPI application entry point for the Model Steganalysis API.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import config
from api.scans import router as scans_router
from storage.database import create_tables


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise database tables, reset orphans, and ensure storage directories exist."""
    create_tables()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    from storage.database import reset_orphaned_scans
    orphans = reset_orphaned_scans()
    if orphans > 0:
        print(f"Reset {orphans} orphaned 'running' scans to 'failed' on startup.")
    yield


app = FastAPI(
    title="Model Steganalysis API",
    version="0.1.0",
    lifespan=lifespan,
    description=(
        "REST API for the AI Model Steganalysis Platform. "
        "Upload ONNX / PyTorch models and receive detailed steganography analysis reports."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(scans_router, prefix="/api")

# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Simple liveness probe."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {"message": "Model Steganalysis API - see /docs for API reference."}


# ---------------------------------------------------------------------------
# Dev-server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
