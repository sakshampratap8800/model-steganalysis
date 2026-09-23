"""
Backend configuration.

All paths are configurable via environment variables.
"""

import os
from pathlib import Path


class Config:
    # --- Storage ---
    DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data/scans"))
    MODELS_DIR: Path = Path(os.environ.get("MODELS_DIR", "../models/references"))

    # --- Scanner CLI ---
    # Prefer environment variable, then default to build output
    SCANNER_CLI: Path = Path(
        os.environ.get(
            "SCANNER_CLI_PATH",
            str(Path(__file__).parent.parent / "build" / "Debug" / "scanner_cli.exe"),
        )
    )

    # --- Scripts ---
    EXTRACT_TENSORS_SCRIPT: Path = Path(
        os.environ.get(
            "EXTRACT_TENSORS_SCRIPT",
            str(Path(__file__).parent.parent / "scripts" / "extract_tensors.py"),
        )
    )

    # --- Limits ---
    MAX_UPLOAD_SIZE_BYTES: int = int(
        os.environ.get("MAX_UPLOAD_SIZE_BYTES", str(2 * 1024 * 1024 * 1024))  # 2 GB
    )
    SCAN_TIMEOUT_SECONDS: int = int(os.environ.get("SCAN_TIMEOUT_SECONDS", "300"))

    # --- Database ---
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{Path('data/scans') / 'scans.db'}",
    )

    # --- CORS ---
    CORS_ORIGINS: list = os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")


config = Config()
