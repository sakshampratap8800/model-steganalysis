"""
SQLite database setup and CRUD operations using SQLAlchemy 2.x.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    create_engine,
    select,
    update,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import config


# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

def _make_engine():
    db_url = config.DATABASE_URL
    # Ensure parent directory exists for SQLite file paths
    if db_url.startswith("sqlite:///"):
        db_file = Path(db_url[len("sqlite:///"):])
        db_file.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        db_url,
        connect_args={"check_same_thread": False},  # needed for SQLite + threads
        echo=False,
    )


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class ScanRow(Base):
    """SQLAlchemy ORM model for the `scans` table."""

    __tablename__ = "scans"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    sha256 = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    architecture = Column(String, nullable=True)
    format = Column(String, nullable=True)
    status = Column(String, nullable=False, default="queued")
    created_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    report_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------

def create_tables() -> None:
    """Create all tables. Safe to call multiple times (uses checkfirst)."""
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_scan(
    scan_id: str,
    filename: str,
    sha256: str,
    file_size_bytes: int,
) -> ScanRow:
    """Insert a new scan record with status=queued."""
    row = ScanRow(
        id=scan_id,
        filename=filename,
        sha256=sha256,
        file_size_bytes=file_size_bytes,
        status="queued",
        created_at=_now(),
    )
    with SessionLocal() as session:
        session.add(row)
        session.commit()
        session.refresh(row)
        # Detach from session so it can be used outside
        session.expunge(row)
    return row


def get_scan(scan_id: str) -> Optional[ScanRow]:
    """Return scan row or None if not found."""
    with SessionLocal() as session:
        row = session.get(ScanRow, scan_id)
        if row is not None:
            session.expunge(row)
        return row


def update_scan_status(
    scan_id: str,
    status: str,
    architecture: Optional[str] = None,
    format_str: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update status (and optionally architecture/format/error) for a scan."""
    values: dict = {"status": status}
    if architecture is not None:
        values["architecture"] = architecture
    if format_str is not None:
        values["format"] = format_str
    if error_message is not None:
        values["error_message"] = error_message
    if status in ("complete", "failed"):
        values["completed_at"] = _now()

    with SessionLocal() as session:
        session.execute(update(ScanRow).where(ScanRow.id == scan_id).values(**values))
        session.commit()


def update_scan_report(scan_id: str, report: dict) -> None:
    """Serialize report dict to JSON and persist it."""
    with SessionLocal() as session:
        session.execute(
            update(ScanRow)
            .where(ScanRow.id == scan_id)
            .values(
                report_json=json.dumps(report),
                status="complete",
                completed_at=_now(),
            )
        )
        session.commit()


def list_scans(skip: int = 0, limit: int = 20) -> tuple[List[ScanRow], int]:
    """Return (rows, total_count) with pagination."""
    with SessionLocal() as session:
        total = session.scalar(select(func.count()).select_from(ScanRow)) or 0
        rows = (
            session.execute(
                select(ScanRow).order_by(ScanRow.created_at.desc()).offset(skip).limit(limit)
            )
            .scalars()
            .all()
        )
        for r in rows:
            session.expunge(r)
        return list(rows), total

def reset_orphaned_scans() -> int:
    """Reset any scans stuck in 'running' state to 'failed'."""
    with SessionLocal() as session:
        result = session.execute(
            update(ScanRow)
            .where(ScanRow.status == "running")
            .values(
                status="failed",
                error_message="Scan aborted due to server restart.",
                completed_at=_now(),
            )
        )
        session.commit()
        return result.rowcount
