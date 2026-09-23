"""
ScannerBridge: orchestrates the two-step scan pipeline.

Step 1 (Python): extract_tensors.py  ONNX  -> meta.json + weights.bin
Step 2 (C++):    scanner_cli         scan dir -> JSON report
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from config import config


from datetime import datetime, timezone

class ScannerError(RuntimeError):
    """Raised when a subprocess in the pipeline fails."""

    def __init__(self, step: str, returncode: int, stderr: str) -> None:
        self.step = step
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"[{step}] exited with code {returncode}.\nstderr:\n{stderr}"
        )


class ScannerBridge:
    """
    Orchestrates the two-step scan process:
      1. Python: extract_tensors.py  (ONNX -> intermediate format)
      2. C++:    scanner_cli         (intermediate format -> JSON report)
    """

    # Global semaphore to limit subprocess concurrency
    _semaphore = asyncio.Semaphore(4)

    def __init__(
        self,
        extract_script: Optional[Path] = None,
        scanner_cli: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.extract_script = extract_script or config.EXTRACT_TENSORS_SCRIPT
        self.scanner_cli = scanner_cli or config.SCANNER_CLI
        self.timeout = timeout if timeout is not None else config.SCAN_TIMEOUT_SECONDS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_scan(
        self,
        scan_id: str,
        model_path: Path,
        scan_dir: Path,
    ) -> dict:
        """
        Run the full scan pipeline and return the report dict.

        Raises ScannerError if any step fails.
        """
        async with self._semaphore:
            # Step 1: extract tensors (Python)
            await self._extract_tensors(model_path, scan_dir, scan_id)

            # Step 2: run C++ scanner
            report = await self._run_cpp_scanner(scan_dir)
            return report

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    async def _extract_tensors(
        self,
        model_path: Path,
        scan_dir: Path,
        scan_id: str,
    ) -> None:
        """
        Run extract_tensors.py as a subprocess.

        Command:
          python <extract_script> <model_path> --output-dir <scan_dir> --scan-id <scan_id>
        """
        cmd = [
            sys.executable,
            str(self.extract_script),
            str(model_path),
            "--output-dir", str(scan_dir),
            "--scan-id", scan_id,
        ]

        stdout, stderr, returncode = await self._run_subprocess(
            cmd, step="extract_tensors", timeout=self.timeout
        )

        if returncode != 0:
            raise ScannerError("extract_tensors", returncode, stderr)

    async def _run_cpp_scanner(self, scan_dir: Path) -> dict:
        """
        Run scanner_cli as a subprocess.

        Expected command:
          scanner_cli --scan-dir <scan_dir> --output <scan_dir>/report.json

        Falls back to a synthetic report when the CLI binary is missing
        (development mode) so the API remains functional without the C++ build.
        """
        report_path = scan_dir / "report.json"

        cli_path = Path(self.scanner_cli)
        # Fallback check if it's in the actual cmake output dir
        if not cli_path.exists():
            alt_path = cli_path.parent.parent / "core" / "scanner_cli.exe"
            if alt_path.exists():
                cli_path = alt_path
                
        if not cli_path.exists():
            # Development fallback: generate a stub report from meta.json
            return self._build_stub_report(scan_dir, report_path)

        cmd = [
            str(cli_path),
            str(scan_dir / "meta.json"),
            "--output", str(report_path),
        ]
        
        baseline_path = Path(__file__).parent.parent / "data" / "baselines.json"
        if baseline_path.exists():
            cmd.extend(["--baseline", str(baseline_path)])

        stdout, stderr, returncode = await self._run_subprocess(
            cmd, step="scanner_cli", timeout=self.timeout
        )

        if returncode != 0:
            raise ScannerError("scanner_cli", returncode, stderr)

        # Parse the report written by the CLI
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)

        raise ScannerError(
            "scanner_cli",
            0,
            f"scanner_cli exited 0 but report not found at {report_path}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _run_subprocess(
        cmd: list,
        step: str,
        timeout: int,
    ) -> tuple[str, str, int]:
        """
        Run *cmd* asynchronously, capturing stdout/stderr.

        Returns (stdout_str, stderr_str, returncode).
        Raises asyncio.TimeoutError if the process exceeds *timeout* seconds.
        """
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise asyncio.TimeoutError(
                f"[{step}] process timed out after {timeout}s"
            )

        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        return stdout, stderr, proc.returncode

    @staticmethod
    def _build_stub_report(scan_dir: Path, report_path: Path) -> dict:
        """
        Build a minimal stub report from meta.json when the C++ scanner is absent.
        Used in development / CI environments where only the Python stack is available.
        """
        meta_path = scan_dir / "meta.json"
        meta: dict = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        report = {
            "scan_id": meta.get("scan_id", "unknown"),
            "model_profile": {
                "architecture": meta.get("architecture", "unknown"),
                "format": meta.get("format", "onnx"),
                "total_tensors": meta.get("tensor_count", 0),
                "total_params": meta.get("total_parameter_count", 0),
                "sha256": meta.get("sha256", ""),
                "file_size_bytes": meta.get("file_size_bytes", 0),
                "opset_version": meta.get("opset_version"),
            },
            "tensor_profiles": [],
            "evidence_scores": [
                {
                    "family": "stub",
                    "score": 0.0,
                    "confidence": 0.0,
                    "description": "C++ scanner not available - stub report generated",
                    "details": None,
                }
            ],
            "flagged_layers": [],
            "overall_score": 0.0,
            "verdict": "unknown",
            "verdict_confidence": 0.0,
            "summary": (
                "Scan completed in development mode. C++ scanner binary was not found. "
                "Tensor extraction succeeded; statistical analysis requires the C++ scanner."
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scanner_version": "stub-dev",
            "raw_scanner_output": None,
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report
