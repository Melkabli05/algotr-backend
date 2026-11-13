import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

SIDECAR = Path(__file__).resolve().parents[3] / "finrl_sidecar"
RUNNER = SIDECAR / "runner.py"
TIMEOUT_SECONDS = 1800
LIMIT_MARGIN = 30
MIN_LIMIT = 60


class SidecarUnavailableError(RuntimeError):
    """Raised when the FinRL sidecar environment has not been set up."""


class JobResult(BaseModel):
    task: str
    status: str
    seconds: float
    data: dict[str, Any] | None = None
    error: str | None = None


def sidecar_python() -> Path:
    return SIDECAR / ".venv" / "bin" / "python"


def available() -> bool:
    return sidecar_python().exists() and RUNNER.exists()


def sealed_environment(home: Path) -> dict[str, str]:
    """What the FinRL process may see: a path and a scratch home, never a broker or API secret."""
    return {"PATH": os.defpath, "HOME": str(home), "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"}


def run_job(
    task: str,
    params: dict[str, Any],
    data: dict[str, pd.DataFrame],
    timeout: float = TIMEOUT_SECONDS,
) -> JobResult:
    """Run one FinRL task in the sidecar; the prices go in as files and the result comes back."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        files = {}
        for name, frame in data.items():
            files[name] = str(root / f"{name}.csv")
            frame.to_csv(files[name], index=False)
        job, result = root / "job.json", root / "result.json"
        limit = max(int(timeout) - LIMIT_MARGIN, MIN_LIMIT)
        job.write_text(json.dumps({"task": task, "files": files, "params": params, "limit": limit}))
        try:
            subprocess.run(  # noqa: S603
                [str(sidecar_python()), str(RUNNER), str(job), str(result)],
                env=sealed_environment(root),
                cwd=root,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return JobResult(
                task=task,
                status="error",
                seconds=round(time.monotonic() - started, 2),
                error=f"Did not finish in {timeout:.0f}s",
            )
        if not result.exists():
            return JobResult(
                task=task,
                status="error",
                seconds=round(time.monotonic() - started, 2),
                error="The FinRL process ended without a result",
            )
        outcome = json.loads(result.read_text())
    return JobResult(
        task=task,
        status=outcome["status"],
        seconds=outcome["seconds"],
        data=outcome.get("data"),
        error=outcome.get("error"),
    )
