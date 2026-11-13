import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

AUDIT_PATH = Path("data/audit.jsonl")
GENESIS = "0" * 64
PACKAGES = (
    "pandas",
    "numpy",
    "scipy",
    "arch",
    "nautilus_trader",
    "empyrical-reloaded",
    "torch",
    "stable-baselines3",
    "gymnasium",
)


class Entry(BaseModel):
    seq: int
    time: str
    kind: str
    payload: dict[str, Any]
    previous: str
    hash: str


class Verification(BaseModel):
    valid: bool
    entries: int
    broken_at: int | None = None


def digest(value: Any) -> str:
    """SHA-256 of the value's canonical JSON, so equal content always gives an equal hash."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def chain_hash(entry: dict[str, Any]) -> str:
    return digest({key: entry[key] for key in ("seq", "time", "kind", "payload", "previous")})


class AuditLog:
    """An append-only JSON-lines log in which every entry carries the hash of the one before.

    Verification catches an entry edited, removed or reordered anywhere but the end. It cannot
    see entries cut off the end or a file rewritten with every hash recomputed, and it does not
    show that what was recorded is true, so entries hold the inputs needed to re-run and compare.
    """

    def __init__(self, path: Path = AUDIT_PATH) -> None:
        self.path = path
        self.lock = RLock()

    def entries(self, limit: int | None = None) -> list[Entry]:
        if not self.path.exists():
            return []
        lines = [line for line in self.path.read_text().splitlines() if line.strip()]
        found = [Entry.model_validate_json(line) for line in lines]
        return found if limit is None else found[-limit:]

    def record(self, kind: str, payload: dict[str, Any]) -> Entry:
        with self.lock:
            last = self.entries(1)
            body: dict[str, Any] = {
                "seq": last[0].seq + 1 if last else 1,
                "time": datetime.now(UTC).isoformat(),
                "kind": kind,
                "payload": to_jsonable_python(payload, inf_nan_mode="null", fallback=str),
                "previous": last[0].hash if last else GENESIS,
            }
            entry = Entry(**body, hash=chain_hash(body))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as log:
                log.write(entry.model_dump_json() + "\n")
            return entry

    def verify(self) -> Verification:
        previous = GENESIS
        found = self.entries()
        for position, entry in enumerate(found, start=1):
            body = entry.model_dump(mode="json")
            if (
                entry.seq != position
                or entry.previous != previous
                or entry.hash != chain_hash(body)
            ):
                return Verification(valid=False, entries=len(found), broken_at=position)
            previous = entry.hash
        return Verification(valid=True, entries=len(found))

    def count(self, kind: str) -> int:
        return sum(entry.kind == kind for entry in self.entries())


def fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """What data a run saw: its span and a hash of every value, so any changed cell shows.

    Works for a price table indexed by date and for the long layout (date, tic, ...) alike.
    """
    content = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    content.update(",".join(map(str, frame.columns)).encode())
    if isinstance(frame.index, pd.DatetimeIndex):
        first, last = frame.index[0].isoformat(), frame.index[-1].isoformat()
    else:
        first, last = str(frame["date"].min()), str(frame["date"].max())
    symbols = sorted(frame["tic"].unique()) if "tic" in frame else list(frame.columns)
    return {
        "symbols": [str(symbol) for symbol in symbols],
        "rows": len(frame),
        "first": first,
        "last": last,
        "sha256": content.hexdigest(),
    }


def git_state(root: Path | None = None) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip()

    commit, status = run("rev-parse", "HEAD"), run("status", "--porcelain")
    return {"commit": commit, "uncommitted_changes": None if status is None else bool(status)}


def environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "packages": versions}


def manifest(
    config: dict[str, Any], data: dict[str, pd.DataFrame], seed: int | None = None
) -> dict[str, Any]:
    """Everything needed to re-run and compare: config, data, code and environment."""
    return {
        "config": config,
        "config_hash": digest(config),
        "data": {name: fingerprint(frame) for name, frame in data.items()},
        "git": git_state(),
        "environment": environment(),
        "seed": seed,
    }
