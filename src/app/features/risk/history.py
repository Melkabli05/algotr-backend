import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel

from app.features.risk.schemas import RiskSummary

logger = logging.getLogger(__name__)

PATH = Path("data/risk_history.jsonl")
EVERY = timedelta(minutes=5)
KEPT = timedelta(days=14)
NEAR = 0.8


class Use(BaseModel):
    name: str
    used: float | None
    value: float | None
    threshold: float


class Snapshot(BaseModel):
    time: datetime
    kill_state: str
    var_95: float | None
    limits: list[Use]


class NearMiss(BaseModel):
    time: datetime
    symbol: str
    side: str
    qty: float
    limit: str
    used: float


def snapshot_of(summary: RiskSummary, kill_state: str, now: datetime) -> Snapshot:
    var = next(
        (row.dollars for row in summary.var if row.confidence == 0.95 and row.horizon == 1), None
    )
    return Snapshot(
        time=now,
        kill_state=kill_state,
        var_95=var,
        limits=[
            Use(name=item.name, used=item.used, value=item.value, threshold=item.threshold)
            for item in summary.limits
            if item.scope == "account"
        ],
    )


def near_misses(entries: Any, since: datetime) -> list[NearMiss]:
    """Accepted orders that used at least 80% of a limit on the way through."""
    found = []
    for entry in entries:
        payload = entry.payload
        if entry.kind != "order" or payload.get("outcome") != "accepted":
            continue
        moment = datetime.fromisoformat(entry.time)
        if moment < since:
            continue
        usage = (payload.get("result") or {}).get("usage") or {}
        request = payload.get("request") or {}
        for name, used in usage.items():
            if used >= NEAR:
                found.append(
                    NearMiss(
                        time=moment,
                        symbol=str(request.get("symbol", "")),
                        side=str(request.get("side", "")),
                        qty=float(request.get("qty", 0) or 0),
                        limit=name,
                        used=float(used),
                    )
                )
    return sorted(found, key=lambda item: item.time, reverse=True)


class RiskRecorder:
    """Samples the risk summary every five minutes so the day's limit use can be replayed."""

    def __init__(
        self,
        sample: Callable[[], Snapshot],
        path: Path = PATH,
    ) -> None:
        self.sample = sample
        self.path = path
        self.lock = Lock()
        self.listeners: list[Callable[[Snapshot], None]] = []
        self.scheduler = AsyncIOScheduler(timezone=UTC)

    def listen(self, callback: Callable[[Snapshot], None]) -> None:
        self.listeners.append(callback)

    def record(self) -> Snapshot:
        found = self.sample()
        cutoff = found.time - KEPT
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            kept = [
                line
                for line in (self.path.read_text().splitlines() if self.path.exists() else [])
                if line and datetime.fromisoformat(json.loads(line)["time"]) >= cutoff
            ]
            kept.append(found.model_dump_json())
            self.path.write_text("\n".join(kept) + "\n")
        for callback in self.listeners:
            try:
                callback(found)
            except Exception:
                logger.exception("A risk snapshot listener failed")
        return found

    def history(self, since: datetime) -> list[Snapshot]:
        if not self.path.exists():
            return []
        rows = [
            Snapshot.model_validate_json(line)
            for line in self.path.read_text().splitlines()
            if line
        ]
        return [row for row in rows if row.time >= since]

    async def _tick(self) -> None:
        try:
            await asyncio.to_thread(self.record)
        except Exception:
            logger.exception("The risk snapshot failed")

    def start(self) -> None:
        self.scheduler.add_job(
            self._tick,
            "interval",
            seconds=EVERY.total_seconds(),
            id="risk-snapshot",
            coalesce=True,
            max_instances=1,
            next_run_time=datetime.now(UTC) + timedelta(seconds=30),
        )
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
