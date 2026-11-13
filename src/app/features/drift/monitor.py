import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Lock

import numpy as np
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel
from river import drift
from scipy.stats import ks_2samp, wasserstein_distance

from app.core import audit
from app.core.drift import ReturnSource, ReturnStream
from app.features.drift.schemas import Action, DriftEvent, DriftStatus, Kind, StreamStatus

logger = logging.getLogger(__name__)

PATH = Path("data/drift.json")
JOB_ID = "drift-check"
MIN_LIVE = 60
MIN_RECENT = 10
RECENT_DAYS = 60
KEPT_EVENTS = 200
SHAPES: dict[Kind, np.ufunc] = {"mean": np.positive, "volatility": np.abs}


@dataclass(frozen=True)
class Rules:
    action: Action = "act"
    threshold: float = 0.1
    significance: float = 0.001
    scale: float = 0.5
    hold_days: int = 30
    baseline_days: int = 500
    delta: float = 0.002


RULES = Rules()


class State(BaseModel):
    started: date | None = None
    recorded: list[DriftEvent] = []


def utc_today() -> date:
    return datetime.now(UTC).date()


def distance_of(baseline: np.ndarray, recent: np.ndarray) -> float:
    """The Wasserstein distance between two samples measured in baseline standard deviations."""
    return float(wasserstein_distance(baseline, recent))


def judged(
    name: str, kind: Kind, day: date, recent: np.ndarray, baseline: np.ndarray, rules: Rules
) -> DriftEvent | None:
    """A change the detector found, kept only when it is large (distance) and not noise (p)."""
    if len(recent) < MIN_RECENT:
        return None
    shape = SHAPES[kind]
    distance = distance_of(baseline, recent)
    p_value = float(ks_2samp(shape(baseline), shape(recent)).pvalue)
    if distance < rules.threshold or p_value >= rules.significance:
        return None
    adverse = bool(recent.mean() < 0) if kind == "mean" else bool(recent.std() > 1)
    return DriftEvent(
        stream=name,
        time=day,
        kind=kind,
        adverse=adverse,
        distance=distance,
        p_value=p_value,
        window=len(recent),
    )


def scan(stream: ReturnStream, rules: Rules) -> StreamStatus:
    """Changes of mean or volatility since the baseline: ADWIN raises them, 60 days confirm."""
    values = stream.returns.dropna()
    status = StreamStatus(name=stream.name, kind=stream.kind, monitored=False, days=len(values))
    if values.empty:
        return status.model_copy(update={"note": "No returns yet"})
    seen = {"first": values.index[0].date(), "last": values.index[-1].date()}
    status = status.model_copy(update=seen)
    needed = rules.baseline_days + MIN_LIVE
    if len(values) < needed:
        return status.model_copy(update={"note": f"Needs {needed} days; has {len(values)}"})
    base = values.iloc[: rules.baseline_days]
    if not base.std() > 0:
        return status.model_copy(update={"note": "The baseline never moves"})
    standard = (values - base.mean()) / base.std()
    moves, baseline = standard.to_numpy(), standard.iloc[: rules.baseline_days].to_numpy()
    found: list[DriftEvent] = []
    for kind, shape in SHAPES.items():
        watcher = drift.ADWIN(delta=rules.delta, clock=1)
        armed = -1
        for index, value in enumerate(moves):
            watcher.update(float(shape(value)))
            if index < rules.baseline_days:
                continue
            armed = index + RECENT_DAYS if watcher.drift_detected else armed
            if index <= armed:
                recent = moves[max(0, index + 1 - RECENT_DAYS) : index + 1]
                event = judged(
                    stream.name, kind, standard.index[index].date(), recent, baseline, rules
                )
                if event:
                    found.append(event)
                    armed = -1
    return status.model_copy(
        update={
            "monitored": True,
            "distance": distance_of(baseline, moves[-RECENT_DAYS:]),
            "events": sorted(found, key=lambda event: event.time),
        }
    )


class DriftMonitor:
    """Checks each strategy and agent for drift every day and holds the size cut it asks for."""

    def __init__(
        self,
        rules: Rules = RULES,
        log: audit.AuditLog | None = None,
        path: Path = PATH,
        today: Callable[[], date] = utc_today,
    ) -> None:
        self.rules = rules
        self.log = log
        self.path = path
        self.today = today
        self.sources: list[ReturnSource] = []
        self.retrain: Callable[[str], str] = lambda name: "not applicable"
        self.lock = Lock()
        self.state = self._load()
        self.statuses: list[StreamStatus] = []
        self.checked: datetime | None = None
        self.checked_on: date | None = None
        self.scheduler = AsyncIOScheduler(timezone=UTC)

    def _load(self) -> State:
        try:
            return State.model_validate_json(self.path.read_text())
        except (FileNotFoundError, ValueError):
            return State()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.state.model_dump_json(indent=2))

    def refresh(self) -> list[DriftEvent]:
        """Scan every stream, record and act on the events that are new, and return them."""
        with self.lock:
            if self.rules.action == "off":
                self.statuses = []
                self.checked, self.checked_on = datetime.now(UTC), self.today()
                return []
            if self.state.started is None:
                self.state.started = self.today()
            streams = [stream for source in self.sources for stream in source.streams()]
            scans = [scan(stream, self.rules) for stream in streams]
            known = {(event.stream, event.time, event.kind) for event in self.state.recorded}
            fresh = [
                self._act(event)
                for status in scans
                for event in status.events
                if event.time >= self.state.started
                and (event.stream, event.time, event.kind) not in known
            ]
            self.state.recorded = [*self.state.recorded, *fresh][-KEPT_EVENTS:]
            self._save()
            self.statuses = [self._held(status) for status in scans]
            self.checked, self.checked_on = datetime.now(UTC), self.today()
            return fresh

    def _act(self, event: DriftEvent) -> DriftEvent:
        """Log the event; an adverse one with the action on also cuts size and asks a retrain."""
        acts = event.adverse and self.rules.action == "act"
        recorded = event.model_copy(
            update={
                "scale": self.rules.scale if acts else None,
                "retrain": self._retrain(event.stream) if acts else None,
            }
        )
        if self.log is not None:
            self.log.record("drift", {"event": recorded.model_dump(mode="json")})
        return recorded

    def _retrain(self, name: str) -> str:
        try:
            return self.retrain(name)
        except Exception:
            logger.exception("The retraining of %s could not be scheduled", name)
            return "could not be scheduled"

    def _held(self, status: StreamStatus) -> StreamStatus:
        """The status with the size cut that is in force: the days since its last adverse event."""
        started = self.state.started or self.today()
        adverse = [
            event.time
            for event in status.events
            if event.adverse and event.time >= started and self.rules.action == "act"
        ]
        until = max(adverse) + timedelta(days=self.rules.hold_days) if adverse else None
        cut = until is not None and self.today() <= until
        return status.model_copy(
            update={"until": until if cut else None, "scale": self.rules.scale if cut else 1.0}
        )

    def fresh(self) -> None:
        """Check once a day, whenever the first question of the day arrives."""
        if self.checked_on is None or self.checked_on < self.today():
            try:
                self.refresh()
            except Exception:
                logger.exception("The drift check failed; the last answer stands")

    def scale(self, name: str) -> float:
        """The multiplier on a strategy's positions while it is in a drift hold."""
        self.fresh()
        return next((status.scale for status in self.statuses if status.name == name), 1.0)

    def status(self) -> DriftStatus:
        self.fresh()
        return DriftStatus(
            action=self.rules.action,
            threshold=self.rules.threshold,
            significance=self.rules.significance,
            scale=self.rules.scale,
            hold_days=self.rules.hold_days,
            baseline_days=self.rules.baseline_days,
            started=self.state.started,
            checked=self.checked,
            streams=self.statuses,
            recorded=list(reversed(self.state.recorded)),
        )

    async def _check(self) -> None:
        await asyncio.to_thread(self.refresh)

    def start(self) -> None:
        self.scheduler.add_job(
            self._check,
            CronTrigger(hour=0, minute=1, timezone=UTC),
            id=JOB_ID,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
