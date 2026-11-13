import asyncio
import hashlib
import inspect
import logging
import subprocess
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from threading import Lock
from typing import Any, Literal

import pandas as pd
import sqlalchemy as sa
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from app.core.database import transaction
from app.core.tables import lifecycle_events, strategies, strategy_decisions, strategy_versions
from app.features.lifecycle import gates
from app.features.lifecycle.gates import Gate, Track
from app.features.systematic import backtest, signals
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.schemas import StrategySpec
from app.quant import history, performance

logger = logging.getLogger(__name__)

Stage = Literal["research", "validated", "shadow", "canary", "live", "retired"]
Actor = Literal["auto", "manual", "bootstrap"]
WORDING = frozenset({"title", "description", "note"})
REWORDED = "Only its wording changed; the rule and the code are the same, so it keeps its stage"
ENGINE = "vectorized-daily"
NEXT: dict[str, Stage] = {
    "research": "validated",
    "validated": "shadow",
    "shadow": "canary",
    "canary": "live",
}
CAPITAL: dict[str, float] = {"canary": 0.1, "live": 1.0}
TRADING: tuple[Stage, ...] = ("canary", "live")
APPROVED_BY_HAND: tuple[Stage, ...] = ("live",)
CODE = (signals, backtest, performance)
Prices = Callable[[StrategySpec], history.Prices]
JOB_ID = "lifecycle"
EVERY_HOURS = 1


class ApprovalError(ValueError):
    """Raised when a promotion is approved that is not waiting for approval."""


class Version(BaseModel):
    id: uuid.UUID
    strategy: str
    code_hash: str
    git_sha: str | None
    params: dict[str, Any]
    engine: str
    data_start: date | None
    data_end: date | None
    metrics: dict[str, Any]
    created: datetime


class StrategyState(BaseModel):
    name: str
    title: str
    stage: Stage
    capital: float
    stage_since: datetime
    version: Version
    next_stage: Stage | None
    gates: list[Gate] = []
    evaluated: datetime | None = None
    awaiting_approval: bool = False


class Event(BaseModel):
    id: int
    strategy: str
    time: datetime
    from_stage: str | None
    to_stage: str
    version_id: uuid.UUID
    actor: str
    reason: str
    gates: list[Gate]


def code_hash(spec: StrategySpec) -> str:
    """The rule's parameters and the code that turns them into weights and returns."""
    digest = hashlib.sha256(spec.model_dump_json().encode())
    for module in CODE:
        digest.update(inspect.getsource(module).encode())
    return digest.hexdigest()


def git_sha() -> str | None:
    try:
        found = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return found.stdout.strip() or None


def by_day(frame: pd.DataFrame, days: Iterable[date] | None = None) -> dict[date, dict[str, float]]:
    wanted = None if days is None else set(days)
    return {
        stamp.date(): {str(name): float(value) for name, value in row.items() if value}
        for stamp, row in frame.iterrows()
        if wanted is None or stamp.date() in wanted
    }


class Lifecycle:
    """Moves each strategy from research to live through its gates, one stage at a time.

    The version a strategy is promoted under is the hash of its parameters and code; a change to
    either sends it back to research, so what trades is always what passed.
    """

    def __init__(self, prices: Prices, now: Callable[[], datetime] | None = None) -> None:
        self.prices = prices
        self.now = now or (lambda: datetime.now(UTC))
        self.lock = Lock()
        self.last: dict[str, tuple[datetime, list[Gate]]] = {}
        self.scheduler = AsyncIOScheduler(timezone=UTC)

    async def _evaluate(self) -> None:
        await asyncio.to_thread(self.evaluate)

    def start(self) -> None:
        """Judge now and every hour; deciding twice a day is harmless, so a sleeping laptop
        catches up on waking."""
        self.scheduler.add_job(
            self._evaluate,
            IntervalTrigger(hours=EVERY_HOURS),
            id=JOB_ID,
            coalesce=True,
            max_instances=1,
            next_run_time=datetime.now(UTC),
            misfire_grace_time=None,
        )
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def bootstrap(self, trading_now: Iterable[str]) -> None:
        """Strategies the autopilot already trades start live; the rest start in research."""
        trading = set(trading_now)
        with self.lock, transaction() as connection:
            known = {row.name for row in connection.execute(sa.select(strategies.c.name))}
            for name, spec in STRATEGIES.items():
                if name in known:
                    continue
                version = self.ensure_version(connection, spec, None)
                stage: Stage = "live" if name in trading else "research"
                reason = (
                    "Traded by the autopilot before the lifecycle existed"
                    if stage == "live"
                    else "New to the lifecycle"
                )
                self.move(connection, name, None, stage, version.id, "bootstrap", reason, [])

    def ensure_version(
        self, connection: Connection, spec: StrategySpec, track: Track | None
    ) -> Version:
        found = connection.execute(
            sa.select(strategy_versions).where(
                strategy_versions.c.strategy == spec.name,
                strategy_versions.c.code_hash == code_hash(spec),
            )
        ).one_or_none()
        if found is not None:
            return Version.model_validate(found._mapping)
        metrics: dict[str, Any] = {}
        start = end = None
        if track is not None and track.days:
            metrics = {
                "sharpe": track.sharpe(),
                "max_drawdown": track.drawdown(),
                "fold_sharpes": gates.fold_sharpes(track),
                "days": track.days,
            }
            start, end = track.returns.index[0].date(), track.returns.index[-1].date()
        row = connection.execute(
            sa.insert(strategy_versions)
            .values(
                id=uuid.uuid4(),
                strategy=spec.name,
                code_hash=code_hash(spec),
                git_sha=git_sha(),
                params=spec.model_dump(mode="json"),
                engine=ENGINE,
                data_start=start,
                data_end=end,
                metrics=metrics,
            )
            .returning(*strategy_versions.c)
        ).one()
        return Version.model_validate(row._mapping)

    def move(
        self,
        connection: Connection,
        name: str,
        from_stage: str | None,
        to_stage: Stage,
        version_id: uuid.UUID,
        actor: Actor,
        reason: str,
        passed: list[Gate],
    ) -> None:
        now = self.now()
        values = {
            "stage": to_stage,
            "version_id": version_id,
            "capital": CAPITAL.get(to_stage, 0.0),
            "stage_since": now,
            "updated": now,
        }
        statement = insert(strategies).values(name=name, **values)
        connection.execute(
            statement.on_conflict_do_update(index_elements=[strategies.c.name], set_=values)
        )
        connection.execute(
            sa.insert(lifecycle_events).values(
                strategy=name,
                time=now,
                from_stage=from_stage,
                to_stage=to_stage,
                version_id=version_id,
                actor=actor,
                reason=reason,
                gates=[gate.model_dump() for gate in passed],
            )
        )
        logger.info("Strategy %s moved from %s to %s: %s", name, from_stage, to_stage, reason)

    def evaluate(self) -> list[Event]:
        """Record today's decisions and apply each strategy's next gate; returns the moves."""
        moved: list[Event] = []
        for name, spec in STRATEGIES.items():
            try:
                prices = self.prices(spec)
            except Exception:
                logger.exception("No prices to judge %s", name)
                continue
            with self.lock, transaction() as connection:
                event = self.evaluate_one(connection, spec, prices)
            if event is not None:
                moved.append(event)
        return moved

    def evaluate_one(
        self, connection: Connection, spec: StrategySpec, prices: history.Prices
    ) -> Event | None:
        closes = prices.closes
        returns = backtest.daily_returns(closes, spec, opens=prices.opens)
        full = Track(returns, spec.periods_per_year)
        version = self.ensure_version(connection, spec, full)
        row = connection.execute(
            sa.select(strategies).where(strategies.c.name == spec.name)
        ).one_or_none()
        if row is None:
            self.move(connection, spec.name, None, "research", version.id, "auto", "New", [])
            return self.latest_event(connection, spec.name)
        held = signals.weights(closes, spec)
        finished = held.loc[held.index.date < self.now().date()]
        if len(finished):
            self.record(connection, spec.name, version.id, finished.iloc[[-1]])
        if row.version_id != version.id and self.reworded(connection, row.version_id, spec):
            self.adopt(connection, spec.name, row.stage, row.version_id, version.id)
            return self.latest_event(connection, spec.name)
        if row.version_id != version.id:
            reason = "Its parameters or code changed; the new version starts in research"
            self.move(connection, spec.name, row.stage, "research", version.id, "auto", reason, [])
            return self.latest_event(connection, spec.name)
        found = self.gates_for(connection, spec, row, full, held)
        self.last[spec.name] = (self.now(), found)
        following = NEXT.get(row.stage)
        if following is None or not found or not all(gate.passed for gate in found):
            return None
        if following in APPROVED_BY_HAND:
            return None
        reason = f"Passed every gate to {following}"
        self.move(connection, spec.name, row.stage, following, version.id, "auto", reason, found)
        return self.latest_event(connection, spec.name)

    @staticmethod
    def reworded(connection: Connection, held: uuid.UUID, spec: StrategySpec) -> bool:
        """The version a strategy holds differs from `spec` only in wording, under the same code."""
        row = connection.execute(
            sa.select(strategy_versions.c.params, strategy_versions.c.code_hash).where(
                strategy_versions.c.id == held
            )
        ).one_or_none()
        if row is None:
            return False
        try:
            before = StrategySpec.model_validate(row.params)
        except ValueError:
            return False
        if code_hash(before) != row.code_hash:
            return False
        return before.model_dump(exclude=set(WORDING)) == spec.model_dump(exclude=set(WORDING))

    def adopt(
        self,
        connection: Connection,
        name: str,
        stage: str,
        held: uuid.UUID,
        version_id: uuid.UUID,
    ) -> None:
        """Move a strategy to a reworded version without restarting it: same stage, same day
        count, and the decisions it logged so far count for the new version."""
        now = self.now()
        connection.execute(
            sa.update(strategies)
            .where(strategies.c.name == name)
            .values(version_id=version_id, updated=now)
        )
        connection.execute(
            sa.update(strategy_decisions)
            .where(strategy_decisions.c.strategy == name, strategy_decisions.c.version_id == held)
            .values(version_id=version_id)
        )
        connection.execute(
            sa.insert(lifecycle_events).values(
                strategy=name,
                time=now,
                from_stage=stage,
                to_stage=stage,
                version_id=version_id,
                actor="auto",
                reason=REWORDED,
                gates=[],
            )
        )
        logger.info("Strategy %s keeps %s: %s", name, stage, REWORDED)

    def gates_for(
        self,
        connection: Connection,
        spec: StrategySpec,
        row: Any,
        full: Track,
        held: pd.DataFrame,
    ) -> list[Gate]:
        since = row.stage_since.date()
        if row.stage == "research":
            return [gates.research_gate(full)]
        if row.stage == "validated":
            recorded = self.decisions(connection, spec.name, row.version_id, since)
            return [gates.validation_gate(recorded, by_day(held, recorded))]
        stretch = Track(full.returns.loc[full.returns.index.date >= since], full.periods_per_year)
        if row.stage == "shadow":
            return gates.shadow_gate(stretch)
        if row.stage == "canary":
            return gates.canary_gate(stretch, self.shadow_sharpe(connection, spec.name))
        return []

    @staticmethod
    def record(
        connection: Connection, name: str, version_id: uuid.UUID, latest: pd.DataFrame
    ) -> None:
        """Keep the decision as it was taken; a later replay is compared with it, never it."""
        for session, weights in by_day(latest).items():
            connection.execute(
                insert(strategy_decisions)
                .values(strategy=name, session=session, version_id=version_id, weights=weights)
                .on_conflict_do_nothing()
            )

    @staticmethod
    def decisions(
        connection: Connection, name: str, version_id: uuid.UUID, since: date
    ) -> dict[date, dict[str, float]]:
        rows = connection.execute(
            sa.select(strategy_decisions).where(
                strategy_decisions.c.strategy == name,
                strategy_decisions.c.version_id == version_id,
                strategy_decisions.c.session >= since,
            )
        )
        return {row.session: dict(row.weights) for row in rows}

    @staticmethod
    def shadow_sharpe(connection: Connection, name: str) -> float | None:
        row = connection.execute(
            sa.select(lifecycle_events.c.gates)
            .where(lifecycle_events.c.strategy == name, lifecycle_events.c.to_stage == "canary")
            .order_by(lifecycle_events.c.id.desc())
            .limit(1)
        ).one_or_none()
        if row is None:
            return None
        for gate in row.gates:
            if gate["name"].startswith("Shadow Sharpe"):
                return float(gate["value"]) if gate.get("value") is not None else None
        return None

    @staticmethod
    def latest_event(connection: Connection, name: str) -> Event:
        row = connection.execute(
            sa.select(lifecycle_events)
            .where(lifecycle_events.c.strategy == name)
            .order_by(lifecycle_events.c.id.desc())
            .limit(1)
        ).one()
        return Event.model_validate(row._mapping)

    def approve(self, name: str, reason: str) -> StrategyState:
        """A person's approval of a promotion the gates leave to them: canary to live."""
        with self.lock, transaction() as connection:
            row = connection.execute(
                sa.select(strategies).where(strategies.c.name == name)
            ).one_or_none()
            if row is None:
                raise LookupError(name)
            following = NEXT.get(row.stage)
            _, found = self.last.get(name, (None, []))
            if following not in APPROVED_BY_HAND:
                raise ApprovalError(f"{name} is {row.stage}; nothing waits for approval")
            if not found or not all(gate.passed for gate in found):
                raise ApprovalError(f"{name} has not passed every gate to {following}")
            self.move(
                connection, name, row.stage, following, row.version_id, "manual", reason, found
            )
        return self.state(name)

    def set_stage(self, name: str, stage: Stage, reason: str) -> StrategyState:
        """A person's override: any stage, recorded with their reason."""
        with self.lock, transaction() as connection:
            row = connection.execute(
                sa.select(strategies).where(strategies.c.name == name)
            ).one_or_none()
            if row is None:
                raise LookupError(name)
            self.move(connection, name, row.stage, stage, row.version_id, "manual", reason, [])
        return self.state(name)

    def states(self) -> list[StrategyState]:
        with transaction() as connection:
            rows = connection.execute(
                sa.select(strategies, strategy_versions)
                .join(strategy_versions, strategies.c.version_id == strategy_versions.c.id)
                .order_by(strategies.c.name)
            ).mappings()
            return [self.view(row) for row in rows]

    def state(self, name: str) -> StrategyState:
        found = [item for item in self.states() if item.name == name]
        if not found:
            raise LookupError(name)
        return found[0]

    def view(self, row: Any) -> StrategyState:
        name = row["name"]
        evaluated, found = self.last.get(name, (None, []))
        spec = STRATEGIES.get(name)
        return StrategyState(
            name=name,
            title=spec.title if spec else name,
            stage=row["stage"],
            capital=row["capital"],
            stage_since=row["stage_since"],
            version=Version(
                id=row["version_id"],
                strategy=name,
                code_hash=row["code_hash"],
                git_sha=row["git_sha"],
                params=row["params"],
                engine=row["engine"],
                data_start=row["data_start"],
                data_end=row["data_end"],
                metrics=row["metrics"],
                created=row["created"],
            ),
            next_stage=NEXT.get(row["stage"]),
            gates=found,
            evaluated=evaluated,
            awaiting_approval=NEXT.get(row["stage"]) in APPROVED_BY_HAND
            and bool(found)
            and all(gate.passed for gate in found),
        )

    def events(self, limit: int = 100) -> list[Event]:
        with transaction() as connection:
            rows = connection.execute(
                sa.select(lifecycle_events).order_by(lifecycle_events.c.id.desc()).limit(limit)
            )
            return [Event.model_validate(row._mapping) for row in rows]

    def trading(self) -> list[str]:
        """The strategies the autopilot trades: those in canary or live."""
        with transaction() as connection:
            rows = connection.execute(
                sa.select(strategies.c.name).where(strategies.c.stage.in_(TRADING))
            )
            return [row.name for row in rows]

    def capital(self, name: str) -> float:
        """The share of its allocation a strategy trades with: a tenth in canary, all when live."""
        with transaction() as connection:
            found = connection.execute(
                sa.select(strategies.c.capital).where(strategies.c.name == name)
            ).scalar_one_or_none()
        return 0.0 if found is None else float(found)
