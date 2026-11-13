from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, Self

import sqlalchemy as sa
from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import ActorId, PriceType
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from app.bridge import commands
from app.core import breakers
from app.core.agents import ENSEMBLE
from app.core.tables import arbiter_runs, autopilot, strategy_targets
from app.features.models import health
from trading_node import metrics
from trading_node.alpaca_adapter.constants import ALPACA_VENUE
from trading_node.alpaca_adapter.parsing import USD, instrument_id_of, symbol_of
from trading_node.arbiter import (
    TARGETS_TOPIC,
    Arbitration,
    Limits,
    StrategyTarget,
    arbitrate,
)
from trading_node.shadow import ShadowExecutionClient
from trading_node.state_writer import StateWriter
from trading_node.strategies.rule import next_daily

REBALANCE_AT = timedelta(minutes=15)
BREAKER_CUT = 0.5
AGENTS = ("ppo", "sac", "a2c")
WATCH_EVERY = timedelta(minutes=1)
RELOAD_REASON = "The lifecycle changed; the node restarts to load its strategies"
EXPIRES = timedelta(minutes=2)
TRADING_MODE = "paper"


def target_row(target: StrategyTarget, source: str) -> dict[str, Any]:
    return {
        "strategy": target.strategy,
        "source": source,
        "stage": target.stage,
        "allocation": target.allocation,
        "weights": dict(target.weights),
        "version_id": target.version_id,
        "session": target.session,
        "updated": datetime.now(UTC),
    }


def api_targets(connection: Connection) -> list[StrategyTarget]:
    """What the API's own sources want held: the agents' ensemble and the scanner's sleeve."""
    rows = connection.execute(sa.select(strategy_targets).where(strategy_targets.c.source == "api"))
    return [
        StrategyTarget(
            strategy=row.strategy,
            stage=row.stage,
            allocation=row.allocation,
            weights=dict(row.weights),
            version_id=row.version_id,
            session=None if row.session is None else row.session.isoformat(),
        )
        for row in rows
    ]


def subjects(target: StrategyTarget) -> set[str]:
    """Whose drift governs a target: the strategy's, and for the ensemble its agents'."""
    return {target.strategy, *(AGENTS if target.strategy == ENSEMBLE else ())}


def screen(
    targets: list[StrategyTarget], blocked: dict[str, str], factors: dict[str, float]
) -> tuple[list[StrategyTarget], list[str]]:
    """The risk actor's health check before any order: a technical failure blocks a target, a
    statistical one scales its allocation by the degradation factor."""
    kept, notes = [], []
    for target in targets:
        found = sorted(subjects(target) & set(blocked))
        if found:
            notes.append(f"{target.strategy} blocked, technical failure: {blocked[found[0]]}")
            continue
        scale = min((factors[name] for name in subjects(target) & set(factors)), default=1.0)
        if scale < 1.0:
            target = replace(target, allocation=target.allocation * scale)
            notes.append(f"{target.strategy} at {scale:.0%} of its allocation, statistical failure")
        kept.append(target)
    return kept, notes


def managed_part(
    positions: dict[str, float], managed: set[str], targets: list[StrategyTarget]
) -> dict[str, float]:
    """The positions the arbiter may trade: the autopilot's own and any a target names."""
    named = {symbol for target in targets for symbol in target.weights}
    return {symbol: qty for symbol, qty in positions.items() if symbol in managed | named}


class PortfolioArbiter(DataActor):
    """Turns every strategy's target into one portfolio and the one set of orders to reach it.

    Strategies send targets on the message bus; the API's sources write theirs to Postgres. Once
    a day the arbiter combines the canary and live ones by allocation, applies the limits no
    strategy can override, nets them against what the account holds and sends the result as a
    single rebalance command, so every order still passes the platform's checks and Nautilus'
    RiskEngine. Shadow strategies' orders go to the shadow client instead.
    """

    def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
        return super().__new__(cls)

    def __init__(
        self,
        writer: StateWriter,
        engine: Engine,
        limits: Callable[[], Limits],
        sector_of: Callable[[str], str],
    ) -> None:
        super().__init__(DataActorConfig(actor_id=ActorId("PortfolioArbiter-001")))
        self.writer = writer
        self.engine = engine
        self.limits = limits
        self.sector_of = sector_of
        self.targets: dict[str, StrategyTarget] = {}
        self.shadow: ShadowExecutionClient | None = None
        self.last: Arbitration | None = None
        self.watch: Callable[[], object] | None = None
        self.loaded: object = None

    def on_start(self) -> None:
        self.shadow = ShadowExecutionClient(self.writer, self.book_of, self.quote_of)
        self.subscribe_topic(TARGETS_TOPIC, self.on_target)
        now = self.clock.utc_now()
        self.clock.set_timer(
            "arbiter-daily",
            timedelta(days=1),
            start_time=next_daily(now, REBALANCE_AT),
            callback=lambda _: self.rebalance(),
        )
        if self.watch is not None:
            self.loaded = self.watch()
            self.clock.set_timer("lifecycle-watch", WATCH_EVERY, callback=lambda _: self.reload())

    def reload(self) -> bool:
        """A promotion is a database write: the node restarts to load what the table now says."""
        if self.watch is None or self.watch() == self.loaded:
            return False
        self.log.warning(RELOAD_REASON)
        self.shutdown_system(RELOAD_REASON)
        return True

    def on_target(self, message: object) -> None:
        if not isinstance(message, StrategyTarget):
            return
        self.targets[message.strategy] = message
        row = target_row(message, "node")

        def work(connection: Connection) -> None:
            statement = insert(strategy_targets).values(**row)
            keep = {key: value for key, value in row.items() if key != "strategy"}
            connection.execute(
                statement.on_conflict_do_update(index_elements=["strategy"], set_=keep)
            )

        self.writer.put(work)

    def book_of(self, symbol: str) -> Any:
        return self.cache.order_book(instrument_id_of(symbol))

    def quote_of(self, symbol: str) -> Any:
        return self.cache.quote(instrument_id_of(symbol))

    def price_of(self, symbol: str, targets: list[StrategyTarget]) -> float | None:
        instrument_id = instrument_id_of(symbol)
        for kind in (PriceType.LAST, PriceType.MID):
            found = self.cache.price(instrument_id, kind)
            if found is not None:
                return found.as_double()
        for target in targets:
            closes = target.closes.get(symbol)
            if closes:
                return float(closes[-1])
        return None

    def account(
        self, targets: list[StrategyTarget], managed: set[str]
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """The account's whole value, the part of it the arbiter may trade and a price for every
        symbol involved."""
        positions = {
            symbol_of(position.instrument_id): float(position.signed_qty)
            for position in self.cache.positions_open()
        }
        held = managed_part(positions, managed, targets)
        symbols = set(positions) | {symbol for target in targets for symbol in target.weights}
        prices = {
            symbol: price
            for symbol in symbols
            if (price := self.price_of(symbol, targets)) is not None
        }
        found = self.cache.account_for_venue(ALPACA_VENUE)
        balance = None if found is None else found.balance_total(USD)
        cash = 0.0 if balance is None else balance.as_double()
        equity = cash + sum(qty * prices.get(symbol, 0.0) for symbol, qty in positions.items())
        return equity, held, prices

    def rebalance(self) -> Arbitration | None:
        with self.engine.begin() as connection:
            settings = connection.execute(sa.select(autopilot)).one_or_none()
            targets = [*self.targets.values(), *api_targets(connection)]
            stopped = breakers.stopped(connection)
            paused = breakers.paused(connection)
            now = datetime.now(UTC)
            found = health.read(connection, now)
            blocked, factors = health.blocked(found), health.factors(found)
            cut = BREAKER_CUT if breakers.portfolio_cut(connection) else 1.0
        if settings is None or settings.mode == "off":
            return None
        everyone = targets
        targets = [] if stopped else [t for t in targets if t.strategy not in paused]
        targets, drift_notes = screen(targets, blocked, factors)
        managed = set(settings.managed or [])
        equity, held, prices = self.account(everyone, managed)
        result = arbitrate(targets, self.limits(), self.sector_of, equity, held, prices, cut)
        if stopped:
            result.notes.append(f"{stopped}; nothing is held")
        for name in sorted(paused):
            result.notes.append(f"{name} paused: {paused[name]}")
        result.notes.extend(drift_notes)
        self.last = result
        for target in targets:
            if target.stage == "shadow" and self.shadow is not None:
                self.shadow.submit(target, equity, prices)
        trades = settings.mode == TRADING_MODE and bool(result.orders)
        metrics.ARBITER_RUNS.labels("yes" if trades else "no").inc()
        with self.engine.begin() as connection:
            command_id = None
            if trades:
                expires = datetime.now(UTC) + EXPIRES
                payload = {
                    "orders": result.orders,
                    "cancel": sorted({str(order["symbol"]) for order in result.orders}),
                    "expires_at": expires.isoformat(),
                }
                command_id = commands.submit(connection, "rebalance", payload, "arbiter").id
                bought = {str(o["symbol"]) for o in result.orders if o.get("side") == "buy"}
                connection.execute(sa.update(autopilot).values(managed=sorted(managed | bought)))
            connection.execute(
                sa.insert(arbiter_runs).values(
                    equity=equity,
                    targets=[target_row(t, "node") | {"updated": None} for t in targets],
                    weights=result.weights,
                    notes=result.notes,
                    orders=result.orders,
                    command_id=command_id,
                    detail=None if trades else f"Autopilot is {settings.mode}; nothing was sent",
                )
            )
        return result
