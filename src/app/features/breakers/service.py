import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import empyrical
import numpy as np
import pandas as pd
import sqlalchemy as sa
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.engine import Connection

from app.bridge import commands, node_state
from app.core import breakers, metrics
from app.core.breakers import Breaker
from app.core.database import transaction
from app.core.settings import Settings
from app.core.tables import autopilot, node_positions, node_reconciliation, strategies
from app.features.models import drift

logger = logging.getLogger(__name__)

JOB_ID = "breakers"
EVERY = timedelta(seconds=30)
SLOW_EVERY = timedelta(hours=1)
DISCONNECT_GRACE = timedelta(seconds=60)
SUSPENDED = 3 * EVERY
FLATTEN_EXPIRES = timedelta(minutes=2)
DRIFT_WINDOW = timedelta(minutes=10)
SHARPE_DAYS = 30
UNCAPPED = ("Fund", "Crypto", "Unclassified")


@dataclass(frozen=True)
class Thresholds:
    strategy_drawdown: float
    strategy_sharpe: float
    portfolio_drawdown: float
    portfolio_var: float
    portfolio_correlation: float
    heartbeat_seconds: int

    @classmethod
    def of(cls, settings: Settings) -> "Thresholds":
        return cls(
            strategy_drawdown=settings.strategy_max_drawdown,
            strategy_sharpe=settings.strategy_min_sharpe,
            portfolio_drawdown=settings.portfolio_max_drawdown,
            portfolio_var=settings.portfolio_var_limit,
            portfolio_correlation=settings.portfolio_max_correlation,
            heartbeat_seconds=settings.heartbeat_stale_seconds,
        )


def drawdown(values: pd.Series) -> float:
    """The fall from the highest value to the latest, as a share of the highest."""
    if values.empty:
        return 0.0
    peak = float(values.max())
    return 0.0 if peak <= 0 else max(0.0, 1 - float(values.iloc[-1]) / peak)


def sharpe(returns: pd.Series, periods: int) -> float:
    value = float(empyrical.sharpe_ratio(returns, annualization=periods)) if len(returns) else 0.0
    return value if np.isfinite(value) else 0.0


def strategy_problem(returns: pd.Series, periods: int, limits: Thresholds) -> str | None:
    """Why a strategy's own returns pause it: its drawdown, or thirty days of poor Sharpe."""
    growth = (1 + returns).cumprod()
    fall = drawdown(growth)
    if fall > limits.strategy_drawdown:
        return f"Drawdown {fall:.1%}, past {limits.strategy_drawdown:.0%}"
    recent = returns.iloc[-SHARPE_DAYS:]
    found = sharpe(recent, periods)
    if len(recent) >= SHARPE_DAYS and found < limits.strategy_sharpe:
        return f"Sharpe over 30 days {found:.2f}, below {limits.strategy_sharpe:g}"
    return None


def portfolio_problem(equity: pd.Series, holdings: pd.DataFrame, limits: Thresholds) -> str | None:
    """Why the whole portfolio is cut: its drawdown, its one-day 95% VaR, or holdings that move
    together. Funds and coins are left out of the last, as the arbiter's limits leave them."""
    fall = drawdown(equity)
    if fall > limits.portfolio_drawdown:
        return f"Portfolio drawdown {fall:.1%}, past {limits.portfolio_drawdown:.0%}"
    returns = equity.pct_change().dropna()
    if len(returns) >= 20:
        var = -float(np.quantile(returns, 0.05))
        if var > limits.portfolio_var:
            return f"One-day 95% VaR {var:.1%} of the account, past {limits.portfolio_var:.0%}"
    if holdings.shape[1] >= 2:
        matrix = holdings.pct_change().dropna().corr().to_numpy()
        pairs = matrix[np.triu_indices_from(matrix, k=1)]
        average = float(np.nanmean(pairs)) if len(pairs) else 0.0
        if average > limits.portfolio_correlation:
            return (
                f"Holdings move together: average correlation {average:.2f}, "
                f"above {limits.portfolio_correlation:g}"
            )
    return None


def flatten(connection: Connection, reason: str, now: datetime, everything: bool = False) -> int:
    """Halt, then sell as one rebalance that expires unless the node takes it at once, since a
    close that waited for the node would act on a picture no longer true. An automatic breaker
    sells only what the autopilot bought; a kill switch, engaged by hand, sells `everything`."""
    commands.set_halted(connection, True, reason)
    if everything:
        commands.submit(connection, "cancel_all", {}, "kill_switch")
    managed: set[str] = set(
        connection.execute(sa.select(autopilot.c.managed)).scalar_one_or_none() or []
    )
    held = {
        row.instrument_id.rsplit(".", 1)[0]: float(row.quantity)
        for row in connection.execute(
            sa.select(node_positions.c.instrument_id, node_positions.c.quantity)
        )
    }
    sells = [
        {"symbol": symbol, "side": "sell", "qty": qty}
        for symbol, qty in sorted(held.items())
        if (everything or symbol in managed) and qty > 0
    ]
    if sells:
        payload = {
            "orders": sells,
            "cancel": sorted(managed | {sell["symbol"] for sell in sells}),
            "expires_at": (now + FLATTEN_EXPIRES).isoformat(),
        }
        commands.submit(
            connection, "rebalance", payload, "kill_switch" if everything else "breaker"
        )
    return len(sells)


class BreakerMonitor:
    """Watches the four breaker levels and acts on the ones it owns: strategy, portfolio and
    system. The trade level lives in the trading node, order by order."""

    def __init__(
        self,
        limits: Thresholds,
        connection_problems: Callable[[datetime], list[str]],
        stale_feeds: Callable[[datetime], list[str]],
        reconciliation_ok: Callable[[], bool],
        strategy_returns: Callable[[str, datetime], tuple[pd.Series, int] | None],
        equity_history: Callable[[], pd.Series],
        holdings: Callable[[], pd.DataFrame],
        alert: Callable[[str, str], None] | None = None,
        drift_scores: Callable[[], dict[str, float]] | None = None,
        retrain: Callable[[str], object] | None = None,
    ) -> None:
        self.limits = limits
        self.connection_problems = connection_problems
        self.stale_feeds = stale_feeds
        self.reconciliation_ok = reconciliation_ok
        self.strategy_returns = strategy_returns
        self.equity_history = equity_history
        self.holdings = holdings
        self.alert = alert
        self.drift_scores = drift_scores
        self.retrain = retrain
        self.since: datetime | None = None
        self.checked: datetime | None = None
        self.slow_at: datetime | None = None
        self.scheduler = AsyncIOScheduler(timezone=UTC)

    def system_problems(self, connection: Connection, now: datetime) -> list[str]:
        found = []
        health = node_state.health(connection, now)
        stale = health.heartbeat is not None and (now - health.heartbeat).total_seconds() > (
            self.limits.heartbeat_seconds
        )
        if stale and health.state not in ("stopped", "never started"):
            found.append(f"Trading node heartbeat missed for {(now - health.heartbeat).seconds}s")  # type: ignore[operator]
        if self.checked is not None and now - self.checked > SUSPENDED:
            self.since = None
        self.checked = now
        problems = self.connection_problems(now)
        if problems:
            self.since = self.since or now
            if now - self.since >= DISCONNECT_GRACE:
                found.append(f"Alpaca disconnected: {problems[0]}")
        else:
            self.since = None
        found.extend(f"Data stale: {feed}" for feed in self.stale_feeds(now))
        drift = connection.execute(
            sa.select(node_reconciliation.c.detail).where(
                node_reconciliation.c.severity == "critical",
                node_reconciliation.c.time >= now - DRIFT_WINDOW,
            )
        ).first()
        if drift is not None:
            found.append(f"Reconciliation drift: {drift.detail}")
        if not self.reconciliation_ok():
            found.append("Reconciliation drift: the ledger and Alpaca's quantities differ")
        return found

    def check_system(self, connection: Connection, now: datetime) -> Breaker | None:
        problems = self.system_problems(connection, now)
        current = breakers.read(connection).get(breakers.SYSTEM)
        if not problems or breakers.blocks(current):
            return None
        reason = "; ".join(problems)
        tripped = breakers.trip(connection, breakers.SYSTEM, "system", reason, now)
        closed = flatten(connection, f"System breaker: {reason}", now)
        self.notify("System breaker open", f"{reason}. Trading halted, {closed} positions closing")
        return tripped

    def check_strategies(self, connection: Connection, now: datetime) -> list[Breaker]:
        tripped = []
        rows = connection.execute(
            sa.select(strategies.c.name, strategies.c.stage_since).where(
                strategies.c.stage.in_(("canary", "live"))
            )
        ).all()
        found = breakers.read(connection)
        for row in rows:
            name = breakers.strategy_breaker(row.name)
            series = self.strategy_returns(row.name, row.stage_since)
            if series is None:
                continue
            problem = strategy_problem(series[0], series[1], self.limits)
            current = found.get(name)
            if current is not None and current.state == "half_open":
                breakers.probe(connection, name, problem is None, problem or "Recovered", now)
                continue
            if problem is not None and not breakers.blocks(current):
                tripped.append(breakers.trip(connection, name, "strategy", problem, now))
                self.notify(f"{row.name} paused", problem)
        return tripped

    def check_portfolio(self, connection: Connection, now: datetime) -> Breaker | None:
        current = breakers.read(connection).get(breakers.PORTFOLIO)
        if breakers.blocks(current):
            return None
        problem = portfolio_problem(self.equity_history(), self.holdings(), self.limits)
        if problem is None:
            return None
        self.notify("Portfolio breaker open", f"{problem}. Every position is halved")
        return breakers.trip(connection, breakers.PORTFOLIO, "portfolio", problem, now)

    def retrain_drifted(self, connection: Connection, now: datetime) -> list[str]:
        """High statistical drift of a model asks for it to be retrained."""
        if self.retrain is None:
            return []
        found = drift.latest(connection, now - timedelta(days=1))
        asked = [name for name, (severity, _) in found.items() if severity == "high"]
        for name in asked:
            self.retrain(name)
        return asked

    def notify(self, title: str, detail: str) -> None:
        if self.alert is not None:
            try:
                self.alert(title, detail)
            except Exception:
                logger.exception("Could not raise the breaker alert")

    def check(self, now: datetime | None = None) -> None:
        moment = now or datetime.now(UTC)
        slow = self.slow_at is None or moment - self.slow_at >= SLOW_EVERY
        with transaction() as connection:
            breakers.due(connection, moment)
            self.check_system(connection, moment)
        if slow:
            for level in (self.check_strategies, self.check_portfolio, self.retrain_drifted):
                try:
                    with transaction() as connection:
                        level(connection, moment)
                except Exception:
                    logger.exception("The %s check failed; the others still ran", level.__name__)
        with transaction() as connection:
            self.measure(connection, slow)
        if slow:
            self.slow_at = moment

    def measure(self, connection: Connection, slow: bool) -> None:
        """Expose the breakers, switches, stages and, hourly, rolling Sharpe and drift."""
        metrics.breakers(breakers.read(connection).values())
        metrics.switches(breakers.switches(connection).values())
        rows = connection.execute(sa.select(strategies.c.name, strategies.c.stage)).all()
        metrics.stages({row.name: row.stage for row in rows})
        if not slow:
            return
        for row in rows:
            series = self.strategy_returns(row.name, datetime(1970, 1, 1, tzinfo=UTC))
            if series is None:
                continue
            returns, periods = series
            metrics.sharpes(
                row.name,
                {
                    "30d": sharpe(returns.iloc[-30:], periods),
                    "90d": sharpe(returns.iloc[-90:], periods),
                },
            )
        if self.drift_scores is not None:
            metrics.drift(self.drift_scores())

    async def _check(self) -> None:
        try:
            await asyncio.to_thread(self.check)
        except Exception:
            logger.exception("The breaker check failed")

    def start(self) -> None:
        self.scheduler.add_job(
            self._check,
            IntervalTrigger(seconds=EVERY.total_seconds()),
            id=JOB_ID,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)


def stale_quotes(freshness: Any) -> list[str]:
    """Quote feeds that are stale while their market is open."""
    return [
        feed.name
        for feed in freshness.feeds
        if feed.level == "stale" and feed.kind in ("quotes", "crypto", "book")
    ]


def holdings_frame(
    closes: dict[str, Sequence[float]], sector_of: Callable[[str], str]
) -> pd.DataFrame:
    """Closes of the held names the correlation test applies to: not funds, not coins."""
    kept = {
        name: list(values) for name, values in closes.items() if sector_of(name) not in UNCAPPED
    }
    if not kept:
        return pd.DataFrame()
    length = min(len(values) for values in kept.values())
    return pd.DataFrame({name: values[-length:] for name, values in kept.items()})
