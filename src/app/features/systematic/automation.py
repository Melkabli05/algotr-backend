import asyncio
import logging
from collections.abc import Collection
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from threading import Lock

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel

from app.core import audit
from app.core.agents import ENSEMBLE, AgentSignal
from app.core.compliance import KillState
from app.core.drift import SizingGuard, StageGate
from app.core.settings import Settings
from app.core.sleeve import Sleeve, SleeveProvider
from app.features.portfolio import service as portfolio
from app.features.systematic import execution, lab, nav, rebalance, research, targets
from app.features.systematic.catalog import STRATEGIES, is_crypto
from app.features.systematic.nav import Book
from app.features.systematic.schemas import (
    AutomationConfig,
    AutomationStatus,
    ExecutedOrder,
    Overview,
    Plan,
    ResearchResult,
    RunRecord,
    StrategyName,
    StrategyOverview,
    StrategySpec,
)
from app.quant import history, proxy
from app.quant.history import NotEnoughHistoryError

logger = logging.getLogger(__name__)

STATE_PATH = Path("data/automation.json")
MAX_RUNS = 30
JOB_ID = "daily-rebalance"
CATCH_UP_JOB_ID = "catch-up-rebalance"
RUN_AT = CronTrigger(hour=0, minute=5, timezone=UTC)
RUN_AT_TIME = time(0, 5)
CATCH_UP_DELAY = timedelta(seconds=5)
CATCH_UP_EVERY = timedelta(minutes=10)
MAX_FAILURES = 6
SLEEVE_BOOK = "sleeve"
ENTER_DAYS = 10
PROXY_JOB_ID = "market-proxy"
PROXY_EVERY_HOURS = 24
EXIT_DAYS = 10


COLLECTED_RUNS = 5


class AutomationDisabledError(RuntimeError):
    """Raised when a run is requested while automation is off."""


class LiveAccountError(ValueError):
    """Raised when automation is asked to trade a live account the settings have not opened."""


class State(BaseModel):
    config: AutomationConfig = AutomationConfig()
    runs: list[RunRecord] = []
    halted: str | None = None
    book: Book = Book()
    books: dict[str, Book] = {}
    active: list[str] | None = None
    streaks: dict[str, int] = {}
    judged: date | None = None
    managed: list[str] = []
    sleeve_managed: list[str] = []


class Automation:
    """Rebalances the selected strategies to their daily targets on a schedule, paper only."""

    def __init__(
        self,
        settings: Settings,
        trading: TradingClient,
        stocks: StockHistoricalDataClient,
        crypto: CryptoHistoricalDataClient,
        state_path: Path = STATE_PATH,
        warm_cache: bool = False,
        audit_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.trading = trading
        self.stocks = stocks
        self.crypto = crypto
        self.state_path = state_path
        self.warm_cache = warm_cache
        self.audit = audit.AuditLog(audit_path or audit.AUDIT_PATH)
        self.scheduler = AsyncIOScheduler(timezone=UTC)
        self.lock = Lock()
        self.state = self._load()
        self.sleeve: SleeveProvider | None = None
        self.sizing: SizingGuard | None = None
        self.agents: AgentSignal | None = None
        self.lifecycle: StageGate | None = None
        self.link: targets.PlatformLink | None = None

    def _load(self) -> State:
        try:
            return State.model_validate_json(self.state_path.read_text())
        except (FileNotFoundError, ValueError):
            return State()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(self.state.model_dump_json(indent=2))

    def status(self) -> AutomationStatus:
        job = self.scheduler.get_job(JOB_ID)
        enabled = self.state.config.mode != "off"
        return AutomationStatus(
            config=self.state.config,
            next_run=job.next_run_time if job and enabled else None,
            halted=self.state.halted,
            book=self.state.book,
            books=self.state.books,
            kill_state=self.kill_state(),
            runs=list(reversed(self.state.runs)),
        )

    def configure(self, config: AutomationConfig) -> AutomationStatus:
        """Save the settings; switching automation on deploys immediately instead of at 00:05."""
        live = not self.settings.alpaca_paper
        if config.mode == "paper" and live and self.settings.alpaca_trading_mode != "live":
            raise LiveAccountError(
                "This is a live account; set ALPACA_TRADING_MODE=live to let the autopilot trade it"
            )
        with self.lock:
            switched_on = self.state.config.mode == "off" and config.mode != "off"
            changed = config != self.state.config
            before = self.state.config
            if switched_on:
                self.state.book, self.state.books = Book(), {}
            self.state.config = config
            self._save()
        self.publish()
        if changed:
            self.audit.record(
                "settings-change",
                {"area": "autopilot", "before": before.model_dump(), "after": config.model_dump()},
            )
        if config.mode != "off" and changed and self.scheduler.running:
            self._schedule_run()
        return self.status()

    def publish(self) -> None:
        """Tell the trading node the autopilot's mode and allocation; a fault never blocks it."""
        if self.link is None:
            return
        try:
            self.link.publish(self.state.config, [*self.state.managed, *self.state.sleeve_managed])
        except Exception:
            logger.exception("Could not publish the autopilot's settings to the trading node")

    def halt(self, reason: str) -> RunRecord:
        """Sell everything the autopilot holds and stop it until Resume: the manual kill switch."""
        config = self.state.config
        base = {"time": datetime.now(UTC), "strategies": [], "mode": config.mode}
        detail: str | None = None
        plan: Plan | None = None
        executed: list[ExecutedOrder] = []
        with self.lock:
            try:
                plan = self.plan([], config.allocation, flat=True)
                executed = self._place(config, plan) if config.mode == "paper" else []
                self._remember(plan)
                self._retarget(plan)
            except Exception as exc:
                logger.exception("The manual kill switch could not close the positions")
                detail = f"{type(exc).__name__}: {exc}"
            self.state.halted = f"stopped by hand: {reason}"
            record = RunRecord(
                **base,
                status="halted",
                detail=detail or self.state.halted,
                plan=plan,
                executed=executed,
            )
            self.state.runs = [*self.state.runs, record][-MAX_RUNS:]
            self._save()
        self._audit(record, config)
        return record

    def resume(self) -> AutomationStatus:
        with self.lock:
            self.state.halted = None
            self.state.book = nav.restarted(self.state.book)
            self.state.books = {name: nav.restarted(b) for name, b in self.state.books.items()}
            self._save()
        if self.state.config.mode != "off" and self.scheduler.running:
            self._schedule_run()
        return self.status()

    def closes(self, names: list[StrategyName]) -> dict[StrategyName, pd.DataFrame]:
        return {
            name: history.cached_closes(self.stocks, self.crypto, STRATEGIES[name].universe)
            for name in names
        }

    def plan(
        self,
        names: list[StrategyName],
        allocation: float,
        flat: bool = False,
        cuts: dict[str, float] | None = None,
    ) -> Plan:
        """Today's plan; each strategy's size is the drift monitor's scale times `cuts`."""
        agent = self.agent_part() if self.ensemble_admitted(self.state.config) else None
        scales: dict[str, float] = {
            name: (self.sizing.scale(name) if self.sizing else 1.0)
            * (cuts or {}).get(name, 1.0)
            * self.stage_capital(name)
            for name in [*names, *([ENSEMBLE] if agent else [])]
        }
        return rebalance.build_plan(
            self.trading,
            self.closes(names),
            allocation,
            self.settings.alpaca_paper,
            managed=self.state.managed,
            flat=flat,
            scanner=self._sleeve_request(),
            scales=scales,
            agent=agent,
        )

    def stage_capital(self, name: str) -> float:
        """In automatic mode a strategy trades the share of its capital its stage allows."""
        if self.lifecycle is None or not self.state.config.automatic or name == ENSEMBLE:
            return 1.0
        return self.lifecycle.capital(name)

    def _sleeve_request(self) -> rebalance.SleeveRequest | None:
        """The sleeve's settings and offer, or nothing while it is off and holds nothing."""
        config = self.state.config.sleeve
        if config.mode == "off" and not self.state.sleeve_managed:
            return None
        offered = None if config.mode == "off" else self._offered(config.variant)
        return rebalance.SleeveRequest(config, offered, self.state.sleeve_managed)

    def _offered(self, variant: str) -> Sleeve | None:
        """What the scanner offers; a fault there never stops the strategies."""
        if self.sleeve is None:
            return None
        try:
            return self.sleeve.sleeve(variant)
        except Exception:
            logger.exception("The scanner could not offer holdings for the sleeve")
            return None

    def search(self) -> lab.Search:
        """How wide the search for a tradable rule has been; a fault reading it counts the rules."""
        try:
            return lab.search(self.audit)
        except Exception:
            logger.exception("Could not read the lab's trials from the audit log")
            return lab.Search(trials=STRATEGIES.trials(), spreads={})

    def research_of(self, spec: StrategySpec, search: lab.Search) -> ResearchResult:
        return research.research(
            self.stocks,
            self.crypto,
            spec,
            trials=search.trials,
            spread=search.spreads.get(spec.name),
        )

    def passing(self, search: lab.Search) -> list[StrategyName]:
        """Every strategy that passes all its checks today, crypto included."""
        names: list[StrategyName] = []
        for name, spec in STRATEGIES.items():
            try:
                result = self.research_of(spec, search)
            except (NotEnoughHistoryError, APIError):
                continue
            if all(check.passed for check in result.checks):
                names.append(name)
        return names

    def selected(
        self, config: AutomationConfig, search: lab.Search | None = None
    ) -> list[StrategyName]:
        """The listed strategies, or when automatic those `judge` admitted (crypto if allowed).

        Before the first judgement the admitted ones are simply those passing today.
        """
        if not config.automatic:
            return [name for name in config.strategies if name not in config.paused]
        if self.lifecycle is not None:
            staged = set(self.lifecycle.trading())
            return [
                name
                for name, spec in STRATEGIES.items()
                if name in staged
                and name not in config.paused
                and (config.crypto or not is_crypto(spec))
            ]
        admitted: list[str] = (
            self.state.active
            if self.state.active is not None
            else [str(name) for name in self.passing(search or self.search())]
        )
        return [
            name
            for name, spec in STRATEGIES.items()
            if name in admitted
            and name not in config.paused
            and (config.crypto or not is_crypto(spec))
        ]

    def judge(self, search: lab.Search, today: date) -> None:
        """Once a day, count how long each strategy has passed or failed, and admit or drop it.

        A strategy joins after passing on `ENTER_DAYS` days in a row and leaves after failing on
        `EXIT_DAYS`, so one near its thresholds does not trade in and out. The first judgement
        starts from what the last run traded, so a stricter check never sells all at once.
        """
        passes: set[str] = set(self.passing(search))
        if self.agents is not None and self.agents.ensemble_eligible():
            passes.add(ENSEMBLE)
        if self.state.active is None:
            ran = self.state.runs[-1].strategies if self.state.runs else None
            first = set(ran) if ran is not None else passes
            self.state.active = [name for name in self.candidates() if name in first]
            self.state.judged = today
            return
        if self.state.judged == today:
            return
        active = set(self.state.active)
        for name in self.candidates():
            streak = self.state.streaks.get(name, 0)
            if name in passes:
                streak = streak + 1 if streak > 0 else 1
            else:
                streak = streak - 1 if streak < 0 else -1
            self.state.streaks[name] = streak
            if name in active and streak <= -EXIT_DAYS:
                active.discard(name)
            elif name not in active and streak >= ENTER_DAYS:
                active.add(name)
        self.state.active = [name for name in self.candidates() if name in active]
        self.state.judged = today

    def candidates(self) -> list[str]:
        """Everything automatic mode chooses from: the rules, and the agents when wired."""
        return [*STRATEGIES, *([ENSEMBLE] if self.agents is not None else [])]

    def ensemble_admitted(self, config: AutomationConfig) -> bool:
        """Whether the agents' ensemble trades: automatic mode, and judged in like any rule."""
        return (
            self.agents is not None
            and config.automatic
            and self.state.active is not None
            and ENSEMBLE in self.state.active
        )

    def agent_part(self) -> rebalance.AgentPart | None:
        """The ensemble as a strategy for the plan, or None while it has nothing to offer."""
        if self.agents is None:
            return None
        target, returns = self.agents.ensemble_target(), self.agents.ensemble_returns()
        if not target or returns is None or returns.empty:
            return None
        closes = history.cached_closes(self.stocks, self.crypto, sorted(target))
        return rebalance.AgentPart(target, closes, returns)

    def overview(self) -> Overview:
        """Everything the dashboard shows, from one call: research, today's plan and the status."""
        config = self.state.config
        search = self.search()
        chosen = self.selected(config, search)
        strategies = []
        for name, spec in STRATEGIES.items():
            streak = self.state.streaks.get(name, 0)
            try:
                result = self.research_of(spec, search)
                strategies.append(
                    StrategyOverview(
                        spec=spec, active=name in chosen, research=result, streak=streak
                    )
                )
            except (NotEnoughHistoryError, APIError) as exc:
                strategies.append(
                    StrategyOverview(
                        spec=spec, active=name in chosen, error=str(exc), streak=streak
                    )
                )
        plan, plan_error, equity, drawdown = None, None, None, None
        try:
            equity = rebalance.account_for(self.trading, self.settings.alpaca_paper).equity
            books, book = self._marked()
            drawdown = book.drawdown
            plan = self.plan(chosen, config.allocation, cuts=self._cuts(chosen, books, book))
        except (NotEnoughHistoryError, rebalance.PlanError, APIError) as exc:
            plan_error = str(exc)
        return Overview(
            status=self.status(),
            strategies=strategies,
            plan=plan,
            plan_error=plan_error,
            equity=equity,
            drawdown=drawdown,
            execution=execution.summaries(self.state.runs),
            ensemble_active=self.ensemble_admitted(config),
            ensemble_streak=self.state.streaks.get(ENSEMBLE, 0),
        )

    def run(self) -> RunRecord:
        """Plan and, in paper mode, execute one rebalance; the outcome is always recorded."""
        config = self.state.config
        if config.mode == "off":
            raise AutomationDisabledError("Automation is off")
        with self.lock:
            self._settle()
            record = self._attempt(config)
            self.state.runs = [*self.state.runs, record][-MAX_RUNS:]
            self._save()
        self._audit(record, config)
        return record

    def _audit(self, record: RunRecord, config: AutomationConfig) -> None:
        """Pin what this run saw and did in the audit log; a logging fault never stops trading."""
        try:
            self.audit.record(
                "run",
                {
                    "run": record.model_dump(mode="json"),
                    "manifest": audit.manifest(
                        config.model_dump(),
                        {
                            str(name): frame
                            for name, frame in self.closes(record.strategies).items()
                        },
                    ),
                    "compliance": self._compliance(),
                },
            )
        except Exception:
            logger.exception("Could not write the audit entry")

    def _compliance(self) -> dict[str, object]:
        """The account's standing against the rules: long only, no borrowed cash."""
        positions = portfolio.list_positions(self.trading)
        account = portfolio.get_account(self.trading, self.settings.alpaca_paper)
        shorts = [position.symbol for position in positions if position.qty < 0]
        return {
            "short_positions": shorts,
            "negative_cash": account.cash < 0,
            "compliant": not shorts and account.cash >= 0,
        }

    def _attempt(self, config: AutomationConfig) -> RunRecord:
        now = datetime.now(UTC)
        search = self.search()
        if config.automatic:
            self.judge(search, now.date())
        chosen = self.selected(config, search)
        base = {"time": now, "strategies": chosen, "mode": config.mode}
        if self.state.halted:
            return RunRecord(**base, status="skipped", detail=f"Halted: {self.state.halted}")
        try:
            rebalance.account_for(self.trading, self.settings.alpaca_paper)
            books, book = self._marked()
            self.state.books, self.state.book = books, book
            drawdown = book.drawdown
            breached = drawdown >= config.max_drawdown
            state = nav.kill_state(drawdown, config.max_drawdown, config.reduce_at, False)
            cuts = self._cuts(chosen, books, book)
            plan = self.plan(chosen, config.allocation, flat=breached, cuts=cuts)
            if state == "defensive":
                plan = plan.model_copy(
                    update={"orders": [o for o in plan.orders if o.side == "sell"]}
                )
            executed = self._place(config, plan) if config.mode == "paper" else []
        except (NotEnoughHistoryError, rebalance.PlanError) as exc:
            return RunRecord(**base, status="skipped", detail=str(exc))
        except Exception as exc:
            logger.exception("Scheduled rebalance failed")
            return RunRecord(**base, status="error", detail=f"{type(exc).__name__}: {exc}")
        if config.mode == "paper":
            self._remember(plan)
        self._retarget(plan, defensive=state == "defensive")
        if breached:
            self.state.halted = (
                f"the autopilot's holdings fell {drawdown:.1%} from their peak, reaching the "
                f"{config.max_drawdown:.0%} limit; positions were closed"
            )
            return RunRecord(
                **base, status="halted", detail=self.state.halted, plan=plan, executed=executed
            )
        return RunRecord(**base, status="ok", plan=plan, executed=executed)

    def _marks(self, symbols: Collection[str]) -> dict[str, float]:
        """Prices to value the books at: the account's latest, else the last daily close."""
        held = {p.symbol: p.current_price for p in portfolio.list_positions(self.trading)}
        marks: dict[str, float] = {}
        for symbol in symbols:
            price = held.get(rebalance.plain(symbol))
            if price is None:
                try:
                    closes = history.cached_closes(self.stocks, self.crypto, [symbol])
                    price = float(closes[symbol].iloc[-1])
                except (NotEnoughHistoryError, APIError, KeyError, IndexError):
                    continue
            marks[symbol] = price
        return marks

    def _marked(self) -> tuple[dict[str, Book], Book]:
        """Every book and the total moved to today's prices, without saving them."""
        symbols = {symbol for book in self.state.books.values() for symbol in book.holdings}
        marks = self._marks(symbols) if symbols else {}
        books = {name: nav.marked(book, marks) for name, book in self.state.books.items()}
        return books, nav.total(self.state.book, self.state.books, marks)

    def _cuts(
        self, names: list[StrategyName], books: dict[str, Book], book: Book
    ) -> dict[str, float]:
        """Half size for a strategy whose own book, or the whole autopilot, is deep in drawdown."""
        config = self.state.config
        return {
            name: nav.scale(
                max(books.get(name, Book()).drawdown, book.drawdown),
                config.max_drawdown,
                config.reduce_at,
            )
            for name in [*names, ENSEMBLE]
        }

    def _retarget(self, plan: Plan, defensive: bool = False) -> None:
        """Each book holds what the plan means it to hold from now; a dropped one holds nothing.

        In defensive no buy goes out, so no book grows.
        """
        parts = dict(plan.books)
        if plan.sleeve is not None and plan.sleeve.book is not None:
            parts[SLEEVE_BOOK] = plan.sleeve.book
        for name, book in list(self.state.books.items()):
            if name not in parts and not (name == SLEEVE_BOOK and plan.sleeve is not None):
                self.state.books[name] = nav.retargeted(book, 0.0, {}, {})
        for name, part in parts.items():
            current = self.state.books.get(name, Book())
            targets = nav.capped(part.targets, current) if defensive else part.targets
            self.state.books[name] = nav.retargeted(current, part.capital, targets, part.prices)

    def kill_state(self) -> KillState:
        """The kill switch's step, from the books as last marked (every run and every 10 min)."""
        config = self.state.config
        if config.mode == "off":
            return "normal"
        return nav.kill_state(
            self.state.book.drawdown,
            config.max_drawdown,
            config.reduce_at,
            self.state.halted is not None,
        )

    def breached_now(self) -> bool:
        """Mark the books at the latest prices and keep them; True once past the limit."""
        config = self.state.config
        if config.mode == "off" or self.state.halted or not self.state.books:
            return False
        with self.lock:
            self.state.books, self.state.book = self._marked()
            self._save()
        return self.state.book.drawdown >= config.max_drawdown

    def _place(self, config: AutomationConfig, plan: Plan) -> list[ExecutedOrder]:
        """Strategies never place orders: the API's sources write targets for the node's arbiter,
        which nets them with the node's own strategies into the only orders sent."""
        if self.link is None:
            return []
        self.link.write(targets.api_targets(plan, ENSEMBLE))
        return execution.with_prices(self.link.executed(datetime.now(UTC)), plan.orders)

    def _collect(self) -> bool:
        """Attach to each recent run the orders the arbiter sent for it, once they are done."""
        if self.link is None:
            return False
        changed = False
        for run in self.state.runs[-COLLECTED_RUNS:]:
            if run.mode != "paper" or run.executed or run.plan is None or run.status != "ok":
                continue
            found = self.link.executed(run.time)
            if found:
                run.executed = execution.with_prices(found, run.plan.orders)
                changed = True
        return changed

    def _settle(self) -> bool:
        """Record how the recent orders filled; a fault here never stops a run."""
        try:
            collected = self._collect()
            return collected | execution.settle(
                self.trading,
                self.state.runs,
                execution.bar_open(self.stocks, self.crypto),
                datetime.now(UTC),
            )
        except Exception:
            logger.exception("Could not settle the recent orders")
            return False

    def settle_fills(self) -> None:
        """Settle finished orders now and keep the result."""
        with self.lock:
            if self._settle():
                self._save()

    def _remember(self, plan: Plan) -> None:
        """Symbols the automation has traded stay managed, so dropping a strategy sells them."""

        def wanted(source: str) -> set[str]:
            return {
                target.symbol
                for target in plan.targets
                if target.source == source and target.target_value > 0
            }

        self.state.managed = sorted({*self.state.managed, *wanted("strategy")})
        self.state.sleeve_managed = sorted({*self.state.sleeve_managed, *wanted("sleeve")})
        self.publish()

    def _schedule_run(self) -> None:
        self.scheduler.add_job(self._scheduled, "date", run_date=datetime.now(UTC) + CATCH_UP_DELAY)

    async def _scheduled(self) -> None:
        if self.state.config.mode != "off":
            await asyncio.to_thread(self.run)

    async def _caught_up(self) -> None:
        if self.lock.locked():
            return
        if self._missed_today():
            await self._scheduled()
            return
        try:
            breached = await asyncio.to_thread(self.breached_now)
        except Exception:
            logger.exception("Could not check the drawdown between runs")
            breached = False
        if breached:
            await self._scheduled()
        elif self._unsettled():
            await asyncio.to_thread(self.settle_fills)

    def _unsettled(self) -> bool:
        return any(
            not item.settled and item.order_id for run in self.state.runs for item in run.executed
        )

    def _refresh_proxy(self) -> None:
        """Keep the market's long history current; new data means research is run again."""
        if proxy.refresh():
            research.RESULTS.clear()

    def _warm(self) -> None:
        search = self.search()
        for spec in STRATEGIES.values():
            try:
                self.research_of(spec, search)
            except (NotEnoughHistoryError, APIError):
                logger.warning("Could not warm research for %s", spec.name)

    def start(self) -> None:
        """Schedule the daily run, keep catching up until today's has succeeded, warm the caches."""
        self.scheduler.add_job(
            self._scheduled,
            RUN_AT,
            id=JOB_ID,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=3600,
        )
        self.scheduler.add_job(
            self._caught_up,
            IntervalTrigger(seconds=CATCH_UP_EVERY.total_seconds()),
            id=CATCH_UP_JOB_ID,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.add_job(
            self._refresh_proxy,
            IntervalTrigger(hours=PROXY_EVERY_HOURS),
            id=PROXY_JOB_ID,
            next_run_time=datetime.now(UTC),
            misfire_grace_time=None,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()
        if self.state.config.mode != "off" and self._missed_today():
            self._schedule_run()
        if self.warm_cache:
            self.scheduler.add_job(self._warm, "date")

    def _missed_today(self) -> bool:
        now = datetime.now(UTC)
        today = [run for run in self.state.runs if run.time.date() == now.date()]
        failed = sum(run.status == "error" for run in today)
        settled = len(today) > failed or failed >= MAX_FAILURES
        return not settled and now.time() >= RUN_AT_TIME

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
