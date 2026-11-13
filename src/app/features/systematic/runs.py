import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

import empyrical
import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from pydantic import BaseModel, Field, model_validator

from app.core import audit
from app.features.systematic import backtest, events, lab, research, signals, stats
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.schemas import MAX_YEARS, GrowthPoint, StrategyName, StrategySpec
from app.quant import history, proxy
from app.quant.performance import Check

logger = logging.getLogger(__name__)

PATH = Path("data/backtests.json")
KEPT = 50
FOLDS = 5
MAX_VARIANTS = 120
KIND = "sweep"


class Sweep(BaseModel):
    start: int = Field(ge=1)
    stop: int = Field(ge=1)
    step: int = Field(default=1, ge=1)

    def values(self) -> list[int]:
        return list(range(self.start, self.stop + 1, self.step))


class BacktestRequest(BaseModel):
    """One rule family over a grid of its parameters; the variants count as trials."""

    family: StrategyName
    years: int = Field(default=MAX_YEARS, ge=2, le=MAX_YEARS)
    window: Sweep | None = None
    top_n: Sweep | None = None
    event: bool = False

    @model_validator(mode="after")
    def sweepable(self) -> Self:
        spec = STRATEGIES.get(self.family)
        if spec is None:
            raise ValueError(f"No strategy named {self.family}")
        if spec.signal not in ("sma", "momentum"):
            raise ValueError("Only the trend and momentum rules have parameters to sweep")
        if self.top_n is not None and spec.signal != "momentum":
            raise ValueError("Only the momentum rule holds a number of funds")
        if self.top_n is not None and max(self.top_n.values(), default=0) >= len(spec.universe):
            raise ValueError(f"Hold fewer than the {len(spec.universe)} funds of the universe")
        if len(variants(self)) > MAX_VARIANTS:
            raise ValueError(f"At most {MAX_VARIANTS} variants a run")
        if self.event and spec.signal != "sma":
            raise ValueError("The event-driven check runs the trend rules only")
        return self


class Variant(BaseModel):
    label: str
    window: int
    top_n: int
    sharpe: float
    cagr: float
    max_drawdown: float
    total_return: float
    deflated_sharpe: float
    checks_passed: int


class Fold(BaseModel):
    start: str
    end: str
    sharpe: float | None
    cagr: float
    max_drawdown: float
    win_rate: float | None
    passed: bool


class Trade(BaseModel):
    entry: str
    exit: str
    days: int
    result: float


class Bias(BaseModel):
    name: str
    applied: bool | None
    detail: str


class Heatmap(BaseModel):
    windows: list[int]
    tops: list[int]
    sharpe: list[list[float | None]]


class BacktestRun(BaseModel):
    id: str
    created: datetime
    request: BacktestRequest
    status: Literal["queued", "running", "complete", "failed"] = "queued"
    detail: str | None = None
    start: str | None = None
    end: str | None = None
    trials: int | None = None
    variants: list[Variant] = []
    best: str | None = None
    growth: list[GrowthPoint] = []
    checks: list[Check] = []
    folds: list[Fold] = []
    trades: list[Trade] = []
    heatmap: Heatmap | None = None
    bias: list[Bias] = []
    event: events.EventCheck | None = None


class Runs(BaseModel):
    runs: list[BacktestRun] = []


def label_of(spec: StrategySpec) -> str:
    if spec.signal == "momentum":
        return f"{spec.window}-day, top {spec.top_n}"
    return f"{spec.window}-day average"


def variants(request: BacktestRequest) -> list[StrategySpec]:
    base = STRATEGIES[request.family]
    windows = request.window.values() if request.window else [base.window]
    tops = request.top_n.values() if request.top_n else [base.top_n]
    return [base.model_copy(update={"window": w, "top_n": t}) for w in windows for t in tops]


def folds(returns: pd.Series, periods: int) -> list[Fold]:
    found = []
    for part in np.array_split(np.arange(len(returns)), FOLDS):
        if len(part) < 2:
            continue
        piece = returns.iloc[part]
        sharpe = float(empyrical.sharpe_ratio(piece, annualization=periods))
        cagr = float(empyrical.annual_return(piece, annualization=periods))
        active = piece[piece != 0]
        found.append(
            Fold(
                start=piece.index[0].strftime("%Y-%m-%d"),
                end=piece.index[-1].strftime("%Y-%m-%d"),
                sharpe=sharpe if np.isfinite(sharpe) else None,
                cagr=cagr,
                max_drawdown=float(empyrical.max_drawdown(piece)),
                win_rate=float((active > 0).mean()) if len(active) else None,
                passed=cagr > 0,
            )
        )
    return found


def trades(returns: pd.Series, invested: pd.Series) -> list[Trade]:
    found = []
    for first, last in stats.holdings(invested.reindex(returns.index).fillna(False).astype(bool)):
        found.append(
            Trade(
                entry=returns.index[first].strftime("%Y-%m-%d"),
                exit=returns.index[last].strftime("%Y-%m-%d"),
                days=last - first + 1,
                result=float((1 + returns.iloc[first : last + 1]).prod() - 1),
            )
        )
    return found


def bias(spec: StrategySpec, trials: int, event: bool) -> list[Bias]:
    fills = "the next open" if spec.fills == "next_open" else "the close it decided on"
    return [
        Bias(
            name="Look-ahead",
            applied=True,
            detail=f"each day's weights use closes up to that day; trades fill at {fills}",
        ),
        Bias(
            name="Adjusted prices", applied=True, detail="split- and dividend-adjusted daily bars"
        ),
        Bias(
            name="Transaction costs",
            applied=True,
            detail=f"{spec.fee_bps:g} bps per unit of turnover; also checked at 10, 25 and 50 bps",
        ),
        Bias(
            name="Multiple testing",
            applied=True,
            detail=f"the deflated Sharpe counts {trials} rules and variants ever tried, "
            "this run's included",
        ),
        Bias(
            name="Survivorship",
            applied=False,
            detail="the universe is funds that trade today; one that closed would be missing",
        ),
        Bias(
            name="Purge and embargo",
            applied=None,
            detail="not needed: nothing is fitted, so there is no training window to separate",
        ),
        Bias(
            name="Event-driven validation",
            applied=event,
            detail="the best variant through Nautilus, whole shares and real fees"
            if event
            else "not asked for",
        ),
    ]


def heatmap(found: list[Variant]) -> Heatmap:
    windows = sorted({item.window for item in found})
    tops = sorted({item.top_n for item in found})
    lookup = {(item.window, item.top_n): item.sharpe for item in found}
    return Heatmap(
        windows=windows, tops=tops, sharpe=[[lookup.get((w, t)) for w in windows] for t in tops]
    )


def compute(
    run: BacktestRun,
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    log: audit.AuditLog,
) -> BacktestRun:
    """Every variant of the request through the gates, and the best one in detail."""
    specs = variants(run.request)
    family = run.request.family
    log.record(
        KIND,
        {
            "result": {
                "families": [
                    {"family": family, "candidates": [{"label": label_of(s)} for s in specs]}
                ]
            }
        },
    )
    search = lab.search(log)
    longest = max(specs, key=signals.warmup_days)
    prices = history.cached_prices(stocks, crypto, longest.universe)
    closes = research.recent(prices.closes, longest, run.request.years)
    opens = prices.opens.loc[closes.index]
    market = proxy.market() if longest.proxy else None
    results = {
        label_of(spec): (
            spec,
            backtest.run(
                closes, spec, None, opens, search.trials, search.spreads.get(family), market
            ),
        )
        for spec in specs
    }
    rows = [
        Variant(
            label=label,
            window=spec.window,
            top_n=spec.top_n,
            sharpe=result.metrics.sharpe,
            cagr=result.metrics.cagr,
            max_drawdown=result.metrics.max_drawdown,
            total_return=result.metrics.total_return,
            deflated_sharpe=result.deflated_sharpe,
            checks_passed=sum(check.passed for check in result.checks),
        )
        for label, (spec, result) in results.items()
    ]
    best = max(rows, key=lambda row: row.sharpe if np.isfinite(row.sharpe) else -np.inf)
    spec, result = results[best.label]
    daily = backtest.daily_returns(closes, spec, opens=opens)
    invested = signals.weights(closes, spec).shift(1).sum(axis=1) > 0
    checked = None
    if run.request.event:
        checked = events.check(spec, history.Prices(closes, opens))
    return run.model_copy(
        update={
            "status": "complete",
            "start": result.start.strftime("%Y-%m-%d"),
            "end": result.end.strftime("%Y-%m-%d"),
            "trials": search.trials,
            "variants": rows,
            "best": best.label,
            "growth": result.growth,
            "checks": result.checks,
            "folds": folds(daily, spec.periods_per_year),
            "trades": trades(daily, invested),
            "heatmap": heatmap(rows),
            "bias": bias(spec, search.trials, checked is not None),
            "event": checked,
        }
    )


class BacktestDesk:
    """Runs backtests one at a time in the background and keeps the last 50."""

    def __init__(
        self,
        stocks: StockHistoricalDataClient,
        crypto: CryptoHistoricalDataClient,
        log: audit.AuditLog,
        path: Path = PATH,
    ) -> None:
        self.stocks = stocks
        self.crypto = crypto
        self.log = log
        self.path = path
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="backtest")
        self.state = self._load()
        for item in self.state.runs:
            if item.status in ("queued", "running"):
                item.status, item.detail = "failed", "The app stopped before the run finished"

    def _load(self) -> Runs:
        try:
            return Runs.model_validate_json(self.path.read_text())
        except (FileNotFoundError, ValueError):
            return Runs()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.state.model_dump_json())

    def _put(self, run: BacktestRun) -> None:
        with self.lock:
            others = [item for item in self.state.runs if item.id != run.id]
            self.state.runs = [run, *others][:KEPT]
            self._save()

    def submit(self, request: BacktestRequest) -> BacktestRun:
        run = BacktestRun(id=uuid.uuid4().hex[:12], created=datetime.now(UTC), request=request)
        self._put(run)
        self.pool.submit(self._work, run)
        return run

    def _work(self, run: BacktestRun) -> None:
        self._put(run.model_copy(update={"status": "running"}))
        try:
            done = compute(run, self.stocks, self.crypto, self.log)
        except Exception as exc:
            logger.exception("Backtest %s failed", run.id)
            done = run.model_copy(
                update={"status": "failed", "detail": f"{type(exc).__name__}: {exc}"}
            )
        self._put(done)

    def runs(self) -> list[BacktestRun]:
        return list(self.state.runs)

    def get(self, run_id: str) -> BacktestRun | None:
        return next((item for item in self.state.runs if item.id == run_id), None)

    def stop(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
