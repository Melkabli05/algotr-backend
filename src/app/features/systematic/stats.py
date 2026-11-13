import empyrical
import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.features.systematic import backtest
from app.features.systematic.schemas import RunRecord, StrategySpec
from app.quant.history import Prices


class TradeStats(BaseModel):
    sortino: float | None
    calmar: float | None
    trades: int
    win_rate: float | None
    profit_factor: float | None
    average_days: float | None


class Signal(BaseModel):
    time: str
    weights: dict[str, float]
    status: str


class StrategyStats(BaseModel):
    name: str
    stats: TradeStats
    capital: float | None
    signals: list[Signal]


def finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def holdings(invested: pd.Series) -> list[tuple[int, int]]:
    """The first and last day of every stretch the rule was invested."""
    stretches, start = [], None
    for index, held in enumerate(invested.to_numpy()):
        if held and start is None:
            start = index
        elif not held and start is not None:
            stretches.append((start, index - 1))
            start = None
    if start is not None:
        stretches.append((start, len(invested) - 1))
    return stretches


def trade_stats(returns: pd.Series, invested: pd.Series, periods: int) -> TradeStats:
    """Risk-adjusted figures of the daily returns, and the record of each invested stretch."""
    stretches = holdings(invested.reindex(returns.index).fillna(False).astype(bool))
    results = [float((1 + returns.iloc[a : b + 1]).prod() - 1) for a, b in stretches]
    gains = sum(r for r in results if r > 0)
    losses = -sum(r for r in results if r < 0)
    return TradeStats(
        sortino=finite(empyrical.sortino_ratio(returns, annualization=periods)),
        calmar=finite(empyrical.calmar_ratio(returns, annualization=periods)),
        trades=len(results),
        win_rate=sum(r > 0 for r in results) / len(results) if results else None,
        profit_factor=gains / losses if losses else None,
        average_days=float(np.mean([b - a + 1 for a, b in stretches])) if stretches else None,
    )


def strategy_stats(spec: StrategySpec, prices: Prices, runs: list[RunRecord]) -> StrategyStats:
    """A strategy's trade record from its backtest, its capital now and its recent signals."""
    from app.features.systematic import signals

    returns = backtest.daily_returns(prices.closes, spec, opens=prices.opens)
    held = signals.weights(prices.closes, spec).shift(1).sum(axis=1) > 0
    signals_seen = [
        Signal(
            time=run.time.isoformat(),
            weights=run.plan.signals.get(spec.name, {}),
            status=run.status,
        )
        for run in reversed(runs)
        if run.plan is not None and spec.name in run.plan.signals
    ]
    last = next((run.plan for run in reversed(runs) if run.plan is not None), None)
    book = last.books.get(spec.name) if last else None
    return StrategyStats(
        name=spec.name,
        stats=trade_stats(returns, held, spec.periods_per_year),
        capital=book.capital if book else None,
        signals=signals_seen,
    )
