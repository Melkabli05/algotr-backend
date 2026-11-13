import json
import subprocess
import sys
import tempfile
from pathlib import Path

import empyrical
import pandas as pd
from pydantic import BaseModel

from app.features.systematic import backtest
from app.features.systematic.schemas import StrategySpec
from app.quant.history import Prices

TIMEOUT_SECONDS = 300


class EventCheckError(RuntimeError):
    """Raised when the event-driven run cannot be made or ends without a result."""


class Figures(BaseModel):
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float


class EventCheck(BaseModel):
    """The same rule through two engines over the same days: the gap is what the fast one misses.

    The event-driven run trades whole shares out of one account, pays its fee on what it fills,
    leaves changes inside the 5% band alone and fills each order at the next bar's open.
    """

    strategy: str
    start: str
    end: str
    days: int
    event: Figures
    vector: Figures
    cagr_gap: float
    orders: int
    fills: int
    commissions: float


def figures(returns: pd.Series, periods: int) -> Figures:
    return Figures(
        total_return=float(empyrical.cum_returns_final(returns)),
        cagr=float(empyrical.annual_return(returns, annualization=periods)),
        sharpe=float(empyrical.sharpe_ratio(returns, annualization=periods)),
        max_drawdown=float(empyrical.max_drawdown(returns)),
    )


def command(prices: Path, result: Path, spec: StrategySpec) -> list[str]:
    return [
        sys.executable,
        "-m",
        "trading_node.validate",
        "--prices",
        str(prices),
        "--window",
        str(spec.window),
        "--fee-bps",
        str(spec.fee_bps),
        "--result",
        str(result),
    ]


def check(spec: StrategySpec, prices: Prices) -> EventCheck:
    """Run a trend rule through NautilusTrader's backtest and set it against the vectorised one."""
    if spec.signal != "sma":
        raise EventCheckError("The event-driven check runs the trend rules only")
    rows = [
        {"date": day.strftime("%Y-%m-%d"), "symbol": symbol, "open": opened, "close": closed}
        for symbol in prices.closes
        for day, opened, closed in zip(
            prices.closes.index,
            prices.opens[symbol],
            prices.closes[symbol],
            strict=True,
        )
    ]
    with tempfile.TemporaryDirectory() as folder:
        path, result = Path(folder, "prices.csv"), Path(folder, "result.json")
        pd.DataFrame(rows).to_csv(path, index=False)
        completed = subprocess.run(  # noqa: S603
            command(path, result, spec),
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        if not result.exists():
            raise EventCheckError(
                f"The Nautilus run exited with code {completed.returncode}: "
                f"{completed.stderr.decode(errors='replace')[-300:]}"
            )
        found = json.loads(result.read_text())
    equity = pd.Series(
        {pd.Timestamp(day): value for day, value in found["equity"]}, dtype=float
    ).sort_index()
    vector = backtest.daily_returns(prices.closes, spec, opens=prices.opens)
    event = equity.pct_change().loc[vector.index[0] :].dropna()
    common = event.index.intersection(vector.index)
    if len(common) < backtest.MIN_DAYS:
        raise EventCheckError("Too few common days to compare the two engines")
    event, vector = event.loc[common], vector.loc[common]
    periods = spec.periods_per_year
    event_figures, vector_figures = figures(event, periods), figures(vector, periods)
    return EventCheck(
        strategy=spec.name,
        start=common[0].strftime("%Y-%m-%d"),
        end=common[-1].strftime("%Y-%m-%d"),
        days=len(common),
        event=event_figures,
        vector=vector_figures,
        cagr_gap=event_figures.cagr - vector_figures.cagr,
        orders=int(found["orders"]),
        fills=int(found["fills"]),
        commissions=float(found["commissions"]),
    )
