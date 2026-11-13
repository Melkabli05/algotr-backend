from collections.abc import Callable
from datetime import date
from threading import Condition

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from cachetools import TTLCache, cached
from pydantic import BaseModel

from app.features.systematic import signals
from app.features.systematic.schemas import StrategyName, StrategySpec
from app.quant import history

BARS: TTLCache[str, pd.DataFrame] = TTLCache(maxsize=16, ttl=history.CACHE_SECONDS)
Fill = Callable[[int], tuple[date, float]]


class TraceBar(BaseModel):
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    target: float
    held: float
    lines: dict[str, float | None]
    values: dict[str, float | None]


class TraceTrade(BaseModel):
    """One stretch of holding the symbol; `gain` is its price from entry fill to exit fill."""

    entry: date
    entry_price: float
    exit: date | None
    exit_price: float | None
    days: int
    gain: float
    weight: float
    open: bool
    why_entry: str
    why_exit: str | None


class Trace(BaseModel):
    strategy: StrategyName
    title: str
    symbol: str
    evaluated_from: date
    fills: str
    fee_bps: float
    lines: list[str]
    values: list[str]
    bars: list[TraceBar]
    trades: list[TraceTrade]


def number(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def indicators(
    closes: pd.DataFrame, symbol: str, spec: StrategySpec
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """The rule's own readings for one symbol: price-scale lines, and values in their own units."""
    close = closes[symbol]
    lines: dict[str, pd.Series] = {}
    values: dict[str, pd.Series] = {}
    if spec.signal == "sma":
        lines[f"SMA {spec.window}"] = close.rolling(spec.window).mean()
    elif spec.signal == "donchian":
        length = spec.lookbacks[len(spec.lookbacks) // 2]
        high, low = close.rolling(length).max(), close.rolling(length).min()
        lines[f"High {length}"] = high.shift(1)
        lines[f"Midpoint {length}"] = (high + low) / 2
        values["Votes"] = signals.donchian_ensemble(close, spec.lookbacks)
    else:
        values[f"Gain {spec.window}d"] = close.pct_change(spec.window)
    if spec.target_vol is not None:
        values["Volatility scale"] = signals.volatility_scale(
            closes, spec.target_vol, spec.periods_per_year
        )
    return lines, values


Reason = Callable[[int, bool], str]


def reasons(
    spec: StrategySpec,
    closes: pd.Series,
    lines: dict[str, pd.Series],
    values: dict[str, pd.Series],
    first: int,
) -> Reason:
    """Words for why the rule moved at a position: what it read on the close it decided on."""

    def why(position: int, entering: bool) -> str:
        if entering and position <= first:
            return "Already holding when the test begins"
        at = position - 1
        close = float(closes.iloc[at])
        if spec.signal == "sma":
            average = float(lines[f"SMA {spec.window}"].iloc[at])
            side = "above" if entering else "at or below"
            return f"Closed at {close:,.2f}, {side} its {spec.window}-day average of {average:,.2f}"
        if spec.signal == "donchian":
            vote = float(values["Votes"].iloc[at])
            models = len(spec.lookbacks)
            word = "rose to" if entering else "fell to"
            return f"Breakout vote {word} {vote:.0%} of {models} models"
        gain = float(values[f"Gain {spec.window}d"].iloc[at])
        if entering:
            return f"Gained {gain:+.1%} over {spec.window} days, among the best {spec.top_n}"
        if gain <= 0:
            return f"Its {spec.window}-day gain turned to {gain:+.1%}"
        return f"Gained {gain:+.1%} over {spec.window} days but fell out of the best {spec.top_n}"

    return why


def stretches(
    held: pd.Series,
    opens: pd.Series,
    closes: pd.Series,
    spec: StrategySpec,
    first: int,
    why: Reason,
) -> list[TraceTrade]:
    """The holding stretches of `held`, each filled where the backtest fills it.

    A rule that fills at the next open enters at that day's open; one that fills at the close it
    decided on enters at the previous close. A stretch already open when the backtest's window
    starts at `first` is entered at that bar, since the backtest earns nothing before it.
    """
    next_open = spec.fills == "next_open"
    index = held.index
    found: list[TraceTrade] = []
    start: int | None = None

    def fill(position: int) -> tuple[date, float]:
        at = position if next_open or position <= first else position - 1
        return index[at].date(), float((opens if next_open else closes).iloc[at])

    for position, weight in enumerate(held.to_numpy()):
        if weight > 0 and start is None:
            start = position
        elif weight <= 0 and start is not None:
            found.append(trade(held, closes, start, position, fill, why))
            start = None
    if start is not None:
        found.append(trade(held, closes, start, None, fill, why))
    return found


def trade(
    held: pd.Series, closes: pd.Series, start: int, stop: int | None, fill: Fill, why: Reason
) -> TraceTrade:
    entered, bought = fill(start)
    if stop is None:
        sold, exit_price = None, None
        price = float(closes.iloc[-1])
        end = len(held)
    else:
        sold, exit_price = fill(stop)
        price = exit_price
        end = stop
    return TraceTrade(
        entry=entered,
        entry_price=bought,
        exit=sold,
        exit_price=exit_price,
        days=end - start,
        gain=price / bought - 1,
        weight=float(held.iloc[start:end].mean()),
        open=stop is None,
        why_entry=why(start, True),
        why_exit=None if stop is None else why(stop, False),
    )


def trace(spec: StrategySpec, symbol: str, prices: history.Prices, bars: pd.DataFrame) -> Trace:
    """What the rule saw and did for one symbol, day by day, on the prices a backtest uses.

    `bars` holds that symbol's open, high, low, close and volume on the dates of `prices`.
    """
    closes, opens = prices.closes, prices.opens
    warmup = signals.warmup_days(spec)
    target = signals.weights(closes, spec)
    held = target.shift(1).fillna(0.0)
    held.iloc[:warmup] = 0.0
    lines, values = indicators(closes, symbol, spec)
    whole = bars[["open", "high", "low", "close"]].notna().all(axis=1)
    rows = [
        TraceBar(
            day=day.date(),
            open=float(bars.loc[day, "open"]),
            high=float(bars.loc[day, "high"]),
            low=float(bars.loc[day, "low"]),
            close=float(bars.loc[day, "close"]),
            volume=0.0 if pd.isna(bars.loc[day, "volume"]) else float(bars.loc[day, "volume"]),
            target=float(target[symbol].loc[day]) if day >= closes.index[warmup] else 0.0,
            held=float(held[symbol].loc[day]),
            lines={name: number(series.loc[day]) for name, series in lines.items()},
            values={name: number(series.loc[day]) for name, series in values.items()},
        )
        for day in closes.index
        if whole.get(day, False)
    ]
    return Trace(
        strategy=spec.name,
        title=spec.title,
        symbol=symbol,
        evaluated_from=closes.index[warmup].date(),
        fills=spec.fills,
        fee_bps=spec.fee_bps,
        lines=list(lines),
        values=list(values),
        bars=rows,
        trades=stretches(
            held[symbol],
            opens[symbol],
            closes[symbol],
            spec,
            warmup,
            reasons(spec, closes[symbol], lines, values, warmup),
        ),
    )


@cached(BARS, key=lambda stocks, crypto, symbol: symbol, condition=Condition())
def bars_of(
    stocks: StockHistoricalDataClient, crypto: CryptoHistoricalDataClient, symbol: str
) -> pd.DataFrame:
    """The symbol's adjusted daily open, high, low, close and volume, indexed by date."""
    frame = history.ohlcv(stocks, crypto, [symbol])
    return frame.assign(date=pd.to_datetime(frame["date"])).set_index("date")
