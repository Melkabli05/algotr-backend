import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from itertools import batched
from typing import Any, NamedTuple, cast

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from app.quant.history import SIP_DELAY

COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]
INVALID = re.compile(r"invalid symbol: (\S+)")
WORKERS = 4


class Fetched(NamedTuple):
    bars: pd.DataFrame
    rejected: list[str]


def empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype="str"),
            "date": pd.Series(dtype="datetime64[ns]"),
            **{name: pd.Series(dtype="float64") for name in COLUMNS[2:]},
        }
    )


def to_frame(found: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = [
        (symbol, bar["t"][:10], bar["o"], bar["h"], bar["l"], bar["c"], bar["v"])
        for symbol, bars in found.items()
        for bar in bars
    ]
    if not rows:
        return empty()
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    return frame.astype(dict.fromkeys(COLUMNS[2:], "float64"))


def request(
    stocks: StockHistoricalDataClient,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    rejected: list[str],
) -> pd.DataFrame:
    """One multi-symbol request; a symbol Alpaca refuses is set aside and the rest retried."""
    if not symbols:
        return empty()
    try:
        found = stocks.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=list(symbols),
                timeframe=TimeFrame.Day,
                start=start,
                end=end - SIP_DELAY,
                adjustment=Adjustment.ALL,
                feed=DataFeed.SIP,
            )
        )
    except APIError as exc:
        match = INVALID.search(exc.message)
        if match is None or match.group(1) not in symbols:
            raise
        rejected.append(match.group(1))
        remaining = [symbol for symbol in symbols if symbol != match.group(1)]
        return request(stocks, remaining, start, end, rejected)
    return to_frame(cast(dict[str, list[dict[str, Any]]], found))


def fetch(
    stocks: StockHistoricalDataClient,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    size: int,
    progress: Callable[[int, int], None] | None = None,
) -> Fetched:
    """Adjusted daily bars of completed sessions, `size` symbols per request, a few at a time."""
    groups = list(batched(symbols, size, strict=False))
    rejected: list[str] = []
    frames: list[pd.DataFrame] = []
    done = 0
    if progress:
        progress(0, len(symbols))
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        for group, frame in zip(
            groups,
            pool.map(lambda group: request(stocks, group, start, end, rejected), groups),
            strict=True,
        ):
            frames.append(frame)
            done += len(group)
            if progress:
                progress(done, len(symbols))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    found = [frame for frame in frames if not frame.empty]
    bars = pd.concat(found, ignore_index=True) if found else empty()
    return Fetched(bars[bars["date"] < pd.Timestamp(end.date())], rejected)


def last_session(bars: pd.DataFrame) -> date | None:
    return None if bars.empty else bars["date"].max().date()
