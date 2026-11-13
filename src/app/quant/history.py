from datetime import UTC, datetime, timedelta
from threading import Condition
from typing import Any, NamedTuple, cast

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from cachetools import TTLCache, cached

SIP_DELAY = timedelta(minutes=16)
HISTORY_YEARS = 9
CACHE_SECONDS = 3600
PRICES: TTLCache[tuple[str, ...], "Prices"] = TTLCache(maxsize=32, ttl=CACHE_SECONDS)


class NotEnoughHistoryError(ValueError):
    """Raised when Alpaca returns too little daily history for a universe."""


def bars(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    if "/" in symbol:
        crypto_request = CryptoBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start
        )
        found = cast(dict[str, list[dict[str, Any]]], crypto.get_crypto_bars(crypto_request))
    else:
        stock_request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end - SIP_DELAY,
            adjustment=Adjustment.ALL,
            feed=DataFeed.SIP,
        )
        found = cast(dict[str, list[dict[str, Any]]], stocks.get_stock_bars(stock_request))
    return found.get(symbol, [])


class Prices(NamedTuple):
    """Adjusted daily closes and opens on the same dates, one column per symbol."""

    closes: pd.DataFrame
    opens: pd.DataFrame


def daily_prices(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    symbols: list[str],
    years: float,
) -> Prices:
    """Adjusted daily closes and opens, completed days only, on the dates every symbol traded."""
    now = datetime.now(UTC)
    start = now - timedelta(days=round(365 * years))
    closes, opens = {}, {}
    for symbol in symbols:
        rows = bars(stocks, crypto, symbol, start, now)
        if not rows:
            raise NotEnoughHistoryError(f"Alpaca has no daily bars for {symbol}")
        index = pd.to_datetime([row["t"] for row in rows], utc=True).tz_convert(None).normalize()
        closes[symbol] = pd.Series([row["c"] for row in rows], index=index)
        opens[symbol] = pd.Series([row["o"] for row in rows], index=index)
    close = pd.DataFrame(closes).dropna()
    close = close[close.index < pd.Timestamp(now.date())]
    return Prices(close, pd.DataFrame(opens).reindex(index=close.index, columns=close.columns))


def daily_closes(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    symbols: list[str],
    years: float,
) -> pd.DataFrame:
    """Adjusted daily closes, one column per symbol, completed days only."""
    return daily_prices(stocks, crypto, symbols, years).closes


@cached(PRICES, key=lambda stocks, crypto, symbols: tuple(symbols), condition=Condition())
def cached_prices(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    symbols: list[str],
) -> Prices:
    """Daily closes and opens for `symbols`, shared by research, planning and the dashboard."""
    return daily_prices(stocks, crypto, symbols, HISTORY_YEARS)


def cached_closes(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    symbols: list[str],
) -> pd.DataFrame:
    """Daily closes for `symbols`, from the prices cached for an hour."""
    return cached_prices(stocks, crypto, symbols).closes


def ohlcv(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    symbols: list[str],
    years: float = HISTORY_YEARS,
) -> pd.DataFrame:
    """Adjusted daily bars in the long layout FinRL reads (date, tic, open, high, low, close,
    volume), completed days only and only days on which every symbol traded."""
    now = datetime.now(UTC)
    start = now - timedelta(days=round(365 * years))
    frames = []
    for symbol in symbols:
        rows = bars(stocks, crypto, symbol, start, now)
        if not rows:
            raise NotEnoughHistoryError(f"Alpaca has no daily bars for {symbol}")
        frames.append(
            pd.DataFrame(
                {
                    "date": [row["t"][:10] for row in rows],
                    "tic": symbol,
                    "open": [row["o"] for row in rows],
                    "high": [row["h"] for row in rows],
                    "low": [row["l"] for row in rows],
                    "close": [row["c"] for row in rows],
                    "volume": [row["v"] for row in rows],
                }
            )
        )
    frame = pd.concat(frames)
    frame = frame[frame["date"] < now.date().isoformat()]
    counts = frame.groupby("date")["tic"].transform("nunique")
    return frame[counts == len(symbols)].sort_values(["date", "tic"]).reset_index(drop=True)
