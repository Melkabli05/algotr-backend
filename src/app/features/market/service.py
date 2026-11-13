from datetime import UTC, datetime, timedelta
from typing import Any, cast

from alpaca.common.enums import Sort
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient

from app.features.market.schemas import Bar, Clock, Timeframe

TIMEFRAMES: dict[Timeframe, TimeFrame] = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1Day": TimeFrame(1, TimeFrameUnit.Day),
}
LOOKBACK: dict[Timeframe, timedelta] = {
    "1Min": timedelta(days=5),
    "5Min": timedelta(days=10),
    "15Min": timedelta(days=20),
    "1Hour": timedelta(days=90),
    "1Day": timedelta(days=730),
}


def get_bars(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    symbol: str,
    timeframe: Timeframe,
    limit: int,
    feed: DataFeed,
) -> list[Bar]:
    """The latest `limit` bars, oldest first; the wide window lets closed markets return data."""
    common = {
        "symbol_or_symbols": symbol,
        "timeframe": TIMEFRAMES[timeframe],
        "start": datetime.now(UTC) - LOOKBACK[timeframe],
        "limit": limit,
        "sort": Sort.DESC,
    }
    if "/" in symbol:
        response = crypto.get_crypto_bars(CryptoBarsRequest(**common))
    else:
        response = stocks.get_stock_bars(StockBarsRequest(**common, feed=feed))
    rows = cast(dict[str, list[dict[str, Any]]], response).get(symbol, [])
    return [
        Bar(
            time=row["t"],
            open=row["o"],
            high=row["h"],
            low=row["l"],
            close=row["c"],
            volume=row["v"],
        )
        for row in reversed(rows)
    ]


def get_clock(trading: TradingClient) -> Clock:
    return Clock.model_validate(trading.get_clock())
