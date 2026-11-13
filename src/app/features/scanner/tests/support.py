from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, create_autospec

import httpx
import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import CryptoBarsRequest, NewsRequest, StockBarsRequest
from alpaca.trading.client import TradingClient

from app.core.tests.support import api_error
from app.features.scanner.lists import Source
from app.features.scanner.service import Scanner

SPUS_HEADER = (
    "Date,Account,StockTicker,CUSIP,SecurityName,Shares,Price,MarketValue,Weightings,NetAssets"
)


def spus_csv(tickers: list[str], as_of: str = "11/02/2026", extras: bool = True) -> str:
    rows = [
        f"{as_of},SPUS,{ticker},037833100,{ticker} Inc,100,10,1000,1.00%,1" for ticker in tickers
    ]
    if extras:
        rows += [
            f"{as_of},SPUS,Cash&Other,Cash&Other,Cash & Other,0,0,0,0.23%,1",
            f"{as_of},SPUS,2602335D,BBG01Y2F01K3,TPG Inc,14708,0.0,0.0,0.00%,1",
            f"{as_of},SPUS,003654100CVR,003654100CVR,ABIOMED INC,405,0.0,0.0,0.00%,1",
        ]
    return "\n".join([SPUS_HEADER, *rows]) + "\n"


def asset(symbol: str, exchange: str = "NASDAQ", tradable: bool = True) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "name": f"{symbol} Inc",
        "exchange": exchange,
        "tradable": tradable,
        "fractionable": True,
        "status": "active",
        "class": "us_equity",
    }


def sessions(through: date, count: int) -> list[date]:
    return [stamp.date() for stamp in pd.bdate_range(end=pd.Timestamp(through), periods=count)]


def yesterday() -> date:
    return datetime.now(UTC).date() - timedelta(days=1)


def last_business_day(before: date) -> date:
    return cast(date, pd.bdate_range(end=pd.Timestamp(before), periods=1)[0].date())


def bar(day: date, close: float, volume: float) -> dict[str, Any]:
    return {
        "t": f"{day.isoformat()}T04:00:00Z",
        "o": close,
        "h": close,
        "l": close,
        "c": close,
        "v": volume,
    }


class Market:
    """A stand-in for Alpaca's daily bars: served up to `through`, adjusted when asked to be."""

    def __init__(self) -> None:
        self.series: dict[str, list[dict[str, Any]]] = {}
        self.through = last_business_day(yesterday())
        self.invalid: set[str] = set()
        self.requests: list[StockBarsRequest] = []

    def add(
        self,
        symbol: str,
        price: float = 50.0,
        volume: float = 1_000_000,
        days: int = 2500,
        drift: float = 0.0005,
        noise: float = 0.0,
    ) -> None:
        """A symbol whose last close is `price`, drifting `drift` a session, through today."""
        end = last_business_day(datetime.now(UTC).date())
        steps = np.random.default_rng(sum(map(ord, symbol))).normal(drift, noise, days)
        path = np.exp(np.cumsum(steps) - steps.sum())
        self.series[symbol] = [
            bar(day, float(price * path[index]), volume)
            for index, day in enumerate(sessions(end, days))
        ]

    def split(self, symbol: str, factor: float, on: date) -> None:
        """Halve (for factor 2) every price before `on`, the way Alpaca rescales adjusted bars."""
        for row in self.series[symbol]:
            if row["t"][:10] < on.isoformat():
                for key in ("o", "h", "l", "c"):
                    row[key] = row[key] / factor
                row["v"] = row["v"] * factor

    def served(self, symbol: str) -> list[dict[str, Any]]:
        return [row for row in self.series[symbol] if row["t"][:10] <= self.through.isoformat()]

    def get_stock_bars(self, request: StockBarsRequest) -> dict[str, list[dict[str, Any]]]:
        symbols = request.symbol_or_symbols
        wanted = [symbols] if isinstance(symbols, str) else list(symbols)
        self.requests.append(request)
        for symbol in wanted:
            if symbol in self.invalid:
                raise api_error(f"invalid symbol: {symbol}")
        start = request.start.date().isoformat() if request.start else ""
        found = {}
        for symbol in wanted:
            rows = (
                [row for row in self.served(symbol) if row["t"][:10] >= start]
                if symbol in self.series
                else []
            )
            if rows:
                found[symbol] = rows
        return found

    def client(self) -> MagicMock:
        mock: MagicMock = create_autospec(StockHistoricalDataClient, instance=True)
        mock.get_stock_bars.side_effect = self.get_stock_bars
        return mock

    def history_requests(self) -> list[list[str]]:
        """Symbols of every request that reached back further than the recent window."""
        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=100)
        return [
            [r.symbol_or_symbols]
            if isinstance(r.symbol_or_symbols, str)
            else list(r.symbol_or_symbols)
            for r in self.requests
            if r.start and r.start.replace(tzinfo=None) < cutoff
        ]


def trading_with(assets: list[dict[str, Any]]) -> MagicMock:
    mock: MagicMock = create_autospec(TradingClient, instance=True)
    mock.get_all_assets.return_value = assets
    return mock


def website(files: dict[str, str]) -> httpx.Client:
    """An HTTP client that serves `files` by URL and answers anything else with 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        text = files.get(str(request.url))
        if text is None:
            return httpx.Response(404)
        return httpx.Response(200, text=text)

    return httpx.Client(transport=httpx.MockTransport(handler))


def stock(
    symbol: str,
    days: int,
    daily: float = 0.0,
    noise: float = 0.01,
    price: float = 50.0,
    volume: float = 1_000_000.0,
    seed: int = 0,
    end: date | None = None,
) -> pd.DataFrame:
    """Daily bars drifting `daily` a session with `noise`, ending on `end` (the last session)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp(end or last_business_day(yesterday())), periods=days)
    close = price * np.exp(np.cumsum(rng.normal(daily, noise, days)))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates.astype("datetime64[ns]"),
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": volume,
        }
    )


def panel_of(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"], ignore_index=True)


def random_panel(symbols: int = 30, days: int = 700, seed: int = 0) -> pd.DataFrame:
    """Random walks with a spread of drifts; every third symbol appears 100 sessions late."""
    frames = []
    for index in range(symbols):
        frame = stock(
            f"S{index:02d}",
            days - (100 if index % 3 == 0 else 0),
            daily=0.0006 - 0.0001 * (index % 7),
            noise=0.015,
            volume=1_000_000.0 + 50_000 * index,
            seed=seed + index,
        )
        frames.append(frame)
    return panel_of(*frames)


URL = "https://lists.test/SPUS.csv"
SOURCE = Source(name="SPUS", url=URL)
NAMES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def noisy_market() -> Market:
    """Six stocks with long noisy histories, enough for the variants to hold something."""
    market = Market()
    for index, symbol in enumerate(NAMES):
        market.add(symbol, drift=0.0004 + 0.0001 * index, noise=0.012)
    market.through = last_business_day(yesterday())
    return market


def listed_scanner(
    tmp_path: Path, market: Market, as_of: str | None = None, **extra: Any
) -> Scanner:
    """A scanner over `market` whose Shariah list holds all six stocks."""
    csv = spus_csv(NAMES) if as_of is None else spus_csv(NAMES, as_of=as_of)
    return Scanner(
        trading_with([asset(name) for name in NAMES]),
        market.client(),
        root=tmp_path / "scanner",
        sources=[SOURCE],
        http=website({URL: csv}),
        **extra,
    )


NOW = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)


def article(
    number: int, headline: str, symbols: list[str], age: timedelta, source: str = "benzinga"
) -> dict[str, Any]:
    created = (NOW - age).isoformat().replace("+00:00", "Z")
    return {
        "id": number,
        "headline": headline,
        "summary": "A summary.",
        "created_at": created,
        "updated_at": created,
        "source": source,
        "symbols": symbols,
        "url": f"https://example.test/{number}",
    }


def at(number: int, headline: str, symbols: list[str], when: datetime) -> dict[str, Any]:
    stamp = when.isoformat().replace("+00:00", "Z")
    base = article(number, headline, symbols, timedelta(0))
    return {**base, "created_at": stamp, "updated_at": stamp}


def client_with(articles: list[dict[str, Any]]) -> MagicMock:
    mock: MagicMock = create_autospec(NewsClient, instance=True)

    def get_news(request: NewsRequest) -> dict[str, list[dict[str, Any]]]:
        wanted = set((request.symbols or "").split(","))
        start, end = request.start, request.end
        assert start is not None and end is not None
        found = [
            item
            for item in articles
            if wanted & set(item["symbols"])
            and start <= datetime.fromisoformat(item["created_at"]) <= end
        ]
        return {"news": found}

    mock.get_news.side_effect = get_news
    return mock


def crypto_client(history: dict[str, list[float]], last: date) -> MagicMock:
    """Daily bars per coin: the closes end on `last`, one a day, as Alpaca returns them."""
    mock: MagicMock = create_autospec(CryptoHistoricalDataClient, instance=True)

    def get_crypto_bars(request: CryptoBarsRequest) -> dict[str, list[dict[str, Any]]]:
        wanted = request.symbol_or_symbols
        found: dict[str, list[dict[str, Any]]] = {}
        for symbol in [wanted] if isinstance(wanted, str) else wanted:
            closes = history.get(symbol.removesuffix("/USD"))
            if closes:
                found[symbol] = [
                    {
                        "t": f"{last - timedelta(days=len(closes) - 1 - index)}T00:00:00Z",
                        "o": close,
                        "h": close,
                        "l": close,
                        "c": close,
                        "v": 1.0,
                    }
                    for index, close in enumerate(closes)
                ]
        return found

    mock.get_crypto_bars.side_effect = get_crypto_bars
    return mock
