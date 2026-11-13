from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from statistics import fmean, median, pstdev
from threading import Condition
from typing import Any, cast
from zoneinfo import ZoneInfo

from alpaca.common.enums import Sort
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from cachetools import TTLCache, cached

from app.quant.history import SIP_DELAY, NotEnoughHistoryError

BUCKET_MINUTES = 5
BUCKET = timedelta(minutes=BUCKET_MINUTES)
FIVE_MINUTES = TimeFrame(BUCKET_MINUTES, TimeFrameUnit.Minute)
SESSIONS = 20
MIN_SESSIONS = 5
LOOKBACK = timedelta(days=45)
ENDING_BUCKETS = 6
CACHE_SECONDS = 6 * 3600
QUOTE_WINDOW = timedelta(minutes=30)
QUOTES = 1000
SPREAD_SECONDS = 60


@dataclass(frozen=True)
class Session:
    """A trading day as clock minutes in a zone: when it opens and how long it lasts."""

    zone: ZoneInfo
    opens: int
    minutes: int

    @property
    def buckets(self) -> int:
        return self.minutes // BUCKET_MINUTES

    def bucket_of(self, when: datetime) -> int:
        local = when.astimezone(self.zone)
        return (local.hour * 60 + local.minute - self.opens) // BUCKET_MINUTES


STOCK = Session(ZoneInfo("America/New_York"), 9 * 60 + 30, 390)
COIN = Session(ZoneInfo("UTC"), 0, 1440)


@dataclass(frozen=True)
class Day:
    """One full session: the volume and the closing price of each five-minute bucket."""

    volumes: list[float]
    closes: list[float | None]

    @property
    def total(self) -> float:
        return sum(self.volumes)

    def shares(self) -> list[float]:
        return [volume / self.total for volume in self.volumes]

    def steps(self) -> list[float]:
        """The price change of each bucket after the first trade; a bucket without trades is 0."""
        changes: list[float] = []
        last: float | None = None
        for close in self.closes:
            if last is not None:
                changes.append(0.0 if close is None else close - last)
            last = last if close is None else close
        return changes


@dataclass(frozen=True)
class Profile:
    """A symbol's recent sessions: volume by bucket, daily volume and five-minute volatility."""

    shares: tuple[float, ...]
    sessions: int
    session: Session
    volume: float = 0.0
    step: float = 0.0

    def share_at(self, when: datetime) -> float:
        index = self.session.bucket_of(when)
        return self.shares[min(max(index, 0), len(self.shares) - 1)]


def session_of(symbol: str) -> Session:
    return COIN if "/" in symbol else STOCK


def read_days(rows: list[dict[str, Any]], session: Session) -> list[Day]:
    """The full sessions in `rows`, oldest first; trading outside the session is left out."""
    volumes: dict[date, list[float]] = defaultdict(lambda: [0.0] * session.buckets)
    closes: dict[date, list[float | None]] = defaultdict(lambda: [None] * session.buckets)
    for row in rows:
        when = datetime.fromisoformat(row["t"])
        index = session.bucket_of(when)
        if 0 <= index < session.buckets:
            day = when.astimezone(session.zone).date()
            volumes[day][index] += row["v"]
            closes[day][index] = row.get("c")
    return [
        Day(volumes[day], closes[day])
        for day in sorted(volumes)
        if sum(volumes[day][-ENDING_BUCKETS:]) > 0
    ]


def read_bars(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    if "/" in symbol:
        coins = CryptoBarsRequest(
            symbol_or_symbols=symbol, timeframe=FIVE_MINUTES, start=start, end=end
        )
        found = cast(dict[str, list[dict[str, Any]]], crypto.get_crypto_bars(coins))
    else:
        shares = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=FIVE_MINUTES,
            start=start,
            end=end,
            adjustment=Adjustment.RAW,
            feed=DataFeed.SIP,
        )
        found = cast(dict[str, list[dict[str, Any]]], stocks.get_stock_bars(shares))
    return found.get(symbol, [])


PROFILES: TTLCache[tuple[str, date], Profile] = TTLCache(maxsize=64, ttl=CACHE_SECONDS)


@cached(
    PROFILES,
    key=lambda stocks, crypto, symbol, now: (
        symbol,
        now.astimezone(session_of(symbol).zone).date(),
    ),
    condition=Condition(),
)
def profile(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    symbol: str,
    now: datetime,
) -> Profile:
    """The volume profile of `symbol` over its last full sessions, before today's."""
    session = session_of(symbol)
    today = now.astimezone(session.zone).date()
    end = datetime.combine(today, time(), tzinfo=session.zone).astimezone(UTC)
    rows = read_bars(stocks, crypto, symbol, end - LOOKBACK, end)
    days = read_days(rows, session)[-SESSIONS:]
    if len(days) < MIN_SESSIONS:
        raise NotEnoughHistoryError(
            f"Alpaca has {len(days)} full sessions of {symbol}; a profile needs {MIN_SESSIONS}"
        )
    each = [day.shares() for day in days]
    steps = [change for day in days for change in day.steps()]
    return Profile(
        shares=tuple(fmean(share[index] for share in each) for index in range(session.buckets)),
        sessions=len(days),
        session=session,
        volume=fmean(day.total for day in days),
        step=pstdev(steps) if len(steps) > 1 else 0.0,
    )


SPREADS: TTLCache[str, float | None] = TTLCache(maxsize=64, ttl=SPREAD_SECONDS)


@cached(SPREADS, key=lambda stocks, symbol, now: symbol, condition=Condition())
def spread(stocks: StockHistoricalDataClient, symbol: str, now: datetime) -> float | None:
    """The median quoted spread of a stock's latest SIP quotes that the free plan allows."""
    end = now - SIP_DELAY
    request = StockQuotesRequest(
        symbol_or_symbols=symbol,
        start=end - QUOTE_WINDOW,
        end=end,
        limit=QUOTES,
        sort=Sort.DESC,
        feed=DataFeed.SIP,
    )
    found = cast(dict[str, list[dict[str, Any]]], stocks.get_stock_quotes(request)).get(symbol, [])
    gaps = [row["ap"] - row["bp"] for row in found if row["ap"] > row["bp"] > 0]
    return median(gaps) if gaps else None
