import time
from datetime import UTC, date, datetime, timedelta
from statistics import median
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.core.broker import Trading
from app.core.database import transaction
from app.core.settings import get_settings
from app.features.market.health import StreamStats
from app.features.orders.service import list_orders

router = APIRouter(prefix="/system", tags=["system"])

Level = Literal["ok", "warn", "error", "idle"]
CLOSED_ORDERS = ("filled", "canceled", "expired", "rejected")
RECENT_ORDERS = 100
CRYPTO_QUOTE = (timedelta(seconds=30), timedelta(minutes=2))
STOCK_QUOTE = (timedelta(minutes=1), timedelta(minutes=5))
NEWS = (timedelta(minutes=10), timedelta(minutes=30))
LEDGER = (timedelta(minutes=20), timedelta(hours=1))
CRYPTO_TABLE = (timedelta(hours=36), timedelta(hours=72))


class Feed(BaseModel):
    name: str
    per_minute: float | None = None
    kind: Literal["quotes", "news", "ledger", "scan", "book", "crypto"]
    symbol: str | None = None
    last: datetime | None
    age_seconds: float | None
    level: Level
    detail: str


class Freshness(BaseModel):
    market_open: bool
    feeds: list[Feed]
    stale: list[str]


class Connection(BaseModel):
    heartbeat_seconds: float | None = None
    orders_today: int = 0
    streams: list[StreamStats]
    connected: dict[str, bool]
    ack_ms: float | None
    ack_samples: int
    fill_rate: float | None
    orders: dict[str, int]


def level_of(age: timedelta | None, limits: tuple[timedelta, timedelta]) -> Level:
    if age is None:
        return "error"
    return "ok" if age <= limits[0] else "warn" if age <= limits[1] else "error"


def feed(
    name: str,
    kind: Any,
    last: datetime | None,
    now: datetime,
    limits: tuple[timedelta, timedelta],
    symbol: str | None = None,
    idle: bool = False,
) -> Feed:
    age = now - last if last else None
    level: Level = "idle" if idle else level_of(age, limits)
    detail = (
        "market closed" if idle else "never" if age is None else f"{age.total_seconds():.0f}s ago"
    )
    return Feed(
        name=name,
        kind=kind,
        symbol=symbol,
        last=last,
        age_seconds=age.total_seconds() if age else None,
        level=level,
        detail=detail,
    )


def last_session(today: date) -> date:
    day = today - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def is_open(clock: Any) -> bool:
    """Whether Alpaca's clock says the market is open; the client hands it back as a dict."""
    if isinstance(clock, dict):
        return bool(clock.get("is_open", False))
    return bool(getattr(clock, "is_open", False))


def freshness(state: Any, market_open: bool, now: datetime) -> Freshness:
    """How old each feed's latest data is, against thresholds scaled to how often it updates."""
    feeds = []
    for symbol in get_settings().stream_symbols:
        coin = "/" in symbol
        seen = state.hub.seen.get(("quote", symbol))
        found = feed(
            f"Quotes {symbol}",
            "quotes",
            seen,
            now,
            CRYPTO_QUOTE if coin else STOCK_QUOTE,
            symbol,
            idle=not coin and not market_open,
        )
        found.per_minute = state.hub.per_minute("quote", symbol)
        feeds.append(found)
    scanner = state.scanner.status()
    feeds.append(feed("News", "news", scanner.news.polled, now, NEWS))
    feeds.append(feed("Ledger sync", "ledger", state.ledger.synced, now, LEDGER))
    feeds.append(feed("Crypto table", "crypto", scanner.crypto.refreshed, now, CRYPTO_TABLE))
    scanned = scanner.last.scanned if scanner.last else None
    behind = scanned is None or scanned < last_session(now.date())
    feeds.append(
        Feed(
            name="Market scan",
            kind="scan",
            last=None,
            age_seconds=None,
            level="warn" if behind else "ok",
            detail=f"session {scanned}" if scanned else "no scan yet",
        )
    )
    book = state.book.status()
    feeds.append(
        Feed(
            name="Coin order books",
            kind="book",
            last=None,
            age_seconds=book.feed_age,
            level="warn" if book.stale else "ok",
            detail=f"{book.feed_age:.0f}s since the last update",
        )
    )
    stale = [item.name for item in feeds if item.level in ("warn", "error")]
    return Freshness(market_open=market_open, feeds=feeds, stale=stale)


def connected(stats: StreamStats) -> bool:
    if stats.last_connect is None:
        return False
    return stats.last_error_at is None or stats.last_error_at < stats.last_connect


def connection(state: Any, orders: list[Any]) -> Connection:
    """The streams' connections and errors, the order round trip and how orders ended."""
    stats = state.stream_health.report()
    submits = [
        entry.payload
        for entry in state.audit.entries()[-500:]
        if entry.kind == "order" and entry.payload.get("action") == "submit"
    ]
    times = [
        float(found["result"]["ms"])
        for found in submits
        if isinstance(found.get("result"), dict) and "ms" in found["result"]
    ]
    counts = dict.fromkeys(CLOSED_ORDERS, 0)
    for order in orders:
        if order.status in counts:
            counts[order.status] += 1
    closed = sum(counts.values())
    alerts = getattr(state, "alerts", None)
    today = datetime.now(UTC).date()
    return Connection(
        heartbeat_seconds=time.monotonic() - alerts.beat if alerts is not None else None,
        orders_today=sum(
            1 for order in orders if order.created_at and order.created_at.date() == today
        ),
        streams=stats,
        connected={item.name: connected(item) for item in stats},
        ack_ms=median(times) if times else None,
        ack_samples=len(times),
        fill_rate=counts["filled"] / closed if closed else None,
        orders=counts,
    )


@router.get("/freshness")
def read_freshness(request: Request, trading: Trading) -> Freshness:
    """Each feed's last update, its age and whether it is stale for how often it should arrive."""
    clock = trading.get_clock()
    return freshness(request.app.state, is_open(clock), datetime.now(UTC))


@router.get("/connection")
def read_connection(request: Request) -> Connection:
    """Alpaca's streams, the median order round trip and the fill rate of the last 100 orders."""
    with transaction() as database:
        orders = list_orders(database, RECENT_ORDERS)
    return connection(request.app.state, orders)
