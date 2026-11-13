import logging
import time
from collections import deque
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from operator import attrgetter
from pathlib import Path
from statistics import median
from threading import Condition

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from cachetools import TTLCache, cachedmethod

from app.features.book import metrics
from app.features.book.book import LEVELS, Book, BookKeeper
from app.features.book.schemas import (
    BookState,
    BookStatus,
    BookView,
    Depth,
    FlowPoint,
    FlowView,
    Impact,
    Level,
    Point,
    Reading,
)
from app.features.book.store import ROOT, BookStore, Flow, Sample

logger = logging.getLogger(__name__)

JOB_ID = "book-sample"
EVERY = timedelta(seconds=1)
USUAL_WINDOW = timedelta(hours=1)
USUAL_MIN = 60
USUAL_SECONDS = 60
WINDOW_SECONDS = 5.0
WINDOWS_KEPT = 12
SNAPSHOT_WAIT = 10.0
RENEW_GAP = 30.0
IMBALANCE_LEVELS = 10
BPS = 1e4


def depths(
    bids: list[tuple[float, float]], asks: list[tuple[float, float]], mid: float
) -> list[Depth]:
    """The size and value resting within 10 to 100 bps of the mid on each side."""
    found = []
    for bps in metrics.BAND_BPS:
        bid, bid_value = metrics.within(bids, mid, bps, "bid")
        ask, ask_value = metrics.within(asks, mid, bps, "ask")
        found.append(Depth(bps=bps, bid=bid, ask=ask, bid_value=bid_value, ask_value=ask_value))
    return found


def reading(depth: int, value: float | None) -> Reading:
    return Reading(depth=depth, value=value, pressure=metrics.pressure(value))


def readings(
    bids: list[tuple[float, float]], asks: list[tuple[float, float]], mid: float
) -> tuple[list[Reading], list[Reading]]:
    """The imbalance by levels (5, 10, 25) and by distance from the mid (10 to 100 bps)."""
    levels = [reading(n, metrics.by_levels(bids, asks, n)) for n in metrics.LEVEL_COUNTS]
    bands = [reading(b, metrics.by_band(bids, asks, mid, b)) for b in metrics.BAND_BPS]
    return levels, bands


class BookNotFoundError(LookupError):
    """Raised for a symbol whose book is not being followed."""


class BookNotReadyError(RuntimeError):
    """Raised when a followed book has no snapshot yet or the stream has gone quiet."""


def middle(bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> tuple[float, float]:
    """The mid price and the spread of a book's best levels."""
    return (bids[0][0] + asks[0][0]) / 2, asks[0][0] - bids[0][0]


def with_totals(levels: list[tuple[float, float]]) -> list[Level]:
    total, shown = 0.0, []
    for price, size in levels:
        total += size
        shown.append(Level(price=price, size=size, total=total))
    return shown


class BookDesk:
    """Follows the order books of some coins, sampling and storing them every second."""

    def __init__(
        self,
        symbols: Sequence[str],
        root: Path = ROOT,
        clock: Callable[[], float] = time.monotonic,
        *,
        max_spread_multiple: float = 0.0,
        max_slippage_bps: float = 0.0,
        slice_slippage_bps: float = 0.0,
        max_imbalance: float = 0.0,
    ) -> None:
        self.max_spread_multiple = max_spread_multiple
        self.max_imbalance = max_imbalance
        self.max_slippage_bps = max_slippage_bps
        self.slice_slippage_bps = slice_slippage_bps
        self.usual: TTLCache[str, float | None] = TTLCache(maxsize=16, ttl=USUAL_SECONDS)
        self.usual_wait = Condition()
        self.keeper = BookKeeper(symbols, clock)
        self.store = BookStore(root)
        self.renew: Callable[[str], None] = lambda symbol: None
        self.scheduler = AsyncIOScheduler(timezone=UTC)
        self.day: date | None = None
        self.window_began: datetime | None = None
        self.recent: dict[str, deque[Flow]] = {
            symbol: deque(maxlen=WINDOWS_KEPT) for symbol in symbols
        }

    def sides(
        self, symbol: str
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]], datetime]:
        """Every level of each side of a ready book and when it last changed."""
        with self.keeper.lock:
            book = self.keeper.books.get(symbol)
            if book is None:
                raise BookNotFoundError(symbol)
            if not book.ready or self.keeper.stale:
                raise BookNotReadyError(symbol)
            bids, asks = book.sides()
            stamp = book.stamp
        if not bids or not asks or stamp is None:
            raise BookNotReadyError(symbol)
        return bids, asks, stamp

    @cachedmethod(attrgetter("usual"), condition=attrgetter("usual_wait"))
    def usual_spread(self, symbol: str) -> float | None:
        """The median spread in bps of the last hour of stored seconds, kept for a minute."""
        found = self.store.spreads(symbol, datetime.now(UTC) - USUAL_WINDOW)
        return median(found) if len(found) >= USUAL_MIN else None

    def view(self, symbol: str, levels: int = 10) -> BookView:
        bids, asks, stamp = self.sides(symbol)
        mid, spread = middle(bids, asks)
        by_levels, by_bands = readings(bids, asks, mid)
        usual = self.usual_spread(symbol)
        return BookView(
            symbol=symbol,
            time=stamp,
            mid=mid,
            spread=spread,
            spread_bps=spread / mid * BPS,
            spread_usual_bps=usual,
            spread_ratio=spread / mid * BPS / usual if usual else None,
            depth=depths(bids, asks, mid),
            bids=with_totals(bids[:levels]),
            asks=with_totals(asks[:levels]),
            imbalance_levels=by_levels,
            imbalance_bands=by_bands,
            flow=self.flow(symbol),
        )

    def impact(self, symbol: str, side: str, qty: float) -> Impact:
        """What a market order of `qty` would find in the book, and whether to send one."""
        bids, asks, _ = self.sides(symbol)
        mid, spread = middle(bids, asks)
        levels = asks if side == "buy" else bids
        found = metrics.walk(levels, qty)
        usual = self.usual_spread(symbol)
        spread_bps = spread / mid * BPS
        reason = self.judge(symbol, side, qty, (levels, mid, spread_bps, usual)) or self.against(
            symbol, side, bids, asks
        )
        return Impact(
            symbol=symbol,
            side="buy" if side == "buy" else "sell",
            qty=qty,
            mid=mid,
            filled=found.filled,
            average=found.average,
            slippage_bps=None
            if found.average is None
            else metrics.slippage(side, found.average, mid),
            levels=found.levels,
            complete=found.filled >= qty,
            spread_bps=spread_bps,
            spread_ratio=spread_bps / usual if usual else None,
            max_qty=self.cap(levels, side),
            advice="limit" if reason else "market",
            reason=reason,
        )

    def judge(
        self,
        symbol: str,
        side: str,
        qty: float,
        book: tuple[list[tuple[float, float]], float, float, float | None],
    ) -> str | None:
        """The reason a market order should be a limit order, or None."""
        levels, mid, spread_bps, usual = book
        if self.max_spread_multiple and usual and spread_bps > self.max_spread_multiple * usual:
            return (
                f"The spread of {symbol} is {spread_bps:.1f} bps, {spread_bps / usual:.1f} times "
                f"its usual {usual:.1f}; use a limit order"
            )
        if not self.max_slippage_bps:
            return None
        found = metrics.walk(levels, qty)
        if found.average is None or found.filled < qty:
            return (
                f"The book of {symbol} holds {found.filled:g} on that side; a market {side} of "
                f"{qty:g} is too large, use a limit order"
            )
        cost = metrics.slippage(side, found.average, mid)
        if cost > self.max_slippage_bps:
            return (
                f"A market {side} of {qty:g} {symbol} would pay about {cost:,.1f} bps over "
                "the mid; use a limit order or a smaller size"
            )
        return None

    def cap(self, levels: list[tuple[float, float]], side: str) -> float | None:
        """The most the book takes within the slice budget past its best price, if limited."""
        if not self.slice_slippage_bps or not levels:
            return None
        return metrics.max_size(levels, side, levels[0][0], self.slice_slippage_bps)

    def refusal(self, symbol: str, side: str, qty: float) -> str | None:
        """Why a market order should not be sent now; None when it may go or the book cannot say."""
        try:
            bids, asks, _ = self.sides(symbol)
        except (BookNotFoundError, BookNotReadyError):
            return None
        mid, spread = middle(bids, asks)
        levels = asks if side == "buy" else bids
        return self.judge(
            symbol, side, qty, (levels, mid, spread / mid * BPS, self.usual_spread(symbol))
        ) or self.against(symbol, side, bids, asks)

    def against(
        self,
        symbol: str,
        side: str,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> str | None:
        """Why a market order runs into strong flow the other way, or None.

        Both must agree: the best ten levels lean past `max_imbalance` against the order, and
        the last minute's order flow imbalance (Cont, Kukanov and Stoikov) points the same way.
        """
        if not self.max_imbalance:
            return None
        lean = metrics.by_levels(bids, asks, IMBALANCE_LEVELS)
        flow = self.flow(symbol)
        if lean is None or flow is None:
            return None
        sign = 1.0 if side == "buy" else -1.0
        if sign * lean <= -self.max_imbalance and sign * flow.ofi_60 < 0:
            pressure = "selling" if side == "buy" else "buying"
            return (
                f"The book of {symbol} leans {abs(lean):.0%} to {pressure} and the last "
                f"minute's flow agrees; a market {side} now runs against it, use a limit order"
            )
        return None

    def max_size(self, symbol: str, side: str) -> float | None:
        """The largest market order the book takes within the slice budget; None for no limit."""
        try:
            bids, asks, _ = self.sides(symbol)
        except (BookNotFoundError, BookNotReadyError):
            return None
        return self.cap(asks if side == "buy" else bids, side)

    def flow(self, symbol: str) -> FlowView | None:
        """The order flow imbalance of the last 5-second window and of the last minute."""
        windows = list(self.recent.get(symbol, ()))
        if not windows:
            return None
        return FlowView(
            time=windows[-1].at,
            ofi=windows[-1].ofi,
            ofi_60=sum(window.ofi for window in windows),
            windows=len(windows),
        )

    def flows(self, symbol: str, minutes: int) -> list[FlowPoint]:
        if symbol not in self.keeper.books:
            raise BookNotFoundError(symbol)
        since = datetime.now(UTC) - timedelta(minutes=minutes)
        return [
            FlowPoint(time=flow.at, span=flow.span, ofi=flow.ofi)
            for flow in self.store.read_flows(symbol, since)
        ]

    def status(self) -> BookStatus:
        with self.keeper.lock:
            stale = self.keeper.stale
            books = [
                BookState(
                    symbol=book.symbol,
                    ready=book.ready and not stale,
                    bid_levels=len(book.bids),
                    ask_levels=len(book.asks),
                )
                for book in self.keeper.books.values()
            ]
            return BookStatus(feed_age=self.keeper.silence, stale=stale, books=books)

    def history(self, symbol: str, minutes: int) -> list[Point]:
        if symbol not in self.keeper.books:
            raise BookNotFoundError(symbol)
        since = datetime.now(UTC) - timedelta(minutes=minutes)
        return [
            Point(
                time=sample.at,
                mid=sample.mid,
                spread_bps=sample.spread / sample.mid * BPS,
                imbalance=readings(sample.bids, sample.asks, sample.mid)[0],
            )
            for sample in self.store.read(symbol, since)
        ]

    def sample(self, now: datetime | None = None) -> None:
        """Store each ready book, and ask the stream for a snapshot of any that lacks one."""
        moment = now or datetime.now(UTC)
        rows: list[Sample] = []
        closed: list[Flow] = []
        renewals: list[str] = []
        with self.keeper.lock:
            at = self.keeper.clock()
            if self.window_due(moment):
                closed = self.close_window(moment)
            for symbol, book in self.keeper.books.items():
                if self.keeper.stale and book.ready:
                    book.ready = False
                    self.keeper.since[symbol] = at
                if book.ready and not book.crossed():
                    bids, asks = book.top(LEVELS)
                    if bids and asks:
                        mid, spread = middle(bids, asks)
                        rows.append(Sample(moment, symbol, mid, spread, bids, asks))
                if self.wants_snapshot(symbol, book, at):
                    renewals.append(symbol)
        if self.day != moment.date():
            self.day = moment.date()
            self.store.prune(self.day)
        if rows:
            self.store.add(rows)
        if closed:
            self.store.add_flows(closed)
        for symbol in renewals:
            logger.warning("The order book of %s has no snapshot; asking for one", symbol)
            self.renew(symbol)

    def window_due(self, moment: datetime) -> bool:
        """A window closes at the first sample that is within half a second of 5 seconds."""
        if self.window_began is None:
            return True
        return (moment - self.window_began).total_seconds() >= WINDOW_SECONDS - 0.5

    def close_window(self, moment: datetime) -> list[Flow]:
        """End the flow window of every book; the first call only starts the clock."""
        began, self.window_began = self.window_began, moment
        found: list[Flow] = []
        for symbol, book in self.keeper.books.items():
            flow = book.take_flow()
            if began is not None and book.ready:
                found.append(Flow(moment, symbol, (moment - began).total_seconds(), flow))
                self.recent[symbol].append(found[-1])
        return found

    def wants_snapshot(self, symbol: str, book: Book, at: float) -> bool:
        if book.ready and book.crossed():
            book.ready = False
            self.keeper.since[symbol] = at - SNAPSHOT_WAIT
        waiting = not book.ready and at - self.keeper.since[symbol] >= SNAPSHOT_WAIT
        if not waiting or at - self.keeper.renewed[symbol] < RENEW_GAP:
            return False
        self.keeper.renewed[symbol] = at
        self.keeper.since[symbol] = at
        return True

    def start(self) -> None:
        self.scheduler.add_job(
            self.sample,
            IntervalTrigger(seconds=EVERY.total_seconds()),
            id=JOB_ID,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
