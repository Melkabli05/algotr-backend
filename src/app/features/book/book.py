import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

from alpaca.data.models import Orderbook

from app.features.book import metrics

LEVELS = 25
STALE_AFTER = 5.0


@dataclass
class Book:
    """One coin's book: the size at each price on each side, rebuilt from every snapshot."""

    symbol: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    ready: bool = False
    stamp: datetime | None = None
    flow: float = 0.0

    def apply(self, message: Orderbook) -> bool:
        """Take in a message; False when it is an update to a book that has no snapshot yet."""
        if message.reset:
            self.bids.clear()
            self.asks.clear()
            self.ready = True
        elif not self.ready:
            return False
        before = None if message.reset else self.best()
        for side, levels in ((self.bids, message.bids), (self.asks, message.asks)):
            for level in levels:
                if level.size > 0:
                    side[level.price] = level.size
                else:
                    side.pop(level.price, None)
        after = self.best()
        if before is not None and after is not None:
            self.flow += metrics.order_flow(before, after)
        self.stamp = message.timestamp
        return True

    def best(self) -> metrics.Best | None:
        """The best bid and the best ask, each as (price, size); None if a side is empty."""
        if not self.bids or not self.asks:
            return None
        bid, ask = max(self.bids), min(self.asks)
        return (bid, self.bids[bid]), (ask, self.asks[ask])

    def take_flow(self) -> float:
        """The order flow imbalance since it was last taken."""
        flow, self.flow = self.flow, 0.0
        return flow

    def top(
        self, count: int = LEVELS
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """The best `count` levels of each side, best first, as (price, size)."""
        return sorted(self.bids.items(), reverse=True)[:count], sorted(self.asks.items())[:count]

    def sides(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Every level of each side, best first."""
        return sorted(self.bids.items(), reverse=True), sorted(self.asks.items())

    def crossed(self) -> bool:
        return bool(self.bids and self.asks and max(self.bids) >= min(self.asks))


class BookKeeper:
    """The books of the coins being followed: written by the stream's thread, read by the rest."""

    def __init__(self, symbols: Sequence[str], clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.lock = Lock()
        self.books = {symbol: Book(symbol) for symbol in symbols}
        self.heard = clock()
        self.since = dict.fromkeys(symbols, clock())
        self.renewed = dict.fromkeys(symbols, float("-inf"))

    def handle(self, message: Orderbook) -> None:
        with self.lock:
            book = self.books.get(message.symbol)
            if book is not None:
                self.heard = self.clock()
                book.apply(message)

    @property
    def silence(self) -> float:
        """Seconds since any book last heard from the stream."""
        return self.clock() - self.heard

    @property
    def stale(self) -> bool:
        return self.silence > STALE_AFTER
