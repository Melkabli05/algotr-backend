import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime

from app.features.market.schemas import OrderUpdate, StreamEvent

QUEUE_SIZE = 1000
RATE_KEPT = 5000
logger = logging.getLogger(__name__)
Listener = Callable[[StreamEvent], None]


class Hub:
    """Fans stream events out to connected clients and replays the latest quote/trade per symbol."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[str]] = set()
        self._latest: dict[tuple[str, str], str] = {}
        self._listeners: list[Listener] = []
        self.seen: dict[tuple[str, str], datetime] = {}
        self.arrivals: dict[tuple[str, str], deque[float]] = {}

    @property
    def subscribers(self) -> int:
        return len(self._queues)

    def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_SIZE)
        for message in self._latest.values():
            queue.put_nowait(message)
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        self._queues.discard(queue)

    def per_minute(self, kind: str, symbol: str, minutes: float = 5) -> float:
        """Events of `kind` for `symbol` per minute over the last few minutes."""
        arrived = self.arrivals.get((kind, symbol))
        if not arrived:
            return 0.0
        cutoff = time.monotonic() - minutes * 60
        return sum(1 for moment in arrived if moment >= cutoff) / minutes

    def last_price(self, symbol: str) -> float | None:
        """The latest trade's price, else the latest quote's mid, from what the hub replays."""
        trade = self._latest.get(("trade", symbol))
        if trade is not None:
            return float(json.loads(trade)["price"])
        quote = self._latest.get(("quote", symbol))
        if quote is None:
            return None
        found = json.loads(quote)
        bid, ask = float(found["bid_price"]), float(found["ask_price"])
        return (bid + ask) / 2 if bid > 0 and ask > 0 else None

    def listen(self, callback: Listener) -> None:
        """Call `callback` with every event as it is published, in the thread that publishes."""
        self._listeners.append(callback)

    def publish(self, event: StreamEvent) -> None:
        message = event.model_dump_json()
        self.seen[(event.type, event.symbol)] = datetime.now(UTC)
        arrived = self.arrivals.setdefault((event.type, event.symbol), deque(maxlen=RATE_KEPT))
        arrived.append(time.monotonic())
        if not isinstance(event, OrderUpdate):
            self._latest[(event.type, event.symbol)] = message
        for queue in self._queues:
            with suppress(asyncio.QueueFull):
                queue.put_nowait(message)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                logger.exception("A stream listener failed")
