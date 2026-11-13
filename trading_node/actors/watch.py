import json
import logging
import threading
import time
from collections.abc import Callable, Iterable

import psycopg

from app.bridge.fanout import WATCH, plain_dsn

logger = logging.getLogger(__name__)

TTL_SECONDS = 90.0
LIMIT = 20
LISTEN_TIMEOUT_SECONDS = 1.0
RETRY_SECONDS = 2.0


class Leases:
    """Symbols the platform asked the node to stream; each lapses unless asked for again."""

    def __init__(
        self,
        fixed: Iterable[str],
        limit: int = LIMIT,
        ttl: float = TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.fixed = set(fixed)
        self.limit = limit
        self.ttl = ttl
        self.clock = clock
        self.asked: dict[str, float] = {}
        self.lock = threading.Lock()

    def renew(self, symbols: Iterable[str]) -> None:
        now = self.clock()
        with self.lock:
            for symbol in symbols:
                if symbol not in self.fixed:
                    self.asked[symbol] = now

    def active(self) -> list[str]:
        """Unlapsed symbols, the most recently asked for first, up to the limit."""
        now = self.clock()
        with self.lock:
            self.asked = {s: t for s, t in self.asked.items() if now - t < self.ttl}
            ranked = sorted(self.asked, key=lambda symbol: self.asked[symbol], reverse=True)
        return ranked[: self.limit]


class WatchListener:
    """Renews the leases from the platform's announcements on the watch channel."""

    def __init__(self, url: str, leases: Leases) -> None:
        self.dsn = plain_dsn(url)
        self.leases = leases
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, name="market-watch", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.thread.join(LISTEN_TIMEOUT_SECONDS * 3)

    def run(self) -> None:
        while not self.stopping.is_set():
            try:
                with psycopg.connect(self.dsn, autocommit=True) as connection:
                    connection.execute(f"LISTEN {WATCH}")
                    while not self.stopping.is_set():
                        for notice in connection.notifies(timeout=LISTEN_TIMEOUT_SECONDS):
                            self.take(notice.payload)
            except Exception:
                logger.exception("The market watch listener dropped; connecting again")
                self.stopping.wait(RETRY_SECONDS)

    def take(self, payload: str) -> None:
        try:
            symbols = json.loads(payload)["symbols"]
        except (ValueError, KeyError, TypeError):
            logger.warning("A market watch message was not understood: %s", payload[:200])
            return
        self.leases.renew(str(symbol) for symbol in symbols)
