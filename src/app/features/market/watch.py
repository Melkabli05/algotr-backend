import asyncio
import logging
import re
import threading
from collections.abc import Callable, Iterable
from contextlib import suppress

logger = logging.getLogger(__name__)

SYMBOL = re.compile(r"^(?:[A-Z]{1,5}(?:[.-][A-Z])?|[A-Z]{2,6}/[A-Z]{3,4})$")
PER_CLIENT = 5
TOTAL = 20
REFRESH_SECONDS = 30.0
Apply = Callable[[list[str]], None]


class Watchlist:
    """The symbols open charts want streamed live beyond the fixed list.

    Each connected client names the symbols it shows; the union is handed on, and again every
    few seconds, because the trading node holds the stream on a lease that lapses.
    """

    def __init__(
        self,
        fixed: Iterable[str],
        apply: Iterable[Apply] = (),
        per_client: int = PER_CLIENT,
        total: int = TOTAL,
    ) -> None:
        self.fixed = set(fixed)
        self.apply = list(apply)
        self.per_client = per_client
        self.total = total
        self.clients: dict[object, list[str]] = {}
        self.table = threading.Lock()
        self.sending = threading.Lock()

    def valid(self, symbols: Iterable[object]) -> list[str]:
        names: list[str] = []
        for symbol in symbols:
            name = str(symbol).strip().upper()
            if SYMBOL.match(name) and name not in self.fixed and name not in names:
                names.append(name)
        return names[: self.per_client]

    def set(self, client: object, symbols: Iterable[object]) -> None:
        names = self.valid(symbols)
        with self.table:
            self.clients[client] = names

    def drop(self, client: object) -> None:
        with self.table:
            self.clients.pop(client, None)

    def wanted(self) -> list[str]:
        with self.table:
            lists = list(self.clients.values())
        found: list[str] = []
        for names in lists:
            found.extend(name for name in names if name not in found)
        return found[: self.total]

    def refresh(self) -> None:
        """Hand the union on; one hand-over at a time, so a slow one never overlaps the next."""
        with self.sending:
            wanted = self.wanted()
            for apply in self.apply:
                try:
                    apply(wanted)
                except Exception:
                    logger.exception("Could not pass the watched symbols on")

    async def run(self) -> None:
        while True:
            await asyncio.to_thread(self.refresh)
            await asyncio.sleep(REFRESH_SECONDS)


async def stop(task: asyncio.Task[None]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
