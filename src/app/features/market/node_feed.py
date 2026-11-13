import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from datetime import timedelta
from typing import Any

from alpaca.data.models import Orderbook

from app.features.market.hub import Hub
from app.features.market.schemas import Quote, StreamEvent, Trade

logger = logging.getLogger(__name__)

CHECK_EVERY = timedelta(seconds=10)


def to_event(payload: dict[str, Any]) -> StreamEvent | Orderbook | None:
    """One event of the node's market relay as the hub's quote or trade, or a book snapshot."""
    kind = payload.get("type")
    if kind == "quote":
        return Quote.model_validate(payload)
    if kind == "trade":
        return Trade.model_validate(payload)
    if kind == "book":
        raw = {"t": payload["time"], "r": True}
        raw["b"] = [{"p": price, "s": size} for price, size in payload["bids"]]
        raw["a"] = [{"p": price, "s": size} for price, size in payload["asks"]]
        return Orderbook(payload["symbol"], raw)
    return None


class NodeFeed:
    """Hands Alpaca's one market-data connection to the trading node while it runs.

    While the node is up the API's own data streams are stopped and its quotes, trades and coin
    books come from the node's relay instead; when the node stops, the API streams again.
    """

    def __init__(
        self,
        node_up: Callable[[], bool],
        relay: Callable[[], AsyncIterator[dict[str, Any]]],
        pause: Callable[[], None],
        resume: Callable[[], None],
        hub: Hub,
        on_book: Callable[[Orderbook], None],
    ) -> None:
        self.node_up = node_up
        self.relay = relay
        self.pause = pause
        self.resume = resume
        self.hub = hub
        self.on_book = on_book
        self.listening: asyncio.Task[None] | None = None
        self.watching: asyncio.Task[None] | None = None

    async def step(self) -> None:
        """Follow the node: hand over when it is up, take the connection back when it is not."""
        up = await asyncio.to_thread(self.node_up)
        if up and self.listening is None:
            logger.info("The trading node is up; its relay now carries the market data")
            await asyncio.to_thread(self.pause)
            self.listening = asyncio.create_task(self.listen())
        elif not up and self.listening is not None:
            logger.info("The trading node is down; the API streams the market data again")
            await self.stop_listening()
            await asyncio.to_thread(self.resume)

    async def listen(self) -> None:
        while True:
            try:
                async for payload in self.relay():
                    self.take(payload)
            except Exception:
                logger.exception("The node's market relay dropped; listening again")
            await asyncio.sleep(CHECK_EVERY.total_seconds())

    def take(self, payload: dict[str, Any]) -> None:
        try:
            event = to_event(payload)
        except Exception:
            logger.warning("A relayed market event was not understood: %s", str(payload)[:200])
            return
        if isinstance(event, Orderbook):
            self.on_book(event)
        elif event is not None:
            self.hub.publish(event)

    async def stop_listening(self) -> None:
        if self.listening is not None:
            self.listening.cancel()
            with suppress(asyncio.CancelledError):
                await self.listening
            self.listening = None

    async def watch(self) -> None:
        while True:
            try:
                await self.step()
            except Exception:
                logger.exception("Could not check whether the trading node is up")
            await asyncio.sleep(CHECK_EVERY.total_seconds())

    def start(self) -> None:
        self.watching = asyncio.create_task(self.watch())

    async def stop(self) -> None:
        if self.watching is not None:
            self.watching.cancel()
            with suppress(asyncio.CancelledError):
                await self.watching
        await self.stop_listening()
