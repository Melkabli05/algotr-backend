import asyncio
from collections.abc import AsyncIterator
from typing import Any

from alpaca.data.models import Orderbook

from app.features.market.hub import Hub
from app.features.market.node_feed import NodeFeed, to_event
from app.features.market.schemas import Quote, Trade

TIME = "2026-11-11T21:00:00+00:00"
QUOTE = {
    "type": "quote",
    "symbol": "BTC/USD",
    "bid_price": 100.0,
    "bid_size": 1.0,
    "ask_price": 101.0,
    "ask_size": 2.0,
    "time": TIME,
}
BOOK = {
    "type": "book",
    "symbol": "BTC/USD",
    "bids": [[100.0, 1.0]],
    "asks": [[101.0, 2.0]],
    "time": TIME,
}


def test_relayed_events_become_the_hubs_own_and_books_become_snapshots() -> None:
    trade = {"type": "trade", "symbol": "SPY", "price": 500.0, "size": 3.0, "time": TIME}

    book = to_event(BOOK)

    assert isinstance(to_event(QUOTE), Quote)
    assert isinstance(to_event(trade), Trade)
    assert isinstance(book, Orderbook) and book.reset
    assert (book.bids[0].price, book.asks[0].size) == (100.0, 2.0)
    assert to_event({"type": "status"}) is None


def test_the_api_lets_go_of_the_data_connection_while_the_node_runs() -> None:
    up = [True]
    calls: list[str] = []
    books: list[Orderbook] = []
    hub = Hub()

    async def relay() -> AsyncIterator[dict[str, Any]]:
        yield QUOTE
        yield BOOK
        await asyncio.Event().wait()

    feed = NodeFeed(
        lambda: up[0],
        relay,
        lambda: calls.append("pause"),
        lambda: calls.append("resume"),
        hub,
        books.append,
    )

    async def run() -> None:
        await feed.step()
        await asyncio.sleep(0.05)
        up[0] = False
        await feed.step()

    asyncio.run(run())

    assert calls == ["pause", "resume"]
    assert ("quote", "BTC/USD") in hub.seen
    assert len(books) == 1
    assert feed.listening is None
