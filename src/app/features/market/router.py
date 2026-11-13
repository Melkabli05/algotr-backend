import asyncio
import json
import logging
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.requests import HTTPConnection

from app.core.broker import CryptoData, Feed, StockData, Trading
from app.core.settings import Settings, get_settings
from app.core.types import Symbol
from app.features.market import service
from app.features.market.hub import Hub
from app.features.market.schemas import Bar, Clock, DataSource, Timeframe
from app.features.market.watch import Watchlist

router = APIRouter(tags=["market"])
logger = logging.getLogger(__name__)


def get_hub(connection: HTTPConnection) -> Hub:
    hub: Hub = connection.app.state.hub
    return hub


def get_watchlist(connection: HTTPConnection) -> Watchlist:
    watchlist: Watchlist = connection.app.state.watchlist
    return watchlist


@router.get("/market/bars")
def read_bars(
    stocks: StockData,
    crypto: CryptoData,
    feed: Feed,
    symbol: Symbol,
    timeframe: Timeframe = "1Min",
    limit: Annotated[int, Query(ge=1, le=1000)] = 300,
) -> list[Bar]:
    return service.get_bars(stocks, crypto, symbol, timeframe, limit, feed)


@router.get("/market/clock")
def read_clock(trading: Trading) -> Clock:
    """US stock market state; crypto trades around the clock regardless."""
    return service.get_clock(trading)


@router.get("/market/feed")
def read_feed(feed: Feed) -> DataSource:
    """The stock data feed behind the stream, the chart and the price check on market buys."""
    return DataSource(feed=feed.value)


@router.get("/stream/symbols")
def read_stream_symbols(settings: Annotated[Settings, Depends(get_settings)]) -> list[str]:
    return settings.stream_symbols


@router.websocket("/stream")
async def stream(
    websocket: WebSocket,
    hub: Annotated[Hub, Depends(get_hub)],
    watchlist: Annotated[Watchlist, Depends(get_watchlist)],
) -> None:
    """Live quotes, trades and order updates as JSON text messages.

    A client may send `{"action": "watch", "symbols": [...]}` to have more symbols streamed.
    """
    await websocket.accept()
    queue = hub.subscribe()

    async def forward() -> None:
        while True:
            await websocket.send_text(await queue.get())

    async def listen() -> None:
        while True:
            text = await websocket.receive_text()
            try:
                message = json.loads(text)
                if message["action"] == "watch":
                    watchlist.set(websocket, message["symbols"])
                    await asyncio.to_thread(watchlist.refresh)
            except (ValueError, KeyError, TypeError):
                logger.warning("A stream client sent a message that was not understood")

    try:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(forward)
            try:
                await listen()
            except WebSocketDisconnect:
                tasks.cancel_scope.cancel()
    finally:
        hub.unsubscribe(queue)
        watchlist.drop(websocket)
        await asyncio.to_thread(watchlist.refresh)
