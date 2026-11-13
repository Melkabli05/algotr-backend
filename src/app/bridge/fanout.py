import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol

import psycopg

logger = logging.getLogger(__name__)

EVENTS = "node_events"
MARKET = "market_data"
WATCH = "market_watch"


class Fanout(Protocol):
    """Where the trading node's events reach the API; Postgres NOTIFY now, Redis Streams later."""

    def listen(self) -> AsyncIterator[dict[str, Any]]: ...


def plain_dsn(url: str) -> str:
    """SQLAlchemy's `postgresql+psycopg://` URL as the libpq DSN psycopg takes."""
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


class PostgresFanout:
    """LISTEN on the node's channel; each notification carries one event as JSON."""

    def __init__(self, url: str, channel: str = EVENTS) -> None:
        self.dsn = plain_dsn(url)
        self.channel = channel

    async def listen(self) -> AsyncIterator[dict[str, Any]]:
        async with await psycopg.AsyncConnection.connect(self.dsn, autocommit=True) as connection:
            await connection.execute(f"LISTEN {self.channel}")
            async for notice in connection.notifies():
                try:
                    yield json.loads(notice.payload)
                except ValueError:
                    logger.warning("A node event was not JSON: %s", notice.payload[:200])


def announce(url: str, channel: str, event: dict[str, Any]) -> None:
    """Send one event on a channel; the listeners hear it only while they are connected."""
    with psycopg.connect(plain_dsn(url), autocommit=True) as connection:
        connection.execute("select pg_notify(%s, %s)", (channel, json.dumps(event)))
