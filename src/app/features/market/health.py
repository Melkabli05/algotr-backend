import logging
from datetime import UTC, datetime
from threading import Lock

from pydantic import BaseModel

LOGGERS = ("alpaca.data.live.websocket", "alpaca.trading.stream")
THREADS = {
    "TradingStream": "account",
    "StockDataStream": "stock data",
    "CryptoDataStream": "crypto data",
}
CONNECTED = "connected to"


class StreamStats(BaseModel):
    name: str
    connects: int = 0
    reconnects: int = 0
    errors: int = 0
    last_connect: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None


class StreamHealth(logging.Filter):
    """Counts connections and errors from alpaca-py's stream logs; lets only warnings through.

    Each stream runs on a thread named after its class, so a record's thread names its stream.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lock = Lock()
        self.stats: dict[str, StreamStats] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        name = THREADS.get(record.threadName or "")
        if record.name not in LOGGERS or name is None:
            return record.levelno >= logging.WARNING or record.name not in LOGGERS
        message = record.getMessage()
        now = datetime.now(UTC)
        with self.lock:
            found = self.stats.setdefault(name, StreamStats(name=name))
            if message.startswith(CONNECTED):
                found.connects += 1
                found.reconnects = max(0, found.connects - 1)
                found.last_connect = now
            elif record.levelno >= logging.WARNING:
                found.errors += 1
                found.last_error, found.last_error_at = message[:200], now
        return record.levelno >= logging.WARNING

    def install(self) -> None:
        for name in LOGGERS:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            if self not in logger.filters:
                logger.addFilter(self)

    def report(self) -> list[StreamStats]:
        with self.lock:
            return [item.model_copy() for item in self.stats.values()]
