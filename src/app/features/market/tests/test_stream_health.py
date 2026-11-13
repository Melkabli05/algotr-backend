import logging

from app.features.market.health import StreamHealth


def record(thread: str, message: str, level: int = logging.INFO) -> logging.LogRecord:
    found = logging.LogRecord("alpaca.data.live.websocket", level, "", 0, message, None, None)
    found.threadName = thread
    return found


def test_connections_and_errors_are_counted_per_stream_thread() -> None:
    health = StreamHealth()
    assert health.filter(record("StockDataStream", "connected to wss://a")) is False
    assert health.filter(record("CryptoDataStream", "connected to wss://b")) is False
    shown = health.filter(record("CryptoDataStream", "data websocket error", logging.WARNING))
    health.filter(record("CryptoDataStream", "connected to wss://b"))
    stats = {item.name: item for item in health.report()}
    assert shown is True
    assert (stats["stock data"].connects, stats["stock data"].reconnects) == (1, 0)
    assert (stats["crypto data"].connects, stats["crypto data"].reconnects) == (2, 1)
    assert stats["crypto data"].errors == 1
