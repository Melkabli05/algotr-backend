import asyncio
import threading
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, create_autospec

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from trading_node.alpaca_adapter.tests import support

Handler = Callable[[Any], Any]
WAIT_SECONDS = 8.0


class FakeStream:
    """Stands in for an alpaca-py stream: records subscriptions and replays pushed events."""

    def __init__(self) -> None:
        self.handlers: dict[str, Handler] = {}
        self.symbols: dict[str, tuple[str, ...]] = {}
        self.unsubscribed: list[tuple[str, ...]] = []
        self.subscribed = threading.Event()
        self.stopped = threading.Event()
        self.stops = 0

    def _subscribe(self, kind: str, handler: Handler, symbols: tuple[str, ...]) -> None:
        self.handlers[kind] = handler
        self.symbols[kind] = symbols
        self.subscribed.set()

    def subscribe_quotes(self, handler: Handler, *symbols: str) -> None:
        self._subscribe("quotes", handler, symbols)

    def subscribe_trades(self, handler: Handler, *symbols: str) -> None:
        self._subscribe("trades", handler, symbols)

    def subscribe_bars(self, handler: Handler, *symbols: str) -> None:
        self._subscribe("bars", handler, symbols)

    def subscribe_orderbooks(self, handler: Handler, *symbols: str) -> None:
        self._subscribe("orderbooks", handler, symbols)

    def subscribe_trade_updates(self, handler: Handler) -> None:
        self._subscribe("updates", handler, ())

    def unsubscribe_quotes(self, *symbols: str) -> None:
        self.unsubscribed.append(symbols)

    def unsubscribe_trades(self, *symbols: str) -> None:
        self.unsubscribed.append(symbols)

    def unsubscribe_bars(self, *symbols: str) -> None:
        self.unsubscribed.append(symbols)

    def unsubscribe_orderbooks(self, *symbols: str) -> None:
        self.unsubscribed.append(symbols)

    def run(self) -> None:
        self.stopped.wait()

    def stop(self) -> None:
        self.stops += 1
        self.stopped.set()

    def wait_for(self, kind: str) -> Handler:
        deadline = threading.Event()
        for _ in range(int(WAIT_SECONDS * 50)):
            if kind in self.handlers:
                return self.handlers[kind]
            deadline.wait(0.02)
        raise TimeoutError(f"Nothing subscribed to {kind}")

    def push(self, kind: str, data: Any) -> None:
        """Deliver data from a foreign thread with its own loop, as alpaca-py's stream does."""
        handler = self.wait_for(kind)
        thread = threading.Thread(target=lambda: asyncio.run(handler(data)))
        thread.start()
        thread.join(WAIT_SECONDS)


def fake_trading() -> MagicMock:
    trading: MagicMock = create_autospec(TradingClient, instance=True)
    assets = {"BTC/USD": support.asset("BTC/USD"), "AAPL": support.asset("AAPL")}
    trading.get_asset.side_effect = lambda symbol: assets[symbol]
    trading.get_all_assets.return_value = list(assets.values())
    trading.get_account.return_value = support.account()
    trading.get_all_positions.return_value = []
    trading.get_orders.return_value = []
    return trading


def fake_history() -> tuple[MagicMock, MagicMock]:
    stock: MagicMock = create_autospec(StockHistoricalDataClient, instance=True)
    crypto: MagicMock = create_autospec(CryptoHistoricalDataClient, instance=True)
    return stock, crypto
