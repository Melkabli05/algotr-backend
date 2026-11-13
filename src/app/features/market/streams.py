import asyncio
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Protocol, cast

from alpaca.data.live import CryptoDataStream, StockDataStream
from alpaca.data.models import Orderbook as AlpacaBook
from alpaca.data.models import Quote as AlpacaQuote
from alpaca.data.models import Trade as AlpacaTrade
from alpaca.trading.models import TradeUpdate
from alpaca.trading.stream import TradingStream

from app.core.settings import Settings
from app.features.market.hub import Hub
from app.features.market.schemas import OrderUpdate, Quote, StreamEvent, Trade

RESUBSCRIBE_DELAY = 1.0


class Stoppable(Protocol):
    def run(self) -> None: ...

    def stop(self) -> None: ...


def to_quote(quote: AlpacaQuote) -> Quote:
    return Quote(
        symbol=quote.symbol,
        bid_price=quote.bid_price,
        bid_size=quote.bid_size,
        ask_price=quote.ask_price,
        ask_size=quote.ask_size,
        time=quote.timestamp,
    )


def to_trade(trade: AlpacaTrade) -> Trade:
    return Trade(symbol=trade.symbol, price=trade.price, size=trade.size, time=trade.timestamp)


def to_order_update(update: TradeUpdate) -> OrderUpdate:
    order = update.order
    return OrderUpdate(
        event=str(getattr(update.event, "value", update.event)),
        order_id=str(order.id),
        symbol=order.symbol,
        side=order.side,
        status=order.status,
        qty=order.qty,
        filled_qty=order.filled_qty or 0,
        filled_avg_price=order.filled_avg_price,
        time=update.timestamp,
        execution_id=None if update.execution_id is None else str(update.execution_id),
        price=update.price,
        fill_qty=update.qty,
        position_qty=update.position_qty,
    )


class MarketStreams:
    """Runs the Alpaca account and market-data streams in daemon threads, publishing to the hub.

    Alpaca allows one connection per data stream, so `pause_data` frees it for another process.
    """

    def __init__(
        self,
        hub: Hub,
        settings: Settings,
        loop: asyncio.AbstractEventLoop,
        on_book: Callable[[AlpacaBook], None] | None = None,
    ) -> None:
        self._hub = hub
        self._settings = settings
        self._loop = loop
        self._on_book = on_book
        self._account: Stoppable | None = None
        self._data: list[Stoppable] = []
        self._crypto: CryptoDataStream | None = None
        self._stock: StockDataStream | None = None
        self._watched: set[str] = set()

    def start(self) -> None:
        trading = TradingStream(
            self._settings.alpaca_api_key,
            self._settings.alpaca_api_secret,
            paper=self._settings.alpaca_paper,
        )
        trading.subscribe_trade_updates(self._on_trade_update)
        self._account = trading
        self._run(trading)
        self.resume_data()

    def stop(self) -> None:
        self.pause_data()
        if self._account is not None:
            self._stop(self._account)
            self._account = None

    def data_paused(self) -> bool:
        """Whether the market-data streams are handed over to another process."""
        return not self._data

    def pause_data(self) -> None:
        for stream in self._data:
            self._stop(stream)
        self._data = []
        self._crypto = None
        self._stock = None

    def resume_data(self) -> None:
        if self._data:
            return
        key, secret = self._settings.alpaca_api_key, self._settings.alpaca_api_secret
        symbols = [*self._settings.stream_symbols, *sorted(self._watched)]
        stocks = [symbol for symbol in symbols if "/" not in symbol]
        crypto = [symbol for symbol in symbols if "/" in symbol]
        feeds: list[tuple[Any, list[str]]] = []
        if stocks:
            self._stock = StockDataStream(key, secret, feed=self._settings.data_feed)
            feeds.append((self._stock, stocks))
        if crypto:
            self._crypto = CryptoDataStream(key, secret)
            feeds.append((self._crypto, crypto))
        for stream, names in feeds:
            stream.subscribe_quotes(self._on_quote, *names)
            stream.subscribe_trades(self._on_trade, *names)
            if stream is self._crypto and self._on_book is not None:
                cast(CryptoDataStream, stream).subscribe_orderbooks(self._book, *names)
            self._data.append(stream)
            self._run(stream)

    def watch(self, symbols: list[str]) -> None:
        """Stream these symbols too while the API holds the data connection, and no others."""
        wanted = set(symbols)
        added, dropped = wanted - self._watched, self._watched - wanted
        self._watched = wanted
        pairs: list[tuple[Any, bool]] = [(self._stock, False), (self._crypto, True)]
        for stream, names in pairs:
            if stream is None:
                continue
            on = [s for s in added if ("/" in s) == names]
            off = [s for s in dropped if ("/" in s) == names]
            with suppress(Exception):
                if on:
                    stream.subscribe_quotes(self._on_quote, *on)
                    stream.subscribe_trades(self._on_trade, *on)
                if off:
                    stream.unsubscribe_quotes(*off)
                    stream.unsubscribe_trades(*off)

    def _publish(self, event: StreamEvent) -> None:
        self._loop.call_soon_threadsafe(self._hub.publish, event)

    async def _on_trade_update(self, update: TradeUpdate) -> None:
        self._publish(to_order_update(update))

    async def _book(self, book: AlpacaBook | dict[Any, Any]) -> None:
        if self._on_book is not None and isinstance(book, AlpacaBook):
            self._on_book(book)

    def renew_book(self, symbol: str) -> None:
        """Unsubscribe from a coin's book and subscribe again a second later, for a new snapshot."""
        if self._crypto is None:
            return
        with suppress(Exception):
            self._crypto.unsubscribe_orderbooks(symbol)
        timer = threading.Timer(RESUBSCRIBE_DELAY, self._subscribe_book, (symbol,))
        timer.daemon = True
        timer.start()

    def _subscribe_book(self, symbol: str) -> None:
        if self._crypto is not None:
            with suppress(Exception):
                self._crypto.subscribe_orderbooks(self._book, symbol)

    async def _on_quote(self, quote: AlpacaQuote) -> None:
        self._publish(to_quote(quote))

    async def _on_trade(self, trade: AlpacaTrade) -> None:
        self._publish(to_trade(trade))

    @staticmethod
    def _run(stream: Stoppable) -> None:
        threading.Thread(target=stream.run, daemon=True, name=type(stream).__name__).start()

    @staticmethod
    def _stop(stream: Stoppable) -> None:
        with suppress(Exception):
            stream.stop()
