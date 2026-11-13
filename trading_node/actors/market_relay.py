from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Self

from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import ActorId, BookType, InstrumentId, QuoteTick, TradeTick

from trading_node.actors.watch import Leases, WatchListener
from trading_node.alpaca_adapter.parsing import instrument_id_of, is_crypto, symbol_of
from trading_node.state_writer import StateWriter

EVERY = timedelta(milliseconds=250)
BOOK_EVERY_MS = 500
BOOK_DEPTH = 20
TRADES_KEPT = 200
LOAD_SECONDS = 10.0
RETRY_SECONDS = 300.0


def moment(nanos: int) -> str:
    return datetime.fromtimestamp(nanos / 1e9, UTC).isoformat()


def quote_event(quote: QuoteTick) -> dict[str, Any]:
    return {
        "type": "quote",
        "symbol": symbol_of(quote.instrument_id),
        "bid_price": quote.bid_price.as_double(),
        "bid_size": quote.bid_size.as_double(),
        "ask_price": quote.ask_price.as_double(),
        "ask_size": quote.ask_size.as_double(),
        "time": moment(quote.ts_event),
    }


def trade_event(trade: TradeTick) -> dict[str, Any]:
    return {
        "type": "trade",
        "symbol": symbol_of(trade.instrument_id),
        "price": trade.price.as_double(),
        "size": trade.size.as_double(),
        "time": moment(trade.ts_event),
    }


def book_event(book: Any, depth: int = BOOK_DEPTH) -> dict[str, Any]:
    def levels(found: dict[Any, Any]) -> list[list[float]]:
        return [[float(price), float(size)] for price, size in found.items()]

    return {
        "type": "book",
        "symbol": symbol_of(book.instrument_id),
        "bids": levels(book.bids_to_dict(depth)),
        "asks": levels(book.asks_to_dict(depth)),
        "time": moment(book.ts_last),
    }


class MarketRelay(DataActor):
    """Relays the node's market data to the platform; the node holds Alpaca's one data connection.

    Quotes and books are sent latest-first a few times a second; every trade print is sent.
    Symbols the platform asks for through `leases` are streamed while they stay asked for.
    """

    def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
        return super().__new__(cls)

    def __init__(
        self,
        writer: StateWriter,
        instrument_ids: Sequence[InstrumentId],
        leases: Leases | None = None,
        listener: WatchListener | None = None,
    ) -> None:
        super().__init__(DataActorConfig(actor_id=ActorId("MarketRelay-001")))
        self.writer = writer
        self.instrument_ids = list(instrument_ids)
        self.leases = leases
        self.listener = listener
        self.loading: dict[str, float] = {}
        self.streaming: set[str] = set()
        self.failed: dict[str, float] = {}
        self.latest: dict[tuple[str, str], dict[str, Any]] = {}
        self.trades: list[dict[str, Any]] = []

    def on_start(self) -> None:
        for instrument_id in self.instrument_ids:
            self.subscribe_quotes(instrument_id)
            self.subscribe_trades(instrument_id)
            if is_crypto(instrument_id):
                self.subscribe_book_at_interval(
                    instrument_id, BookType.L2_MBP, BOOK_EVERY_MS, depth=BOOK_DEPTH
                )
        if self.listener is not None:
            self.listener.start()
        self.clock.set_timer("market-relay", EVERY, callback=lambda _: self.tick())

    def on_stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()

    def tick(self) -> None:
        self.sync()
        self.flush()

    def seconds(self) -> float:
        return self.clock.timestamp_ns() / 1e9

    def sync(self) -> None:
        """Stream the symbols asked for once their instruments load, and no others."""
        if self.leases is None:
            return
        now = self.seconds()
        wanted = self.leases.active()
        for symbol in wanted:
            if symbol in self.streaming or symbol in self.loading:
                continue
            if now - self.failed.get(symbol, -RETRY_SECONDS) < RETRY_SECONDS:
                continue
            self.loading[symbol] = now
            self.request_instrument(instrument_id_of(symbol))
        for symbol, since in list(self.loading.items()):
            instrument_id = instrument_id_of(symbol)
            if symbol not in wanted:
                del self.loading[symbol]
            elif self.cache.instrument(instrument_id) is not None:
                del self.loading[symbol]
                self.streaming.add(symbol)
                self.subscribe_quotes(instrument_id)
                self.subscribe_trades(instrument_id)
            elif now - since > LOAD_SECONDS:
                del self.loading[symbol]
                self.failed[symbol] = now
                self.log.warning(f"Could not load {symbol} to stream it")
        for symbol in self.streaming - set(wanted):
            self.streaming.discard(symbol)
            instrument_id = instrument_id_of(symbol)
            self.unsubscribe_quotes(instrument_id)
            self.unsubscribe_trades(instrument_id)

    def on_quote(self, quote: QuoteTick) -> None:
        event = quote_event(quote)
        self.latest[("quote", event["symbol"])] = event

    def on_trade(self, trade: TradeTick) -> None:
        self.trades = [*self.trades[-(TRADES_KEPT - 1) :], trade_event(trade)]

    def on_book(self, book: Any) -> None:
        event = book_event(book)
        self.latest[("book", event["symbol"])] = event

    def flush(self) -> None:
        events = [*self.latest.values(), *self.trades]
        self.latest, self.trades = {}, []
        if events:
            self.writer.market(events)
