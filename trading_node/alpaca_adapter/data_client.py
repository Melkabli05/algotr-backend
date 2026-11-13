import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.live import CryptoDataStream, StockDataStream
from alpaca.data.models import Bar as AlpacaBar
from alpaca.data.models import BarSet
from alpaca.data.models import Orderbook as AlpacaBook
from alpaca.data.models import Quote as AlpacaQuote
from alpaca.data.models import Trade as AlpacaTrade
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from nautilus_trader import live
from nautilus_trader.common import Logger
from nautilus_trader.live import BarsResponse, InstrumentResponse, InstrumentsResponse
from nautilus_trader.live.clients import MarketDataClient
from nautilus_trader.model import BarAggregation, BarType, InstrumentId

from trading_node.alpaca_adapter.config import AlpacaDataClientConfig
from trading_node.alpaca_adapter.constants import ALPACA_VENUE
from trading_node.alpaca_adapter.parsing import (
    AlpacaInstrument,
    instrument_id_of,
    is_crypto,
    parse_bar,
    parse_book,
    parse_quote,
    parse_trade,
    symbol_of,
    to_timeframe,
)
from trading_node.alpaca_adapter.providers import AlpacaInstrumentProvider
from trading_node.alpaca_adapter.sdk import call, call_soon
from trading_node.alpaca_adapter.streams import StreamThread

BAR_GAP = timedelta(minutes=1, seconds=30)


@dataclass
class DataFeedThread:
    stream: StockDataStream | CryptoDataStream
    thread: StreamThread


class AlpacaDataClient(MarketDataClient):
    """Streams Alpaca quotes, trades, 1-minute bars and coin books, and serves historical bars."""

    def __init__(
        self,
        *,
        name: str,
        config: AlpacaDataClientConfig,
        cache: live.ClientCache,
        clock: Any,
        instrument_provider: AlpacaInstrumentProvider,
    ) -> None:
        super().__init__(
            name=name,
            config=config,
            cache=cache,
            clock=clock,
            venue=ALPACA_VENUE,
            instrument_provider=instrument_provider,
        )
        self._alpaca = config
        self._provider = instrument_provider
        self._log = Logger(type(self).__name__)
        self._feeds: dict[bool, DataFeedThread] = {}
        self._bar_types: dict[str, BarType] = {}
        self._last_bar: dict[str, datetime] = {}
        self._stock_history = StockHistoricalDataClient(config.api_key, config.api_secret)
        self._crypto_history = CryptoHistoricalDataClient(config.api_key, config.api_secret)

    async def _connect(self) -> None:
        await self._provider.initialize()
        for instrument in self._provider.list_all():
            self._handle_instrument(instrument)

    async def _disconnect(self) -> None:
        for feed in self._feeds.values():
            await asyncio.to_thread(feed.thread.stop)
        self._feeds.clear()

    def _feed(self, crypto: bool) -> DataFeedThread:
        if crypto not in self._feeds:
            key, secret = self._alpaca.api_key, self._alpaca.api_secret
            stream = (
                CryptoDataStream(key, secret)
                if crypto
                else StockDataStream(key, secret, feed=DataFeed(self._alpaca.feed))
            )
            name = "alpaca-crypto-data" if crypto else "alpaca-stock-data"
            self._feeds[crypto] = DataFeedThread(stream, StreamThread(stream, name))
        return self._feeds[crypto]

    def _instrument_of(self, symbol: str) -> AlpacaInstrument | None:
        instrument = self._provider.find_instrument(instrument_id_of(symbol))
        if instrument is None:
            self._log.warning(f"Dropping data for unloaded instrument {symbol}")
        return instrument

    async def _subscribe_quotes(self, command: live.SubscribeQuotes) -> None:
        feed = self._feed(is_crypto(command.instrument_id))
        feed.stream.subscribe_quotes(self._on_quote, symbol_of(command.instrument_id))
        feed.thread.start()

    async def _unsubscribe_quotes(self, command: live.UnsubscribeQuotes) -> None:
        self._feed(is_crypto(command.instrument_id)).stream.unsubscribe_quotes(
            symbol_of(command.instrument_id)
        )

    async def _subscribe_trades(self, command: live.SubscribeTrades) -> None:
        feed = self._feed(is_crypto(command.instrument_id))
        feed.stream.subscribe_trades(self._on_trade, symbol_of(command.instrument_id))
        feed.thread.start()

    async def _unsubscribe_trades(self, command: live.UnsubscribeTrades) -> None:
        self._feed(is_crypto(command.instrument_id)).stream.unsubscribe_trades(
            symbol_of(command.instrument_id)
        )

    async def _subscribe_book_deltas(self, command: live.SubscribeBookDeltas) -> None:
        if not is_crypto(command.instrument_id):
            raise ValueError("Alpaca streams order books for crypto only")
        feed = self._feed(True)
        cast(CryptoDataStream, feed.stream).subscribe_orderbooks(
            self._on_book, symbol_of(command.instrument_id)
        )
        feed.thread.start()

    async def _unsubscribe_book_deltas(self, command: live.UnsubscribeBookDeltas) -> None:
        if is_crypto(command.instrument_id):
            stream = cast(CryptoDataStream, self._feed(True).stream)
            stream.unsubscribe_orderbooks(symbol_of(command.instrument_id))

    async def _subscribe_bars(self, command: live.SubscribeBars) -> None:
        spec = command.bar_type.spec
        if spec.step != 1 or spec.aggregation != BarAggregation.MINUTE:
            raise ValueError("Alpaca streams 1-minute bars only; aggregate others internally")
        instrument_id = command.bar_type.instrument_id
        self._bar_types[symbol_of(instrument_id)] = command.bar_type
        feed = self._feed(is_crypto(instrument_id))
        feed.stream.subscribe_bars(self._on_bar, symbol_of(instrument_id))
        feed.thread.start()

    async def _unsubscribe_bars(self, command: live.UnsubscribeBars) -> None:
        instrument_id = command.bar_type.instrument_id
        self._bar_types.pop(symbol_of(instrument_id), None)
        self._feed(is_crypto(instrument_id)).stream.unsubscribe_bars(symbol_of(instrument_id))

    async def _on_quote(self, quote: AlpacaQuote | dict[Any, Any]) -> None:
        if isinstance(quote, AlpacaQuote):
            call_soon(self.loop, self._emit_quote, quote)

    async def _on_trade(self, trade: AlpacaTrade | dict[Any, Any]) -> None:
        if isinstance(trade, AlpacaTrade):
            call_soon(self.loop, self._emit_trade, trade)

    async def _on_book(self, book: AlpacaBook | dict[Any, Any]) -> None:
        if isinstance(book, AlpacaBook):
            call_soon(self.loop, self._emit_book, book)

    async def _on_bar(self, bar: AlpacaBar | dict[Any, Any]) -> None:
        if isinstance(bar, AlpacaBar) and self.loop is not None:
            asyncio.run_coroutine_threadsafe(self._receive_bar(bar), self.loop)

    async def _receive_bar(self, bar: AlpacaBar) -> None:
        """A live bar after a gap, as after a reconnect, is preceded by the bars it missed,
        fetched over REST from the last one seen, so no minute is lost at the switchover."""
        last = self._last_bar.get(bar.symbol)
        if last is not None and bar.timestamp - last > BAR_GAP:
            for missed in await self._missed_bars(bar.symbol, last, bar.timestamp):
                self._emit_bar(missed)
        if last is None or bar.timestamp > last:
            self._last_bar[bar.symbol] = bar.timestamp
        self._emit_bar(bar)

    async def _missed_bars(self, symbol: str, after: datetime, before: datetime) -> list[AlpacaBar]:
        params: dict[str, Any] = {
            "symbol_or_symbols": symbol,
            "timeframe": TimeFrame.Minute,
            "start": after + timedelta(minutes=1),
            "end": before - timedelta(seconds=1),
        }
        try:
            if "/" in symbol:
                found = await call(
                    self._crypto_history.get_crypto_bars, BarSet, CryptoBarsRequest(**params)
                )
            else:
                request = StockBarsRequest(**params, feed=DataFeed(self._alpaca.feed))
                found = await call(self._stock_history.get_stock_bars, BarSet, request)
        except Exception as exc:
            self._log.warning(f"Could not backfill {symbol} after a gap: {exc}")
            return []
        bars = found.data.get(symbol, [])
        return [item for item in bars if after < item.timestamp < before]

    def _emit_quote(self, quote: AlpacaQuote) -> None:
        if (instrument := self._instrument_of(quote.symbol)) is not None:
            self._handle_data(parse_quote(quote, instrument, self.clock.timestamp_ns()))

    def _emit_trade(self, trade: AlpacaTrade) -> None:
        if (instrument := self._instrument_of(trade.symbol)) is not None:
            self._handle_data(parse_trade(trade, instrument, self.clock.timestamp_ns()))

    def _emit_book(self, book: AlpacaBook) -> None:
        if (instrument := self._instrument_of(book.symbol)) is not None:
            self._handle_data(parse_book(book, instrument, self.clock.timestamp_ns()))

    def _emit_bar(self, bar: AlpacaBar) -> None:
        bar_type = self._bar_types.get(bar.symbol)
        instrument = self._instrument_of(bar.symbol)
        if bar_type is not None and instrument is not None:
            self._handle_data(parse_bar(bar, bar_type, instrument, self.clock.timestamp_ns()))

    async def _instrument(self, instrument_id: InstrumentId) -> AlpacaInstrument:
        await self._provider.load_async(instrument_id)
        instrument = self._provider.find_instrument(instrument_id)
        if instrument is None:
            raise ValueError(f"Unknown or untradable instrument {instrument_id}")
        return instrument

    async def _request_instrument(self, request: live.RequestInstrument) -> None:
        instrument = await self._instrument(request.instrument_id)
        self._handle_instrument(instrument)
        self._handle_response(
            InstrumentResponse(
                self.client_id,
                request.instrument_id,
                instrument,
                request.request_id,
                self.clock.timestamp_ns(),
                request.start_ns,
                request.end_ns,
                request.params,
            )
        )

    async def _request_instruments(self, request: live.RequestInstruments) -> None:
        await self._provider.load_all_async()
        self._handle_response(
            InstrumentsResponse(
                self.client_id,
                ALPACA_VENUE,
                self._provider.list_all(),
                request.request_id,
                self.clock.timestamp_ns(),
                request.start_ns,
                request.end_ns,
                request.params,
            )
        )

    async def _fetch_bars(self, instrument: AlpacaInstrument, request: live.RequestBars) -> BarSet:
        params: dict[str, Any] = {
            "symbol_or_symbols": symbol_of(instrument.id),
            "timeframe": to_timeframe(request.bar_type),
            "start": request.start,
            "end": request.end,
            "limit": request.limit,
        }
        if is_crypto(instrument.id):
            return await call(
                self._crypto_history.get_crypto_bars, BarSet, CryptoBarsRequest(**params)
            )
        stock_request = StockBarsRequest(**params, feed=DataFeed(self._alpaca.feed))
        return await call(self._stock_history.get_stock_bars, BarSet, stock_request)

    async def _request_bars(self, request: live.RequestBars) -> None:
        if request.start is None:
            raise ValueError("Alpaca bar requests need a start time")
        instrument = await self._instrument(request.bar_type.instrument_id)
        response = await self._fetch_bars(instrument, request)
        ts_init = self.clock.timestamp_ns()
        bars = [
            parse_bar(bar, request.bar_type, instrument, ts_init)
            for bar in response.data.get(symbol_of(instrument.id), [])
        ]
        self._handle_response(
            BarsResponse(
                self.client_id,
                request.bar_type,
                bars,
                request.request_id,
                ts_init,
                request.start_ns,
                request.end_ns,
                request.params,
            )
        )
