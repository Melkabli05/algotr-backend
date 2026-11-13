import asyncio
import gc
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import requests
from alpaca.data.models import BarSet
from nautilus_trader.common import Environment
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.live import LiveNode
from nautilus_trader.model import (
    AccountId,
    BarType,
    InstrumentId,
    OrderSide,
    OrderStatus,
    Price,
    Quantity,
    QuoteTick,
    TimeInForce,
    TraderId,
)
from nautilus_trader.trading import Strategy

from trading_node.actors import market_relay
from trading_node.actors.market_relay import MarketRelay
from trading_node.actors.watch import Leases
from trading_node.alpaca_adapter import data_client, execution_client, factories
from trading_node.alpaca_adapter.config import AlpacaDataClientConfig, AlpacaExecClientConfig
from trading_node.alpaca_adapter.constants import ALPACA
from trading_node.alpaca_adapter.factories import (
    AlpacaDataClientFactory,
    AlpacaExecutionClientFactory,
)
from trading_node.alpaca_adapter.tests import fakes, support
from trading_node.state_writer import StateWriter

BTC = InstrumentId.from_str("BTC/USD.ALPACA")
NODE_TIMEOUT_SECONDS = 15
PAPER_ACCOUNT = AccountId("ALPACA-PAPER-001")


@dataclass
class Rig:
    trading: MagicMock
    trading_stream: fakes.FakeStream
    stock_stream: fakes.FakeStream
    crypto_stream: fakes.FakeStream
    stock_history: MagicMock
    crypto_history: MagicMock


@pytest.fixture(autouse=True)
def collect_node_cycles() -> Iterator[None]:
    yield
    gc.collect()


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> Rig:
    trading = fakes.fake_trading()
    trading_stream, stock_stream, crypto_stream = (fakes.FakeStream() for _ in range(3))
    stock_history, crypto_history = fakes.fake_history()
    monkeypatch.setattr(factories, "TradingClient", lambda *args, **kwargs: trading)
    monkeypatch.setattr(execution_client, "TradingStream", lambda *args, **kwargs: trading_stream)
    monkeypatch.setattr(data_client, "StockDataStream", lambda *args, **kwargs: stock_stream)
    monkeypatch.setattr(data_client, "CryptoDataStream", lambda *args, **kwargs: crypto_stream)
    monkeypatch.setattr(data_client, "StockHistoricalDataClient", lambda *args: stock_history)
    monkeypatch.setattr(data_client, "CryptoHistoricalDataClient", lambda *args: crypto_history)
    return Rig(trading, trading_stream, stock_stream, crypto_stream, stock_history, crypto_history)


class Recorder(Strategy):
    """Collects what the engine tells a strategy so tests can assert on it."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.reasons: list[str] = []

    def on_order_event(self, event: Any) -> None:
        self.events.append(type(event).__name__)

    def on_order_rejected(self, event: Any) -> None:
        self.reasons.append(event.reason)

    def on_order_denied(self, event: Any) -> None:
        self.reasons.append(event.reason)

    def on_order_cancel_rejected(self, event: Any) -> None:
        self.reasons.append(event.reason)


class QuoteThenOrder(Recorder):
    """Submits one order on the first quote so the risk engine has a price to check."""

    order_type = "market"

    def __init__(self) -> None:
        super().__init__()
        self.order: Any = None
        self.fills: list[Any] = []

    def on_start(self) -> None:
        self.subscribe_quotes(BTC)

    def on_quote(self, quote: QuoteTick) -> None:
        if self.order is None:
            self.order = self.make_order()
            self.submit_order(self.order)

    def make_order(self) -> Any:
        quantity = Quantity.from_str("0.001")
        price = Price.from_str("70000.000000000")
        if self.order_type == "limit":
            return self.order_factory.limit(BTC, OrderSide.BUY, quantity, price, TimeInForce.GTC)
        if self.order_type == "if_touched":
            return self.order_factory.limit_if_touched(BTC, OrderSide.BUY, quantity, price, price)
        return self.order_factory.market(BTC, OrderSide.BUY, quantity, TimeInForce.GTC)

    def on_order_filled(self, event: Any) -> None:
        self.fills.append(event)


def build_node(strategy: Strategy, *, load: str = str(BTC)) -> LiveNode:
    provider = InstrumentProviderConfig(load_ids=[load])
    node = (
        LiveNode.builder("ALPACA-TEST", TraderId("TEST-001"), Environment.LIVE)
        .with_delay_post_stop_secs(0)
        .with_timeout_connection(5)
        .with_timeout_reconciliation(5)
        .with_timeout_portfolio(5)
        .with_timeout_disconnection_secs(2)
        .add_data_client(
            ALPACA,
            AlpacaDataClientFactory,
            AlpacaDataClientConfig(api_key="k", api_secret="s", instrument_provider=provider),
        )
        .add_exec_client(
            ALPACA,
            AlpacaExecutionClientFactory,
            AlpacaExecClientConfig(
                api_key="k",
                api_secret="s",
                crypto_maker_fee=Decimal("0.0015"),
                crypto_taker_fee=Decimal("0.0025"),
                instrument_provider=provider,
            ),
        )
        .build()
    )
    node.add_strategy(strategy)
    return node


def run_scripted(node: LiveNode, script: Callable[[], None] | None = None) -> None:
    """Run the node while a script thread plays Alpaca, then stop it and re-raise script errors."""
    errors: list[BaseException] = []
    handle = node.handle()

    def play() -> None:
        try:
            if script is not None:
                script()
                threading.Event().wait(0.4)
                handle.stop()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            handle.stop()

    thread = threading.Thread(target=play)
    thread.start()

    async def run() -> None:
        async with asyncio.timeout(NODE_TIMEOUT_SECONDS):
            await node.run_async()

    try:
        asyncio.run(run())
    finally:
        thread.join()
    if errors:
        raise errors[0]


def order_of(node: LiveNode, strategy: "QuoteThenOrder") -> Any:
    order = node.cache.order(strategy.order.client_order_id)
    assert order is not None
    return order


def wait_until(condition: Callable[[], bool]) -> None:
    for _ in range(int(fakes.WAIT_SECONDS * 50)):
        if condition():
            return
        threading.Event().wait(0.02)
    raise TimeoutError("Condition never became true")


def submitted_request(rig: Rig) -> Any:
    wait_until(lambda: rig.trading.submit_order.call_args is not None)
    return rig.trading.submit_order.call_args.args[0]


def venue_order(request: Any, **overrides: Any) -> Any:
    market = type(request).__name__ == "MarketOrderRequest"
    fields: dict[str, Any] = {
        "client_order_id": request.client_order_id,
        "symbol": request.symbol,
        "qty": str(request.qty),
        "order_type": "market" if market else "limit",
        "type": "market" if market else "limit",
        "limit_price": None if market else str(request.limit_price),
        **overrides,
    }
    return support.order(**fields)


def push_update(rig: Rig, request: Any, event: str, **overrides: Any) -> None:
    order_fields = overrides.pop("order", {})
    update = support.trade_update(event, order=venue_order(request, **order_fields), **overrides)
    rig.trading_stream.push("updates", update)


def test_a_streamed_quote_reaches_the_strategy_with_the_instruments_precision(rig: Rig) -> None:
    class QuoteStrategy(Recorder):
        quote: QuoteTick | None = None

        def on_start(self) -> None:
            self.subscribe_quotes(BTC)

        def on_quote(self, quote: QuoteTick) -> None:
            self.quote = quote
            self.shutdown_system("quote received")

    strategy = QuoteStrategy()
    node = build_node(strategy)
    run_scripted(node, lambda: rig.crypto_stream.push("quotes", support.quote()))

    assert strategy.quote is not None
    assert strategy.quote.instrument_id == BTC
    assert strategy.quote.bid_price.as_decimal() == Decimal("81000.1")
    assert strategy.quote.ask_size.as_decimal() == Decimal("2.25")
    assert rig.crypto_stream.symbols["quotes"] == ("BTC/USD",)
    assert node.cache.instrument(BTC) is not None
    assert node.cache.account(PAPER_ACCOUNT) is not None
    assert rig.crypto_stream.stops == 1
    assert rig.trading_stream.stops == 1
    assert rig.stock_stream.handlers == {}


def test_a_coin_book_is_kept_by_nautilus_and_read_at_an_interval(rig: Rig) -> None:
    from nautilus_trader.model import BookType

    class BookStrategy(Recorder):
        best: tuple[Decimal, Decimal] | None = None

        def on_start(self) -> None:
            self.subscribe_book_at_interval(BTC, BookType.L2_MBP, 100, depth=10)

        def on_book(self, book: Any) -> None:
            if book.best_bid_price() is None or book.best_ask_price() is None:
                return
            self.best = (book.best_bid_price().as_decimal(), book.best_ask_price().as_decimal())
            self.shutdown_system("book received")

    def push() -> None:
        snapshot = support.book([(81000.1, 1.5), (80999.0, 2.0)], [(81001.9, 0.75)], reset=True)
        rig.crypto_stream.push("orderbooks", snapshot)
        update = support.book([(81000.1, 0.0)], [], reset=False)
        rig.crypto_stream.push("orderbooks", update)

    strategy = BookStrategy()
    node = build_node(strategy)
    run_scripted(node, push)

    assert strategy.best == (Decimal("80999.0"), Decimal("81001.9"))
    assert rig.crypto_stream.symbols["orderbooks"] == ("BTC/USD",)


def test_a_trade_and_a_bar_are_streamed_to_the_strategy(rig: Rig) -> None:
    class DataStrategy(Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.trade: Any = None
            self.bar: Any = None
            self.bar_type = BarType.from_str(f"{BTC}-1-MINUTE-LAST-EXTERNAL")

        def on_start(self) -> None:
            self.subscribe_trades(BTC)
            self.subscribe_bars(self.bar_type)

        def on_trade(self, trade: Any) -> None:
            self.trade = trade

        def on_bar(self, bar: Any) -> None:
            self.bar = bar

    strategy = DataStrategy()
    node = build_node(strategy)

    def script() -> None:
        rig.crypto_stream.push("trades", support.trade())
        rig.crypto_stream.push("bars", support.bar())

    run_scripted(node, script)

    assert str(strategy.trade.trade_id) == "42"
    assert strategy.trade.price.as_decimal() == Decimal("81000.5")
    assert strategy.bar.bar_type == strategy.bar_type
    assert strategy.bar.close.as_decimal() == Decimal("81000.5")


class RelayWriter:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def market(self, events: list[dict[str, Any]]) -> None:
        self.events.extend(events)

    def traded(self, symbol: str) -> bool:
        return any(e["type"] == "trade" and e["symbol"] == symbol for e in self.events)


def test_a_symbol_asked_for_is_loaded_streamed_and_released_when_its_lease_lapses(
    rig: Rig,
) -> None:
    now = [0.0]
    leases = Leases(["BTC/USD"], ttl=30.0, clock=lambda: now[0])
    writer = RelayWriter()
    node = build_node(Recorder())
    node.add_actor(MarketRelay(cast(StateWriter, writer), [BTC], leases))

    def script() -> None:
        leases.renew(["AAPL"])
        rig.stock_stream.push("trades", support.trade("AAPL", 10))
        wait_until(
            lambda: any(e["type"] == "trade" and e["symbol"] == "AAPL" for e in writer.events)
        )
        assert rig.stock_stream.symbols["quotes"] == ("AAPL",)
        now[0] = 31.0
        wait_until(lambda: ("AAPL",) in rig.stock_stream.unsubscribed)

    run_scripted(node, script)


def test_a_symbol_alpaca_does_not_know_is_skipped_and_the_node_carries_on(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(market_relay, "LOAD_SECONDS", 0.5)
    known = rig.trading.get_asset.side_effect
    rig.trading.get_asset.side_effect = lambda symbol: (
        known(symbol) if symbol != "NOPE" else (_ for _ in ()).throw(support.api_error(404))
    )
    leases = Leases(["BTC/USD"])
    writer = RelayWriter()
    relay = MarketRelay(cast(StateWriter, writer), [BTC], leases)
    node = build_node(Recorder())
    node.add_actor(relay)

    def script() -> None:
        leases.renew(["NOPE"])
        wait_until(lambda: "NOPE" in relay.failed)
        leases.renew(["AAPL"])
        rig.stock_stream.push("trades", support.trade("AAPL", 10))
        wait_until(
            lambda: any(e["type"] == "trade" and e["symbol"] == "AAPL" for e in writer.events)
        )

    run_scripted(node, script)
    assert "NOPE" not in relay.streaming


def test_bars_missed_across_a_gap_are_backfilled_before_the_next_live_one(rig: Rig) -> None:
    from datetime import timedelta

    from alpaca.data.models import Bar as AlpacaBar

    def minute(at: int) -> AlpacaBar:
        stamp = (support.NOW + timedelta(minutes=at)).isoformat()
        return AlpacaBar("BTC/USD", {**support.RAW_BAR, "t": stamp, "c": 81000 + at})

    missed = [
        {**support.RAW_BAR, "t": (support.NOW + timedelta(minutes=at)).isoformat(), "c": 81000 + at}
        for at in (1, 2, 3)
    ]
    rig.crypto_history.get_crypto_bars.return_value = BarSet({"BTC/USD": missed})

    class Bars(Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.closes: list[float] = []

        def on_start(self) -> None:
            self.subscribe_bars(BarType.from_str(f"{BTC}-1-MINUTE-LAST-EXTERNAL"))

        def on_bar(self, bar: Any) -> None:
            self.closes.append(float(bar.close))

    strategy = Bars()
    node = build_node(strategy)

    def script() -> None:
        rig.crypto_stream.push("bars", minute(0))
        threading.Event().wait(0.2)
        rig.crypto_stream.push("bars", minute(4))

    run_scripted(node, script)

    assert strategy.closes == [81000, 81001, 81002, 81003, 81004]
    request = rig.crypto_history.get_crypto_bars.call_args.args[0]
    assert request.start == support.NOW.replace(tzinfo=None) + timedelta(minutes=1)


def test_only_one_minute_bars_can_be_streamed(rig: Rig) -> None:
    class FiveMinuteStrategy(Recorder):
        def on_start(self) -> None:
            self.subscribe_bars(BarType.from_str(f"{BTC}-5-MINUTE-LAST-EXTERNAL"))
            self.subscribe_quotes(BTC)

    def script() -> None:
        rig.crypto_stream.wait_for("quotes")

    node = build_node(FiveMinuteStrategy())
    run_scripted(node, script)
    assert "bars" not in rig.crypto_stream.handlers


def test_historical_bars_are_fetched_from_alpaca(rig: Rig) -> None:
    raw = {"BTC/USD": [support.RAW_BAR, {**support.RAW_BAR, "t": "2026-10-28T12:01:00Z"}]}
    rig.crypto_history.get_crypto_bars.return_value = BarSet(raw)

    class HistoryStrategy(Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.bars: list[Any] = []

        def on_start(self) -> None:
            bar_type = BarType.from_str(f"{BTC}-1-MINUTE-LAST-EXTERNAL")
            self.request_bars(bar_type, start=support.NOW)

        def on_historical_bars(self, bars: Any) -> None:
            self.bars = list(bars)
            self.shutdown_system("bars received")

    strategy = HistoryStrategy()
    node = build_node(strategy)
    run_scripted(node)

    request = rig.crypto_history.get_crypto_bars.call_args.args[0]
    assert request.symbol_or_symbols == "BTC/USD"
    assert (request.timeframe.amount, request.timeframe.unit.value) == (1, "Min")
    assert [bar.close.as_decimal() for bar in strategy.bars] == [Decimal("81000.5")] * 2


def test_a_market_order_is_sent_to_alpaca_and_its_fills_reach_the_strategy(rig: Rig) -> None:
    strategy = QuoteThenOrder()
    node = build_node(strategy)

    def script() -> None:
        rig.crypto_stream.push("quotes", support.quote())
        request = submitted_request(rig)
        push_update(rig, request, "new")
        first = {"qty": 0.0004, "price": 81000.0, "execution_id": support.FILL_ID_ONE}
        push_update(rig, request, "partial_fill", **first)
        filled = {"status": "filled", "filled_qty": "0.001", "filled_avg_price": "81006"}
        second = {"qty": 0.0006, "price": 81010.0, "execution_id": support.FILL_ID_TWO}
        push_update(rig, request, "fill", order=filled, **second)

    run_scripted(node, script)

    request = rig.trading.submit_order.call_args.args[0]
    assert (request.symbol, float(request.qty)) == ("BTC/USD", 0.001)
    assert request.time_in_force.value == "gtc"
    assert strategy.events[:3] == ["OrderInitialized", "OrderSubmitted", "OrderAccepted"]
    assert strategy.events.count("OrderFilled") == 2
    assert order_of(node, strategy).status == OrderStatus.FILLED
    fees = [fill.commission.as_decimal() for fill in strategy.fills]
    assert fees == [Decimal("0.000001"), Decimal("0.0000015")]
    assert {fill.commission.currency.code for fill in strategy.fills} == {"BTC"}
    assert [str(fill.trade_id) for fill in strategy.fills] == [
        str(support.FILL_ID_ONE),
        str(support.FILL_ID_TWO),
    ]


def test_a_fill_refreshes_the_account_from_alpaca(rig: Rig) -> None:
    strategy = QuoteThenOrder()
    node = build_node(strategy)

    def script() -> None:
        rig.crypto_stream.push("quotes", support.quote())
        request = submitted_request(rig)
        push_update(rig, request, "new")
        calls_before = rig.trading.get_account.call_count
        fill = {"qty": 0.001, "price": 81000.0, "execution_id": support.FILL_ID_ONE}
        push_update(rig, request, "fill", order={"status": "filled"}, **fill)
        wait_until(lambda: rig.trading.get_account.call_count > calls_before)

    run_scripted(node, script)
    assert rig.trading.get_account.call_count >= 2


def test_a_rejected_order_reports_alpacas_reason(rig: Rig) -> None:
    rig.trading.submit_order.side_effect = support.api_error(403, "insufficient buying power")
    strategy = QuoteThenOrder()
    node = build_node(strategy)
    run_scripted(node, lambda: rig.crypto_stream.push("quotes", support.quote()))

    assert strategy.events == ["OrderInitialized", "OrderSubmitted", "OrderRejected"]
    assert strategy.reasons == ["insufficient buying power"]
    assert order_of(node, strategy).status == OrderStatus.REJECTED


@pytest.mark.parametrize(
    "failure", [support.api_error(503), requests.ConnectionError("reset")], ids=["5xx", "network"]
)
def test_an_unknown_outcome_is_never_turned_into_a_rejection(rig: Rig, failure: Exception) -> None:
    rig.trading.submit_order.side_effect = failure
    strategy = QuoteThenOrder()
    node = build_node(strategy)

    def script() -> None:
        rig.crypto_stream.push("quotes", support.quote())
        wait_until(lambda: rig.trading.submit_order.call_count == 1)

    run_scripted(node, script)

    assert "OrderRejected" not in strategy.events
    assert order_of(node, strategy).status == OrderStatus.SUBMITTED


def test_unsupported_order_types_are_denied_without_calling_alpaca(rig: Rig) -> None:
    strategy = IfTouchedBuy()
    node = build_node(strategy)
    run_scripted(node, lambda: rig.crypto_stream.push("quotes", support.quote()))

    assert strategy.events[-1] == "OrderDenied"
    assert "order type" in strategy.reasons[0]
    rig.trading.submit_order.assert_not_called()


class LimitBuy(QuoteThenOrder):
    order_type = "limit"


class IfTouchedBuy(QuoteThenOrder):
    order_type = "if_touched"


class CancelOnAccept(LimitBuy):
    def on_order_accepted(self, event: Any) -> None:
        self.cancel_order(self.order.client_order_id)


def test_cancelling_an_order_uses_the_venue_order_id_and_ends_canceled(rig: Rig) -> None:
    strategy = CancelOnAccept()
    node = build_node(strategy)

    def script() -> None:
        rig.crypto_stream.push("quotes", support.quote())
        request = submitted_request(rig)
        push_update(rig, request, "new")
        wait_until(lambda: rig.trading.cancel_order_by_id.call_args is not None)
        push_update(rig, request, "canceled", order={"status": "canceled"})

    run_scripted(node, script)

    rig.trading.cancel_order_by_id.assert_called_once_with(str(support.ORDER_ID))
    assert strategy.events[-1] == "OrderCanceled"
    assert order_of(node, strategy).status == OrderStatus.CANCELED


def test_a_refused_cancel_is_reported_to_the_strategy(rig: Rig) -> None:
    rig.trading.cancel_order_by_id.side_effect = support.api_error(422, "order not cancelable")
    strategy = CancelOnAccept()
    node = build_node(strategy)

    def script() -> None:
        rig.crypto_stream.push("quotes", support.quote())
        push_update(rig, submitted_request(rig), "new")
        wait_until(lambda: rig.trading.cancel_order_by_id.call_args is not None)

    run_scripted(node, script)

    assert "OrderCancelRejected" in strategy.events
    assert strategy.reasons == ["order not cancelable"]


def test_cancel_all_cancels_the_strategys_working_orders_one_by_one(rig: Rig) -> None:
    class CancelAllOnAccept(LimitBuy):
        def on_order_accepted(self, event: Any) -> None:
            self.cancel_all_orders(BTC)

    strategy = CancelAllOnAccept()
    node = build_node(strategy)

    def script() -> None:
        rig.crypto_stream.push("quotes", support.quote())
        request = submitted_request(rig)
        push_update(rig, request, "new")
        wait_until(lambda: rig.trading.cancel_order_by_id.call_args is not None)
        push_update(rig, request, "canceled", order={"status": "canceled"})

    run_scripted(node, script)

    rig.trading.cancel_order_by_id.assert_called_once_with(str(support.ORDER_ID))
    assert strategy.events[-1] == "OrderCanceled"


def test_modifying_a_limit_order_replaces_it_at_alpaca(rig: Rig) -> None:
    class ModifyOnAccept(LimitBuy):
        def on_order_accepted(self, event: Any) -> None:
            self.modify_order(self.order.client_order_id, price=Price.from_str("69000.000000000"))

    strategy = ModifyOnAccept()
    node = build_node(strategy)

    def script() -> None:
        rig.crypto_stream.push("quotes", support.quote())
        request = submitted_request(rig)
        push_update(rig, request, "new")
        wait_until(lambda: rig.trading.replace_order_by_id.call_args is not None)
        replaced = {"limit_price": "69000"}
        push_update(rig, request, "replaced", order=replaced)

    run_scripted(node, script)

    order_id, replace = rig.trading.replace_order_by_id.call_args.args
    assert order_id == str(support.ORDER_ID)
    assert replace.limit_price == 69000
    assert strategy.events[-1] == "OrderUpdated"
    assert order_of(node, strategy).price == Price.from_str("69000")


def test_updates_for_orders_nautilus_did_not_place_are_ignored(rig: Rig) -> None:
    strategy = LimitBuy()
    node = build_node(strategy)

    def script() -> None:
        rig.crypto_stream.push("quotes", support.quote())
        request = submitted_request(rig)
        foreign = support.trade_update("new", order=support.order(client_order_id="from-dashboard"))
        rig.trading_stream.push("updates", foreign)
        push_update(rig, request, "new")
        wait_until(lambda: strategy.events.count("OrderAccepted") == 1)

    run_scripted(node, script)
    assert strategy.events.count("OrderAccepted") == 1


def test_updates_with_unknown_events_are_ignored(rig: Rig) -> None:
    strategy = LimitBuy()
    node = build_node(strategy)

    def script() -> None:
        rig.crypto_stream.push("quotes", support.quote())
        request = submitted_request(rig)
        push_update(rig, request, "something_new")
        push_update(rig, request, "pending_new")
        push_update(rig, request, "new")
        wait_until(lambda: "OrderAccepted" in strategy.events)

    run_scripted(node, script)
    assert strategy.events.count("OrderAccepted") == 1


def test_startup_reconciliation_adopts_alpacas_open_orders_and_positions(rig: Rig) -> None:
    working = support.order(client_order_id="external-1", status="new")
    rig.trading.get_orders.return_value = [working]
    rig.trading.get_all_positions.return_value = [support.position()]

    class Idle(Recorder):
        pass

    node = build_node(Idle())
    run_scripted(node, lambda: None)

    adopted = {str(order.venue_order_id): order for order in node.cache.orders()}
    assert adopted[str(support.ORDER_ID)].status == OrderStatus.ACCEPTED
    assert node.cache.positions_open()[0].quantity.as_decimal() == Decimal("0.25")


def test_the_factories_refuse_foreign_configs() -> None:
    from nautilus_trader.config import DataClientConfig, ExecutionClientConfig

    with pytest.raises(TypeError, match="AlpacaDataClientConfig"):
        AlpacaDataClientFactory.create(
            name="x", config=DataClientConfig(), cache=MagicMock(), clock=MagicMock()
        )
    with pytest.raises(TypeError, match="AlpacaExecClientConfig"):
        AlpacaExecutionClientFactory.create(
            name="x",
            config=ExecutionClientConfig(),
            cache=MagicMock(),
            clock=MagicMock(),
            trader_id=TraderId("T-1"),
        )
