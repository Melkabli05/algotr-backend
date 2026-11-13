import asyncio
import json
import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import ANY, MagicMock
from uuid import UUID

import pytest
from alpaca.data.enums import DataFeed
from alpaca.data.models import Orderbook
from alpaca.data.models import Quote as AlpacaQuote
from alpaca.data.models import Trade as AlpacaTrade
from alpaca.trading.enums import OrderSide, OrderStatus, TradeEvent
from alpaca.trading.models import TradeUpdate
from fastapi.testclient import TestClient

from app.core.settings import Settings
from app.features.market import hub as hub_module
from app.features.market import streams as service
from app.features.market.hub import Hub
from app.features.market.schemas import OrderUpdate, Quote, Trade
from app.features.market.watch import Watchlist

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
QUOTE = Quote(symbol="BTC/USD", bid_price=1.0, bid_size=2.0, ask_price=3.0, ask_size=4.0, time=NOW)
TRADE = Trade(symbol="BTC/USD", price=2.0, size=0.5, time=NOW)
UPDATE = OrderUpdate(
    event="fill",
    order_id="abc",
    symbol="BTC/USD",
    side="buy",
    status="filled",
    qty=1,
    filled_qty=1,
    filled_avg_price=2.0,
    time=NOW,
)


def drain(queue: asyncio.Queue[str]) -> list[dict[str, Any]]:
    messages = []
    while not queue.empty():
        messages.append(json.loads(queue.get_nowait()))
    return messages


def test_hub_broadcasts_to_every_subscriber() -> None:
    hub = Hub()
    first, second = hub.subscribe(), hub.subscribe()
    hub.publish(QUOTE)
    assert [m["type"] for m in drain(first)] == ["quote"]
    assert [m["type"] for m in drain(second)] == ["quote"]


def test_hub_replays_latest_quote_and_trade_but_not_order_updates() -> None:
    hub = Hub()
    hub.publish(QUOTE)
    hub.publish(Quote(**{**QUOTE.model_dump(), "bid_price": 9.0}))
    hub.publish(TRADE)
    hub.publish(UPDATE)
    replayed = drain(hub.subscribe())
    assert [(m["type"], m.get("bid_price")) for m in replayed] == [("quote", 9.0), ("trade", None)]


def test_hub_unsubscribed_clients_get_nothing() -> None:
    hub = Hub()
    queue = hub.subscribe()
    hub.unsubscribe(queue)
    hub.publish(UPDATE)
    assert queue.empty()


def test_hub_drops_events_for_a_full_queue_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hub_module, "QUEUE_SIZE", 1)
    hub = Hub()
    queue = hub.subscribe()
    hub.publish(UPDATE)
    hub.publish(UPDATE)
    assert queue.qsize() == 1


def test_websocket_streams_replayed_and_new_messages(client: TestClient, hub: Hub) -> None:
    hub.publish(QUOTE)
    with client.websocket_connect("/stream") as websocket:
        assert websocket.receive_json()["symbol"] == "BTC/USD"


def test_websocket_disconnect_unsubscribes(client: TestClient, hub: Hub) -> None:
    with client.websocket_connect("/stream"):
        pass
    assert hub.subscribers == 0


def make_update(fill: dict[str, Any] | None = None, **order: Any) -> TradeUpdate:
    details = {
        "execution_id": None,
        "price": None,
        "qty": None,
        "position_qty": None,
        **(fill or {}),
    }
    return cast(
        TradeUpdate,
        SimpleNamespace(
            event=TradeEvent.FILL, timestamp=NOW, order=SimpleNamespace(**order), **details
        ),
    )


def test_to_quote_and_trade() -> None:
    quote = AlpacaQuote(
        "BTC/USD",
        {"t": NOW.isoformat(), "bp": 1.5, "bs": 2, "ap": 2.5, "as": 3, "bx": "x", "ax": "x"},
    )
    trade = AlpacaTrade("BTC/USD", {"t": NOW.isoformat(), "p": 2.0, "s": 0.5, "i": 1, "x": "x"})
    assert service.to_quote(quote).model_dump(include={"bid_price", "ask_price"}) == {
        "bid_price": 1.5,
        "ask_price": 2.5,
    }
    assert service.to_trade(trade).price == 2.0


def test_to_order_update_maps_enums_to_plain_strings() -> None:
    update = make_update(
        id="abc",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        status=OrderStatus.FILLED,
        qty="1",
        filled_qty="1",
        filled_avg_price="2",
    )
    converted = service.to_order_update(update)
    assert (converted.event, converted.side, converted.status) == ("fill", "buy", "filled")
    assert (converted.order_id, converted.filled_qty, converted.filled_avg_price) == ("abc", 1, 2)


def test_to_order_update_defaults_missing_fill_quantity() -> None:
    update = make_update(
        id="abc",
        symbol="AAPL",
        side="buy",
        status="new",
        qty=None,
        filled_qty=None,
        filled_avg_price=None,
    )
    converted = service.to_order_update(update)
    assert (converted.filled_qty, converted.qty) == (0, None)


def test_a_fill_carries_its_own_price_quantity_and_the_position_after_it() -> None:
    execution = UUID("5b9a0c52-0000-4000-8000-000000000001")
    update = make_update(
        {"execution_id": execution, "price": 12.5, "qty": 3.0, "position_qty": 10.0},
        id="abc",
        symbol="F",
        side="buy",
        status="partially_filled",
        qty="10",
        filled_qty="3",
        filled_avg_price="12.5",
    )
    converted = service.to_order_update(update)
    assert converted.execution_id == str(execution)
    assert (converted.price, converted.fill_qty, converted.position_qty) == (12.5, 3.0, 10.0)


def test_an_update_without_a_fill_has_no_fill_details() -> None:
    converted = service.to_order_update(
        make_update(
            id="a",
            symbol="F",
            side="buy",
            status="new",
            qty="1",
            filled_qty="0",
            filled_avg_price=None,
        )
    )
    assert (converted.execution_id, converted.price, converted.fill_qty) == (None, None, None)


def test_a_listener_hears_every_event_and_a_failing_one_does_not_stop_the_rest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hub = Hub()
    heard: list[str] = []

    def broken(event: Any) -> None:
        raise RuntimeError("boom")

    hub.listen(broken)
    hub.listen(lambda event: heard.append(event.type))
    hub.publish(QUOTE)
    hub.publish(UPDATE)
    assert heard == ["quote", "order_update"]
    assert "A stream listener failed" in caplog.text


def settings_for(*symbols: str, feed: DataFeed = DataFeed.IEX) -> Settings:
    return Settings(
        alpaca_api_key="key",
        alpaca_api_secret="secret",
        stream_symbols=list(symbols),
        data_feed=feed,
    )


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    made = SimpleNamespace(
        trading=MagicMock(), stock=MagicMock(), crypto=MagicMock(), thread=MagicMock()
    )
    monkeypatch.setattr(service, "TradingStream", MagicMock(return_value=made.trading))
    monkeypatch.setattr(service, "StockDataStream", MagicMock(return_value=made.stock))
    monkeypatch.setattr(service, "CryptoDataStream", MagicMock(return_value=made.crypto))
    monkeypatch.setattr(threading, "Thread", made.thread)
    return made


def test_start_routes_symbols_to_the_right_streams(sdk: SimpleNamespace) -> None:
    service.MarketStreams(Hub(), settings_for("SPY", "BTC/USD", "AAPL"), MagicMock()).start()

    sdk.trading.subscribe_trade_updates.assert_called_once_with(ANY)
    sdk.stock.subscribe_quotes.assert_called_once_with(ANY, "SPY", "AAPL")
    sdk.stock.subscribe_trades.assert_called_once_with(ANY, "SPY", "AAPL")
    sdk.crypto.subscribe_quotes.assert_called_once_with(ANY, "BTC/USD")
    assert sdk.thread.call_count == 3


@pytest.mark.parametrize("feed", [DataFeed.IEX, DataFeed.SIP])
def test_the_stock_stream_reads_the_chosen_feed(
    monkeypatch: pytest.MonkeyPatch, sdk: SimpleNamespace, feed: DataFeed
) -> None:
    stock_class = MagicMock()
    monkeypatch.setattr(service, "StockDataStream", stock_class)
    service.MarketStreams(Hub(), settings_for("SPY", feed=feed), MagicMock()).start()
    assert stock_class.call_args.kwargs["feed"] is feed


def test_start_skips_data_streams_without_symbols(
    monkeypatch: pytest.MonkeyPatch, sdk: SimpleNamespace
) -> None:
    stock_class, crypto_class = MagicMock(), MagicMock()
    monkeypatch.setattr(service, "StockDataStream", stock_class)
    monkeypatch.setattr(service, "CryptoDataStream", crypto_class)
    service.MarketStreams(Hub(), settings_for("SPY"), MagicMock()).start()
    stock_class.assert_called_once()
    crypto_class.assert_not_called()


def test_with_no_symbols_only_the_account_stream_runs(
    monkeypatch: pytest.MonkeyPatch, sdk: SimpleNamespace
) -> None:
    stock_class, crypto_class = MagicMock(), MagicMock()
    monkeypatch.setattr(service, "StockDataStream", stock_class)
    monkeypatch.setattr(service, "CryptoDataStream", crypto_class)
    service.MarketStreams(Hub(), settings_for(), MagicMock()).start()
    stock_class.assert_not_called()
    crypto_class.assert_not_called()
    assert sdk.thread.call_count == 1


def test_stream_handlers_publish_on_the_app_loop(sdk: SimpleNamespace) -> None:
    loop, hub = MagicMock(), Hub()
    service.MarketStreams(hub, settings_for("SPY"), loop).start()
    on_quote = sdk.stock.subscribe_quotes.call_args.args[0]
    quote = AlpacaQuote(
        "SPY", {"t": NOW.isoformat(), "bp": 1, "bs": 1, "ap": 2, "as": 1, "bx": "x", "ax": "x"}
    )
    asyncio.run(on_quote(quote))
    loop.call_soon_threadsafe.assert_called_once_with(hub.publish, ANY)


def test_pausing_data_stops_only_the_market_data_streams(sdk: SimpleNamespace) -> None:
    streams = service.MarketStreams(Hub(), settings_for("SPY", "BTC/USD"), MagicMock())
    streams.start()
    streams.pause_data()
    sdk.stock.stop.assert_called_once()
    sdk.crypto.stop.assert_called_once()
    sdk.trading.stop.assert_not_called()


def test_resuming_data_starts_fresh_streams_once(sdk: SimpleNamespace) -> None:
    streams = service.MarketStreams(Hub(), settings_for("BTC/USD"), MagicMock())
    streams.start()
    streams.resume_data()
    assert sdk.crypto.subscribe_quotes.call_count == 1
    streams.pause_data()
    streams.resume_data()
    streams.resume_data()
    assert sdk.crypto.subscribe_quotes.call_count == 2


def test_pausing_twice_is_harmless(sdk: SimpleNamespace) -> None:
    streams = service.MarketStreams(Hub(), settings_for("BTC/USD"), MagicMock())
    streams.start()
    streams.pause_data()
    streams.pause_data()
    sdk.crypto.stop.assert_called_once()


def test_stop_stops_everything_and_survives_a_stream_that_never_started(
    sdk: SimpleNamespace,
) -> None:
    sdk.crypto.stop.side_effect = AttributeError("'NoneType' object has no attribute 'is_running'")
    streams = service.MarketStreams(Hub(), settings_for("SPY", "BTC/USD"), MagicMock())
    streams.start()
    streams.stop()
    sdk.stock.stop.assert_called_once()
    sdk.trading.stop.assert_called_once()


def test_stream_symbols_endpoint_lists_the_watchlist(client: TestClient) -> None:
    assert client.get("/stream/symbols").json() == ["SPY", "AAPL", "MSFT", "BTC/USD", "ETH/USD"]


def test_the_crypto_stream_subscribes_to_books_when_something_listens(
    sdk: SimpleNamespace,
) -> None:
    service.MarketStreams(
        Hub(), settings_for("SPY", "BTC/USD", "ETH/USD"), MagicMock(), on_book=MagicMock()
    ).start()
    sdk.crypto.subscribe_orderbooks.assert_called_once_with(ANY, "BTC/USD", "ETH/USD")
    sdk.stock.subscribe_orderbooks.assert_not_called()


def test_without_a_listener_no_book_is_subscribed(sdk: SimpleNamespace) -> None:
    service.MarketStreams(Hub(), settings_for("BTC/USD"), MagicMock()).start()
    sdk.crypto.subscribe_orderbooks.assert_not_called()


def test_a_book_message_reaches_the_listener_and_a_raw_one_does_not(
    sdk: SimpleNamespace,
) -> None:
    heard = MagicMock()
    streams = service.MarketStreams(Hub(), settings_for("BTC/USD"), MagicMock(), on_book=heard)
    streams.start()
    handler = sdk.crypto.subscribe_orderbooks.call_args.args[0]
    book = Orderbook("BTC/USD", {"t": "2026-11-03T16:00:00Z", "b": [], "a": [], "r": True})
    asyncio.run(handler(book))
    asyncio.run(handler({"S": "BTC/USD"}))
    heard.assert_called_once_with(book)


def test_renewing_a_book_unsubscribes_now_and_subscribes_again_after_a_moment(
    sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    timers: list[MagicMock] = []

    def make_timer(delay: float, call: Any, args: tuple[Any, ...]) -> MagicMock:
        timers.append(MagicMock(delay=delay, call=call, args=args))
        return timers[-1]

    monkeypatch.setattr(threading, "Timer", make_timer)
    streams = service.MarketStreams(
        Hub(), settings_for("BTC/USD"), MagicMock(), on_book=MagicMock()
    )
    streams.start()
    sdk.crypto.subscribe_orderbooks.reset_mock()
    streams.renew_book("BTC/USD")
    sdk.crypto.unsubscribe_orderbooks.assert_called_once_with("BTC/USD")
    sdk.crypto.subscribe_orderbooks.assert_not_called()
    [timer] = timers
    assert timer.delay == service.RESUBSCRIBE_DELAY and timer.daemon
    timer.call(*timer.args)
    sdk.crypto.subscribe_orderbooks.assert_called_once_with(ANY, "BTC/USD")


def test_renewing_does_nothing_when_the_data_streams_are_paused(sdk: SimpleNamespace) -> None:
    streams = service.MarketStreams(
        Hub(), settings_for("BTC/USD"), MagicMock(), on_book=MagicMock()
    )
    streams.start()
    streams.pause_data()
    streams.renew_book("BTC/USD")
    sdk.crypto.unsubscribe_orderbooks.assert_not_called()


def test_a_client_asking_to_watch_symbols_is_passed_on_and_released_on_disconnect(
    client: TestClient, watchlist: Watchlist
) -> None:
    seen: list[list[str]] = []
    watchlist.apply.append(seen.append)
    with client.websocket_connect("/stream") as websocket:
        websocket.send_json({"action": "watch", "symbols": ["spus", "SPY", "SOL/USD", "no way"]})
        websocket.send_text("not json")
        websocket.send_json({"action": "watch", "symbols": ["SPUS", "SOL/USD"]})
        websocket.send_json({"nothing": True})
    assert ["SPUS", "SOL/USD"] in seen
    assert seen[-1] == []


def test_the_watchlist_keeps_what_every_client_shows_up_to_its_limits() -> None:
    watched = Watchlist(["SPY"], per_client=2, total=3)
    watched.set("a", ["AAPL", "SPY", "TSLA", "NVDA"])
    watched.set("b", ["TSLA", "AMD", "COIN"])
    assert watched.wanted() == ["AAPL", "TSLA", "AMD"]
    watched.drop("a")
    assert watched.wanted() == ["TSLA", "AMD"]


def test_watching_subscribes_new_symbols_and_unsubscribes_dropped_ones(
    sdk: SimpleNamespace,
) -> None:
    streams = service.MarketStreams(Hub(), settings_for("SPY", "BTC/USD"), MagicMock())
    streams.start()
    streams.watch(["SPUS", "SOL/USD"])
    sdk.stock.subscribe_trades.assert_called_with(ANY, "SPUS")
    sdk.crypto.subscribe_quotes.assert_called_with(ANY, "SOL/USD")
    streams.watch(["SPUS"])
    sdk.crypto.unsubscribe_trades.assert_called_once_with("SOL/USD")
    sdk.stock.unsubscribe_trades.assert_not_called()


def test_symbols_watched_before_a_pause_are_streamed_again_on_resume(sdk: SimpleNamespace) -> None:
    streams = service.MarketStreams(Hub(), settings_for("SPY"), MagicMock())
    streams.start()
    streams.pause_data()
    streams.watch(["SPUS"])
    streams.resume_data()
    sdk.stock.subscribe_trades.assert_called_with(ANY, "SPY", "SPUS")


def test_the_watchlist_can_change_while_it_is_being_handed_on() -> None:
    watched = Watchlist(["SPY"])
    watched.set("a", ["AAPL"])
    seen: list[list[str]] = []

    def apply(symbols: list[str]) -> None:
        seen.append(symbols)
        watched.set("b", ["TSLA"])
        watched.drop("a")

    watched.apply.append(apply)
    watched.refresh()
    watched.refresh()
    assert seen == [["AAPL"], ["TSLA"]]


def test_hand_overs_never_overlap() -> None:
    watched = Watchlist([])
    active: list[int] = []
    overlaps: list[bool] = []
    gate = threading.Event()

    def apply(symbols: list[str]) -> None:
        if active:
            overlaps.append(True)
        active.append(1)
        gate.wait(0.05)
        active.pop()

    watched.apply.append(apply)
    threads = [threading.Thread(target=watched.refresh) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert overlaps == []
