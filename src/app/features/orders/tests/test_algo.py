from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.trading.enums import PositionIntent, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from app.core.book import BookGuard
from app.core.tests.support import api_error
from app.features.orders import algo as module
from app.features.orders.algo import AlgoDesk, AlgoNotFoundError
from app.features.orders.schemas import AlgoRequest
from app.features.orders.service import AlgoRefusedError
from app.features.orders.tests.test_iceberg import Broker
from app.quant import volume
from app.quant.history import NotEnoughHistoryError
from app.quant.volume import COIN, STOCK, Profile

START = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)
CLOCK = {"is_open": True, "next_close": "2026-11-03T16:00:00-05:00"}
BOOK = {"bp": 9.99, "ap": 10.01, "bs": 4, "as": 6}
COIN_BOOK = {"bp": 99990.0, "ap": 100010.0, "bs": 0.5, "as": 0.5}


def at(minutes: float) -> datetime:
    return START + timedelta(minutes=minutes)


def request(**fields: Any) -> AlgoRequest:
    body = {"algo": "twap", "symbol": "F", "side": "buy", "qty": 10, "duration": 10, "slices": 5}
    return AlgoRequest.model_validate({**body, **fields})


def settle(broker: Broker, order_id: str, filled: float, price: float = 10.0) -> None:
    """An immediate-or-cancel order ends: filled in full, or canceled with whatever it got."""
    order = broker.orders[order_id]
    if filled:
        broker.fill(order_id, qty=filled, price=price)
    order["status"] = "filled" if filled >= float(order["qty"]) else "canceled"


@pytest.fixture
def broker() -> Broker:
    found = Broker()
    found.client.get_clock.return_value = CLOCK
    found.client.get_open_position.return_value = {"qty_available": "100"}
    return found


@pytest.fixture
def stocks() -> MagicMock:
    mock: MagicMock = create_autospec(StockHistoricalDataClient, instance=True)
    mock.get_stock_latest_quote.return_value = {"F": BOOK}
    return mock


@pytest.fixture
def crypto() -> MagicMock:
    mock: MagicMock = create_autospec(CryptoHistoricalDataClient, instance=True)
    mock.get_crypto_latest_quote.return_value = {"BTC/USD": COIN_BOOK}
    return mock


@pytest.fixture
def desk(broker: Broker, stocks: MagicMock, crypto: MagicMock, tmp_path: Path) -> AlgoDesk:
    return AlgoDesk(broker.client, stocks, crypto, DataFeed.IEX, tmp_path / "algos.json")


def test_a_plan_lists_equal_slices_at_even_times_with_the_arrival_price(desk: AlgoDesk) -> None:
    plan = desk.plan(request(), START)
    assert [part.qty for part in plan.slices] == [2, 2, 2, 2, 2]
    assert [part.due for part in plan.slices] == [at(m) for m in (0, 2, 4, 6, 8)]
    assert plan.end == at(10)
    assert plan.arrival_price == pytest.approx(10.0)


def test_planning_places_nothing(desk: AlgoDesk, broker: Broker) -> None:
    desk.plan(request(), START)
    assert broker.requests == [] and desk.algos() == []


def test_the_first_slice_goes_out_on_the_first_tick_as_an_ioc_market_order(
    desk: AlgoDesk, broker: Broker
) -> None:
    item = desk.place(request(), START)
    assert broker.requests == []
    desk.tick(START)
    [sent] = broker.requests
    assert isinstance(sent, MarketOrderRequest)
    assert (sent.qty, sent.time_in_force) == (2, TimeInForce.IOC)
    assert sent.client_order_id == f"algo-{item.id}-1"
    assert sent.position_intent == PositionIntent.BUY_TO_OPEN


def test_a_price_cap_makes_the_slices_limit_orders(desk: AlgoDesk, broker: Broker) -> None:
    desk.place(request(limit_price=10.5), START)
    desk.tick(START)
    [sent] = broker.requests
    assert isinstance(sent, LimitOrderRequest) and sent.limit_price == 10.5
    assert sent.time_in_force == TimeInForce.IOC


def test_a_sell_closes_a_position(desk: AlgoDesk, broker: Broker) -> None:
    desk.place(request(side="sell"), START)
    desk.tick(START)
    assert broker.requests[0].position_intent == PositionIntent.SELL_TO_CLOSE


def test_nothing_goes_out_until_the_next_interval(desk: AlgoDesk, broker: Broker) -> None:
    desk.place(request(), START)
    desk.tick(START)
    settle(broker, "order-1", 2)
    desk.tick(at(1))
    assert len(broker.requests) == 1
    desk.tick(at(2))
    assert [sent.qty for sent in broker.requests] == [2, 2]


def test_a_tick_with_nothing_running_does_not_call_alpaca(desk: AlgoDesk, broker: Broker) -> None:
    desk.tick(START)
    assert broker.client.mock_calls == []


def test_a_slice_that_fills_short_is_made_up_by_the_slices_after_it(
    desk: AlgoDesk, broker: Broker
) -> None:
    desk.place(request(qty=12), START)
    desk.tick(START)
    settle(broker, "order-1", 0)
    desk.tick(at(2))
    assert [sent.qty for sent in broker.requests] == [2, 3]


def test_the_last_slice_takes_whatever_is_left(desk: AlgoDesk, broker: Broker) -> None:
    desk.place(request(slices=2), START)
    desk.tick(START)
    settle(broker, "order-1", 2)
    desk.tick(at(5))
    assert [sent.qty for sent in broker.requests] == [5, 8]


def test_a_wake_after_missed_intervals_spreads_the_rest_over_the_ones_left(
    desk: AlgoDesk, broker: Broker
) -> None:
    desk.place(request(), START)
    desk.tick(START)
    settle(broker, "order-1", 2)
    desk.tick(at(7))
    assert [sent.qty for sent in broker.requests] == [2, 4]
    settle(broker, "order-2", 4)
    desk.tick(at(7.5))
    assert len(broker.requests) == 2
    desk.tick(at(8))
    assert [sent.qty for sent in broker.requests] == [2, 4, 4]


def test_the_order_is_done_when_everything_has_filled(desk: AlgoDesk, broker: Broker) -> None:
    desk.place(request(), START)
    for minute in (0, 2, 4, 6, 8):
        desk.tick(at(minute))
        settle(broker, f"order-{len(broker.requests)}", broker.requests[-1].qty)
    desk.tick(at(8.1))
    [item] = desk.algos()
    assert (item.status, item.filled_qty) == ("done", 10)
    assert item.average_price == pytest.approx(10.0)


def test_the_end_passes_with_something_left_and_nothing_is_swept_late(
    desk: AlgoDesk, broker: Broker
) -> None:
    desk.place(request(), START)
    desk.tick(START)
    settle(broker, "order-1", 0)
    desk.tick(at(11))
    [item] = desk.algos()
    assert (item.status, item.detail) == ("ended", "10 of 10 not filled")
    assert len(broker.requests) == 1


def test_after_the_last_slice_the_order_ends_without_waiting_for_the_end_time(
    desk: AlgoDesk, broker: Broker
) -> None:
    desk.place(request(slices=2), START)
    desk.tick(START)
    settle(broker, "order-1", 5)
    desk.tick(at(5))
    settle(broker, "order-2", 2)
    desk.tick(at(5.1))
    [item] = desk.algos()
    assert (item.status, item.detail) == ("ended", "3 of 10 not filled")


def test_a_child_still_open_when_its_interval_is_over_is_canceled_once(
    desk: AlgoDesk, broker: Broker
) -> None:
    broker.client.cancel_order_by_id.side_effect = lambda order_id: broker.orders[order_id].update(
        status="pending_cancel"
    )
    desk.place(request(qty=12), START)
    desk.tick(START)
    desk.tick(at(1))
    broker.client.cancel_order_by_id.assert_not_called()
    desk.tick(at(2))
    desk.tick(at(2.1))
    broker.client.cancel_order_by_id.assert_called_once_with("order-1")
    assert len(broker.requests) == 1
    settle(broker, "order-1", 0)
    desk.tick(at(2.2))
    assert [sent.qty for sent in broker.requests] == [2, 3]


def test_a_busy_broker_is_retried_on_the_next_tick_under_the_same_id(
    desk: AlgoDesk, broker: Broker
) -> None:
    item = desk.place(request(), START)
    real = broker.client.submit_order.side_effect
    broker.client.submit_order.side_effect = api_error("busy", 503)
    desk.tick(START)
    assert desk.algos()[0].children == [] and desk.algos()[0].status == "working"
    broker.client.submit_order.side_effect = real
    desk.tick(at(0.05))
    assert [sent.client_order_id for sent in broker.requests] == [f"algo-{item.id}-1"]


def test_a_refusal_ends_the_order_as_failed_with_alpacas_reason(
    desk: AlgoDesk, broker: Broker
) -> None:
    desk.place(request(), START)
    broker.client.submit_order.side_effect = api_error("insufficient buying power", 403)
    desk.tick(START)
    [item] = desk.algos()
    assert (item.status, item.detail) == ("failed", "insufficient buying power")


def test_a_child_already_sent_under_its_id_is_read_back_not_doubled(
    desk: AlgoDesk, broker: Broker
) -> None:
    item = desk.place(request(), START)
    broker.orders["order-9"] = {
        "id": "order-9",
        "symbol": "F",
        "side": "buy",
        "type": "market",
        "status": "filled",
        "qty": "2.0",
        "filled_qty": "2.0",
        "filled_avg_price": "10.0",
        "submitted_at": None,
    }
    broker.by_client[f"algo-{item.id}-1"] = "order-9"
    desk.tick(START)
    [child] = desk.algos()[0].children
    assert (child.order, child.filled_qty) == ("order-9", 2.0)


def test_canceling_stops_the_order_and_cancels_the_child_on_show(
    desk: AlgoDesk, broker: Broker
) -> None:
    item = desk.place(request(), START)
    desk.tick(START)
    stopped = desk.cancel(item.id)
    assert (stopped.status, stopped.detail) == ("canceled", "Canceled")
    broker.client.cancel_order_by_id.assert_called_once_with("order-1")
    desk.tick(at(2))
    assert len(broker.requests) == 1


def test_a_fill_that_lands_after_a_cancel_is_still_counted(desk: AlgoDesk, broker: Broker) -> None:
    broker.client.cancel_order_by_id.side_effect = lambda order_id: broker.orders[order_id].update(
        status="pending_cancel"
    )
    item = desk.place(request(), START)
    desk.tick(START)
    desk.cancel(item.id)
    broker.fill("order-1", qty=1)
    broker.orders["order-1"]["status"] = "canceled"
    desk.tick(at(1))
    assert desk.algos()[0].filled_qty == 1


def test_an_unknown_order_is_not_found(desk: AlgoDesk) -> None:
    with pytest.raises(AlgoNotFoundError):
        desk.cancel("nope")


def test_a_restarted_desk_carries_on_where_it_was(
    desk: AlgoDesk, broker: Broker, stocks: MagicMock, crypto: MagicMock, tmp_path: Path
) -> None:
    desk.place(request(), START)
    desk.tick(START)
    settle(broker, "order-1", 2)
    again = AlgoDesk(broker.client, stocks, crypto, DataFeed.IEX, tmp_path / "algos.json")
    again.tick(at(2))
    assert again.algos()[0].children[0].filled_qty == 2
    assert [sent.qty for sent in broker.requests] == [2, 2]


@pytest.mark.parametrize(
    ("sizes", "side", "first"),
    [
        ({"bs": 900, "as": 100}, "buy", 3),
        ({"bs": 100, "as": 900}, "buy", 1),
        ({"bs": 100, "as": 900}, "sell", 3),
    ],
)
def test_the_slice_leans_with_the_quote_sizes_when_asked(
    desk: AlgoDesk,
    broker: Broker,
    stocks: MagicMock,
    sizes: dict[str, float],
    side: str,
    first: int,
) -> None:
    stocks.get_stock_latest_quote.return_value = {"F": {**BOOK, **sizes}}
    desk.place(request(side=side, tilt=0.5), START)
    desk.tick(START)
    assert broker.requests[0].qty == first


def test_without_a_tilt_the_quote_is_read_once_for_the_arrival_price(
    desk: AlgoDesk, stocks: MagicMock
) -> None:
    desk.place(request(), START)
    desk.tick(START)
    assert stocks.get_stock_latest_quote.call_count == 1


def test_a_sell_needs_a_position_to_close(desk: AlgoDesk, broker: Broker) -> None:
    broker.client.get_open_position.side_effect = api_error("position does not exist", 404)
    with pytest.raises(AlgoRefusedError, match="You hold no F to sell"):
        desk.plan(request(side="sell"), START)


def test_a_sell_larger_than_what_is_held_is_refused_with_the_amount(
    desk: AlgoDesk, broker: Broker
) -> None:
    broker.client.get_open_position.return_value = {"qty_available": "4"}
    with pytest.raises(AlgoRefusedError, match="You can sell 4 F; the order is for 10"):
        desk.plan(request(side="sell"), START)


def test_a_stock_order_needs_the_market_open(desk: AlgoDesk, broker: Broker) -> None:
    broker.client.get_clock.return_value = {**CLOCK, "is_open": False}
    with pytest.raises(AlgoRefusedError, match="market is closed"):
        desk.plan(request(), START)


def test_a_stock_order_must_end_before_the_close(desk: AlgoDesk) -> None:
    five_to_close = datetime(2026, 11, 3, 20, 55, tzinfo=UTC)
    with pytest.raises(AlgoRefusedError, match="closes at 16:00 ET; use at most 5 minutes"):
        desk.plan(request(), five_to_close)


def test_a_coin_order_slices_fractionally_and_does_not_look_at_the_clock(
    desk: AlgoDesk, broker: Broker
) -> None:
    desk.place(request(symbol="BTC/USD", qty=0.2, slices=4, duration=8), START)
    desk.tick(START)
    [sent] = broker.requests
    assert (sent.qty, sent.time_in_force, sent.position_intent) == (0.05, TimeInForce.IOC, None)
    broker.client.get_clock.assert_not_called()


def test_a_coin_slice_must_cost_ten_dollars(desk: AlgoDesk) -> None:
    with pytest.raises(AlgoRefusedError, match="use at most 5 slices"):
        desk.plan(request(symbol="BTC/USD", qty=0.0005, slices=8, duration=8), START)
    with pytest.raises(AlgoRefusedError, match="too little to slice"):
        desk.plan(request(symbol="BTC/USD", qty=0.0001, slices=4, duration=8), START)


def test_a_coin_slice_pushed_below_the_minimum_by_the_tilt_is_carried(
    desk: AlgoDesk, broker: Broker, crypto: MagicMock
) -> None:
    crypto.get_crypto_latest_quote.return_value = {"BTC/USD": {**COIN_BOOK, "bs": 0.1, "as": 0.9}}
    desk.place(request(symbol="BTC/USD", qty=0.0005, slices=4, duration=8, tilt=0.3), START)
    desk.tick(START)
    assert broker.requests == [] and desk.algos()[0].next_interval == 1
    desk.tick(at(2))
    assert broker.requests[0].qty == pytest.approx(0.000126667, rel=1e-3)


@pytest.mark.parametrize(("side", "expected"), [("buy", 100.0), ("sell", -100.0)])
def test_slippage_is_the_average_fill_against_the_arrival_in_basis_points(
    desk: AlgoDesk, broker: Broker, side: str, expected: float
) -> None:
    desk.place(request(side=side), START)
    desk.tick(START)
    settle(broker, "order-1", 2, price=10.1)
    desk.tick(at(0.5))
    [item] = desk.algos()
    assert item.average_price == pytest.approx(10.1)
    assert item.slippage_bps == pytest.approx(expected)


def test_orders_are_listed_newest_first(desk: AlgoDesk) -> None:
    first = desk.place(request(), START)
    second = desk.place(request(), at(1))
    assert [item.id for item in desk.algos()] == [second.id, first.id]


def test_only_the_latest_finished_orders_are_kept(
    desk: AlgoDesk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "KEPT_FINISHED", 2)
    ids = [desk.place(request(), at(n)).id for n in range(4)]
    for algo_id in ids:
        desk.cancel(algo_id)
    assert [item.id for item in desk.algos()] == [ids[3], ids[2]]


@pytest.fixture
def profile(monkeypatch: pytest.MonkeyPatch) -> Profile:
    shares = (*([0.0] * 6), 0.5, 0.3, 0.2, *([0.0] * 69))
    found = Profile(shares=shares, sessions=20, session=STOCK)
    monkeypatch.setattr(volume, "profile", lambda *args: found)
    return found


def vwap(**fields: Any) -> AlgoRequest:
    return request(**{"algo": "vwap", "slices": None, "duration": 15, "qty": 100, **fields})


def test_a_vwap_plan_follows_the_profile(desk: AlgoDesk, profile: Profile) -> None:
    plan = desk.plan(vwap(), START)
    assert [part.qty for part in plan.slices] == [50, 30, 20]
    assert [part.due for part in plan.slices] == [at(0), at(5), at(10)]
    assert plan.end == at(15) and plan.notes == {"sessions": 20}


def test_a_vwap_starting_inside_a_bucket_counts_its_part_and_lists_only_real_slices(
    desk: AlgoDesk, profile: Profile
) -> None:
    plan = desk.plan(vwap(), at(3))
    assert [part.qty for part in plan.slices] == [29, 43, 28]


def test_a_vwap_sends_the_slices_of_its_profile(
    desk: AlgoDesk, broker: Broker, profile: Profile
) -> None:
    desk.place(vwap(), START)
    for minute in (0, 5, 10):
        desk.tick(at(minute))
        settle(broker, f"order-{len(broker.requests)}", broker.requests[-1].qty)
    assert [sent.qty for sent in broker.requests] == [50, 30, 20]
    desk.tick(at(10.1))
    assert desk.algos()[0].status == "done"


def test_a_vwap_without_enough_history_is_refused_with_why(
    desk: AlgoDesk, monkeypatch: pytest.MonkeyPatch
) -> None:
    def short(*args: Any) -> Profile:
        raise NotEnoughHistoryError("Alpaca has 3 full sessions of F; a profile needs 5")

    monkeypatch.setattr(volume, "profile", short)
    with pytest.raises(AlgoRefusedError, match="3 full sessions of F"):
        desk.plan(vwap(), START)


def test_a_coin_vwap_carries_slices_under_the_minimum_until_they_add_up(
    desk: AlgoDesk, broker: Broker, monkeypatch: pytest.MonkeyPatch
) -> None:
    flat = Profile(shares=(1 / 288,) * 288, sessions=20, session=COIN)
    monkeypatch.setattr(volume, "profile", lambda *args: flat)
    desk.place(vwap(symbol="BTC/USD", qty=0.00012, duration=20), START)
    for minute in (0, 5, 10):
        desk.tick(at(minute))
    assert broker.requests == []
    desk.tick(at(15))
    assert [sent.qty for sent in broker.requests] == [0.00012]


def test_a_coin_vwap_that_costs_under_ten_dollars_is_refused(
    desk: AlgoDesk, monkeypatch: pytest.MonkeyPatch
) -> None:
    flat = Profile(shares=(1 / 288,) * 288, sessions=20, session=COIN)
    monkeypatch.setattr(volume, "profile", lambda *args: flat)
    with pytest.raises(AlgoRefusedError, match="too little to slice"):
        desk.plan(vwap(symbol="BTC/USD", qty=0.00005, duration=20), START)


@pytest.fixture
def liquid(monkeypatch: pytest.MonkeyPatch) -> Profile:
    found = Profile(shares=(1 / 78,) * 78, sessions=20, session=STOCK, volume=5e7, step=0.35)
    monkeypatch.setattr(volume, "profile", lambda *args: found)
    monkeypatch.setattr(volume, "spread", lambda *args: 0.05)
    return found


def shortfall(**fields: Any) -> AlgoRequest:
    body = {"algo": "shortfall", "qty": 1000, "duration": 30, "slices": 6}
    return request(**{**body, **fields})


def first_half(desk: AlgoDesk, **fields: Any) -> float:
    plan = desk.plan(shortfall(**fields), START)
    return sum(part.qty for part in plan.slices[:3]) / 1000


def test_more_risk_aversion_puts_more_of_a_shortfall_in_the_first_half(
    desk: AlgoDesk, liquid: Profile
) -> None:
    assert first_half(desk, risk_aversion=0.0) == pytest.approx(0.5, abs=0.002)
    assert 0.52 < first_half(desk, risk_aversion=1.0) < 0.56
    assert first_half(desk, risk_aversion=5.0) > 0.6


def test_a_shortfall_plan_says_what_it_was_calibrated_on(desk: AlgoDesk, liquid: Profile) -> None:
    plan = desk.plan(shortfall(), START)
    assert plan.notes["sessions"] == 20 and plan.notes["spread"] == 0.05
    assert plan.notes["volume"] == 5e7
    assert plan.notes["volatility"] == pytest.approx(0.35 * 78**0.5)
    assert 0.5 < float(plan.notes["urgency"]) < 1.5
    assert plan.end == at(30)


def test_the_slices_of_a_shortfall_shrink_and_add_up(desk: AlgoDesk, liquid: Profile) -> None:
    sizes = [part.qty for part in desk.plan(shortfall(risk_aversion=5.0), START).slices]
    assert sizes == sorted(sizes, reverse=True) and sum(sizes) == 1000


def test_a_shortfall_runs_its_slices_and_remembers_its_risk_aversion(
    desk: AlgoDesk, broker: Broker, liquid: Profile
) -> None:
    item = desk.place(shortfall(risk_aversion=5.0), START)
    assert item.risk_aversion == 5.0
    desk.tick(START)
    assert broker.requests[0].qty == desk.plan(shortfall(risk_aversion=5.0), START).slices[0].qty


def test_a_shortfall_needs_the_quotes_to_size_the_spread(
    desk: AlgoDesk, liquid: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(volume, "spread", lambda *args: None)
    with pytest.raises(AlgoRefusedError, match="No recent quotes of F"):
        desk.plan(shortfall(), START)


def test_a_shortfall_needs_a_stock_that_moves(
    desk: AlgoDesk, monkeypatch: pytest.MonkeyPatch
) -> None:
    still = Profile(shares=(1 / 78,) * 78, sessions=20, session=STOCK, volume=5e7, step=0.0)
    monkeypatch.setattr(volume, "profile", lambda *args: still)
    monkeypatch.setattr(volume, "spread", lambda *args: 0.05)
    with pytest.raises(AlgoRefusedError, match="traded too little"):
        desk.plan(shortfall(), START)


@pytest.fixture
def book() -> MagicMock:
    mock: MagicMock = create_autospec(BookGuard, instance=True)
    mock.max_size.return_value = None
    mock.refusal.return_value = None
    return mock


@pytest.fixture
def guarded(
    broker: Broker, stocks: MagicMock, crypto: MagicMock, tmp_path: Path, book: MagicMock
) -> AlgoDesk:
    return AlgoDesk(
        broker.client, stocks, crypto, DataFeed.IEX, tmp_path / "algos.json", guard=book
    )


def coin(**fields: Any) -> AlgoRequest:
    return request(symbol="BTC/USD", qty=0.2, slices=4, duration=8, **fields)


def test_a_coin_slice_is_cut_to_what_the_book_takes_and_the_rest_carries_on(
    guarded: AlgoDesk, broker: Broker, book: MagicMock
) -> None:
    book.max_size.return_value = 0.03
    guarded.place(coin(), START)
    guarded.tick(START)
    [sent] = broker.requests
    assert sent.qty == 0.03
    book.max_size.assert_called_with("BTC/USD", "buy")
    book.refusal.assert_called_with("BTC/USD", "buy", 0.03)
    settle(broker, "order-1", 0.03)
    book.max_size.return_value = None
    guarded.tick(at(2))
    assert broker.requests[1].qty == pytest.approx(0.17 / 3)


def test_a_slice_the_book_has_room_for_is_not_cut(
    guarded: AlgoDesk, broker: Broker, book: MagicMock
) -> None:
    book.max_size.return_value = 1.0
    guarded.place(coin(), START)
    guarded.tick(START)
    assert broker.requests[0].qty == 0.05


def test_a_coin_slice_is_held_back_while_the_book_says_to_wait(
    guarded: AlgoDesk, broker: Broker, book: MagicMock
) -> None:
    book.refusal.return_value = "The spread of BTC/USD is 14.0 bps, 4.0 times its usual 3.5"
    guarded.place(coin(), START)
    guarded.tick(START)
    [item] = guarded.algos()
    assert broker.requests == [] and item.next_interval == 1 and item.status == "working"
    assert item.detail == (
        "A slice was held back: The spread of BTC/USD is 14.0 bps, 4.0 times its usual 3.5"
    )
    book.refusal.return_value = None
    guarded.tick(at(2))
    assert broker.requests[0].qty == pytest.approx(0.2 / 3)


def test_an_order_whose_last_slice_is_held_back_ends_with_the_rest_unfilled(
    guarded: AlgoDesk, broker: Broker, book: MagicMock
) -> None:
    book.refusal.return_value = "The spread of BTC/USD is wide"
    guarded.place(coin(), START)
    for minute in (0, 2, 4, 6, 8):
        guarded.tick(at(minute))
    [item] = guarded.algos()
    assert broker.requests == [] and item.status == "ended"
    assert item.detail == "0.2 of 0.2 not filled"


def test_a_slice_cut_below_ten_dollars_is_carried(
    guarded: AlgoDesk, broker: Broker, book: MagicMock
) -> None:
    book.max_size.return_value = 0.00005
    guarded.place(coin(), START)
    guarded.tick(START)
    assert broker.requests == [] and guarded.algos()[0].next_interval == 1


def test_a_coin_slice_with_a_price_cap_is_not_the_books_to_judge(
    guarded: AlgoDesk, broker: Broker, book: MagicMock
) -> None:
    guarded.place(coin(limit_price=100500.0), START)
    guarded.tick(START)
    assert isinstance(broker.requests[0], LimitOrderRequest)
    book.max_size.assert_not_called()
    book.refusal.assert_not_called()


def test_a_stock_slice_is_not_the_books_to_judge(
    guarded: AlgoDesk, broker: Broker, book: MagicMock
) -> None:
    guarded.place(request(), START)
    guarded.tick(START)
    assert len(broker.requests) == 1
    book.max_size.assert_not_called()
    book.refusal.assert_not_called()


def test_a_plan_says_what_the_book_takes_in_a_slice(guarded: AlgoDesk, book: MagicMock) -> None:
    book.max_size.return_value = 0.03
    assert guarded.plan(coin(), START).notes["book_cap"] == 0.03
    book.max_size.assert_called_with("BTC/USD", "buy")


def test_a_plan_has_no_book_cap_for_a_stock_a_priced_order_or_a_book_with_no_limit(
    guarded: AlgoDesk, desk: AlgoDesk, book: MagicMock
) -> None:
    book.max_size.return_value = 0.03
    assert "book_cap" not in guarded.plan(request(), START).notes
    assert "book_cap" not in guarded.plan(coin(limit_price=100500.0), START).notes
    book.max_size.return_value = None
    assert "book_cap" not in guarded.plan(coin(), START).notes
    assert "book_cap" not in desk.plan(coin(), START).notes
