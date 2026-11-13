import asyncio
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from alpaca.trading.enums import QueryOrderStatus
from nautilus_trader.common import Clock
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.core import dt_to_unix_nanos
from nautilus_trader.model import (
    ClientOrderId,
    InstrumentId,
    OrderSide,
    OrderStatus,
    PositionSide,
    TraderId,
    VenueOrderId,
)

from trading_node.alpaca_adapter.config import AlpacaExecClientConfig
from trading_node.alpaca_adapter.execution_client import AlpacaExecutionClient
from trading_node.alpaca_adapter.providers import AlpacaInstrumentProvider
from trading_node.alpaca_adapter.tests import fakes, support

BTC = InstrumentId.from_str("BTC/USD.ALPACA")
TAKE_LEG = "22222222-2222-2222-2222-222222222222"
STOP_LEG = "33333333-3333-3333-3333-333333333333"


@pytest.fixture
def trading() -> MagicMock:
    return fakes.fake_trading()


@pytest.fixture
def client(trading: MagicMock) -> AlpacaExecutionClient:
    clock = Clock.new_test()
    return AlpacaExecutionClient(
        name="ALPACA",
        config=AlpacaExecClientConfig(api_key="k", api_secret="s"),
        cache=MagicMock(),
        clock=clock,
        trader_id=TraderId("T-1"),
        instrument_provider=AlpacaInstrumentProvider(trading, clock, InstrumentProviderConfig()),
        trading=trading,
    )


def command(**fields: Any) -> Any:
    defaults = {"instrument_id": None, "start": None, "end": None, "venue_order_id": None}
    return SimpleNamespace(**{**defaults, **fields})


def run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def test_the_client_identifies_the_paper_account(client: AlpacaExecutionClient) -> None:
    assert str(client.account_id) == "ALPACA-PAPER-001"
    assert not client.is_connected


def test_cancel_all_cancels_every_open_order_for_the_instrument(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    trading.get_orders.return_value = [
        support.order(id=support.ORDER_ID),
        support.order(id=support.OTHER_ORDER_ID),
    ]
    run(client._cancel_all_orders(command(instrument_id=BTC, order_side=OrderSide.NO_ORDER_SIDE)))
    request = trading.get_orders.call_args.args[0]
    assert (request.status, request.symbols, request.side) == (
        QueryOrderStatus.OPEN,
        ["BTC/USD"],
        None,
    )
    cancelled = [call.args[0] for call in trading.cancel_order_by_id.call_args_list]
    assert cancelled == [support.ORDER_ID, support.OTHER_ORDER_ID]


def test_cancel_all_can_be_limited_to_one_side(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    run(client._cancel_all_orders(command(instrument_id=BTC, order_side=OrderSide.SELL)))
    assert trading.get_orders.call_args.args[0].side.value == "sell"


def test_cancel_all_keeps_going_when_one_cancel_fails(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    trading.get_orders.return_value = [
        support.order(id=support.ORDER_ID),
        support.order(id=support.OTHER_ORDER_ID),
    ]
    trading.cancel_order_by_id.side_effect = [support.api_error(422), None]
    run(client._cancel_all_orders(command(instrument_id=BTC, order_side=OrderSide.NO_ORDER_SIDE)))
    assert trading.cancel_order_by_id.call_count == 2


def test_order_reports_cover_open_orders_only_when_asked(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    trading.get_orders.return_value = [support.order()]
    reports = run(client._generate_order_status_reports(command(open_only=True, instrument_id=BTC)))
    request = trading.get_orders.call_args.args[0]
    assert (request.status, request.symbols) == (QueryOrderStatus.OPEN, ["BTC/USD"])
    assert [report.order_status for report in reports] == [OrderStatus.ACCEPTED]


def test_order_reports_translate_the_time_window(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    start, end = dt_to_unix_nanos(support.NOW), dt_to_unix_nanos(support.NOW) + 10**9
    run(client._generate_order_status_reports(command(open_only=False, start=start, end=end)))
    request = trading.get_orders.call_args.args[0]
    assert request.status == QueryOrderStatus.ALL
    assert request.after == support.NOW
    assert request.symbols is None


def test_orders_nautilus_cannot_represent_or_instruments_it_cannot_load_are_skipped(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    trading.get_asset.side_effect = [support.asset("BTC/USD"), support.api_error(404)]
    trading.get_orders.return_value = [
        support.order(side=None),
        support.order(symbol="NOPE", client_order_id="unknown-symbol"),
        support.order(client_order_id="good"),
    ]
    reports = run(client._generate_order_status_reports(command(open_only=False)))
    assert [str(report.client_order_id) for report in reports] == ["good"]


def filled(**overrides: Any) -> Any:
    fields = {
        "status": "filled",
        "filled_qty": "0.25",
        "filled_avg_price": "81000.5",
        "filled_at": support.NOW,
        **overrides,
    }
    return support.order(**fields)


def test_fill_reports_come_from_closed_orders_with_fills(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    trading.get_orders.return_value = [filled(), support.order(status="canceled")]
    reports = run(client._generate_fill_reports(command(instrument_id=BTC)))
    assert trading.get_orders.call_args.args[0].status == QueryOrderStatus.CLOSED
    assert [report.last_qty.as_decimal() for report in reports] == [Decimal("0.25")]
    assert reports[0].commission.currency.code == "BTC"


def test_fill_reports_can_be_filtered_to_one_order(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    trading.get_orders.return_value = [filled(), filled(id=support.OTHER_ORDER_ID)]
    wanted = VenueOrderId(str(support.OTHER_ORDER_ID))
    reports = run(client._generate_fill_reports(command(venue_order_id=wanted)))
    assert [str(report.venue_order_id) for report in reports] == [str(support.OTHER_ORDER_ID)]


def test_position_reports_map_alpacas_crypto_symbols_back_to_instruments(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    trading.get_all_positions.return_value = [support.position("BTCUSD", side="short", qty="-0.5")]
    [report] = run(client._generate_position_status_reports(command()))
    assert report.instrument_id == BTC
    assert (report.position_side, report.quantity.as_decimal()) == (
        PositionSide.SHORT,
        Decimal("0.5"),
    )


def test_position_reports_can_be_limited_to_one_instrument(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    trading.get_all_positions.return_value = [
        support.position("BTCUSD"),
        support.position("AAPL", qty="10"),
    ]
    reports = run(
        client._generate_position_status_reports(
            command(instrument_id=InstrumentId.from_str("AAPL.ALPACA"))
        )
    )
    assert [str(report.instrument_id) for report in reports] == ["AAPL.ALPACA"]


def test_a_single_order_report_can_be_found_by_either_id(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    trading.get_order_by_id.return_value = support.order()
    trading.get_order_by_client_id.return_value = support.order(client_order_id="by-client")
    by_venue = run(
        client._generate_order_status_report(
            command(client_order_id=None, venue_order_id=VenueOrderId("abc"))
        )
    )
    by_client = run(
        client._generate_order_status_report(
            command(client_order_id=ClientOrderId("by-client"), venue_order_id=None)
        )
    )
    trading.get_order_by_id.assert_called_once_with("abc")
    trading.get_order_by_client_id.assert_called_once_with("by-client")
    assert str(by_venue.client_order_id) == "O-1"
    assert str(by_client.client_order_id) == "by-client"


def test_a_missing_order_yields_no_report(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    trading.get_order_by_client_id.side_effect = support.api_error(404, "order not found")
    missing = command(client_order_id=ClientOrderId("gone"), venue_order_id=None)
    assert run(client._generate_order_status_report(missing)) is None
    nothing = command(client_order_id=None, venue_order_id=None)
    assert run(client._generate_order_status_report(nothing)) is None


def test_a_fractional_share_position_is_ignored_instead_of_failing_reconciliation(
    client: AlpacaExecutionClient, trading: MagicMock
) -> None:
    trading.get_all_positions.return_value = [
        support.position("AAPL", qty="0.5"),
        support.position("BTCUSD"),
    ]
    reports = run(client._generate_position_status_reports(command()))
    assert [str(report.instrument_id) for report in reports] == ["BTC/USD.ALPACA"]


def unknown_setup(
    client: AlpacaExecutionClient, monkeypatch: pytest.MonkeyPatch, answers: list[Any]
) -> dict[str, list[Any]]:
    seen: dict[str, list[Any]] = {"fetched": [], "applied": [], "rejected": []}

    async def no_wait(_: float) -> None:
        return None

    async def fetch(client_order_id: Any, venue_order_id: Any) -> Any:
        seen["fetched"].append(client_order_id)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def report_for(order: Any) -> Any:
        return f"report of {order}"

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    monkeypatch.setattr(client, "_fetch_order", fetch)
    monkeypatch.setattr(client, "_report_for", report_for)
    monkeypatch.setattr(client, "_handle_report", lambda report: seen["applied"].append(report))
    monkeypatch.setattr(
        client,
        "generate_order_rejected",
        lambda order, reason, ts, due_post_only: seen["rejected"].append(reason),
    )
    return seen


def test_an_unknown_submit_is_looked_up_and_adopted_when_alpaca_has_it(
    client: AlpacaExecutionClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from requests import ConnectionError as Unreachable

    seen = unknown_setup(client, monkeypatch, [Unreachable("down"), None, "order"])
    order = SimpleNamespace(client_order_id=ClientOrderId("CMD-1"))
    run(client._resolve_unknown(order))
    assert seen["fetched"] == [ClientOrderId("CMD-1")] * 3
    assert seen["applied"] == ["report of order"] and seen["rejected"] == []


def test_an_unknown_submit_never_found_is_a_policy_resolution_not_a_resend(
    client: AlpacaExecutionClient, trading: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = unknown_setup(client, monkeypatch, [None, None, None, None])
    order = SimpleNamespace(client_order_id=ClientOrderId("CMD-2"))
    run(client._resolve_unknown(order))
    assert len(seen["fetched"]) == 4 and seen["applied"] == []
    [reason] = seen["rejected"]
    assert reason.startswith("Policy resolution:")
    assert trading.submit_order.call_count == 0


def test_alpacas_bracket_legs_are_mapped_to_the_nautilus_exit_orders(
    client: AlpacaExecutionClient,
) -> None:
    from trading_node.alpaca_adapter.parsing import ListRequest

    take = SimpleNamespace(client_order_id=ClientOrderId("TP-1"))
    stop = SimpleNamespace(client_order_id=ClientOrderId("SL-1"))
    found = ListRequest(
        request=MagicMock(), parent=SimpleNamespace(), take_profit=take, stop_loss=stop
    )
    placed = support.order(
        legs=[
            support.order(
                id="22222222-2222-2222-2222-222222222222", order_type="limit", type="limit"
            ),
            support.order(
                id=STOP_LEG,
                order_type="stop",
                type="stop",
                stop_price="45000",
                limit_price=None,
            ),
        ]
    )
    client._remember_legs(placed, found)
    assert client._legs == {TAKE_LEG: ClientOrderId("TP-1"), STOP_LEG: ClientOrderId("SL-1")}
