from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from trading_node.strategies.commands import command_outcome, sweep_decision


def status(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def order(name: str, closed: bool, filled: str = "0", avg: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        status=status(name),
        is_closed=closed,
        filled_qty=SimpleNamespace(
            as_decimal=lambda: Decimal(filled), __bool__=lambda: filled != "0"
        ),
        avg_px=avg,
    )


def test_a_restart_follows_what_alpaca_has_resends_what_was_never_sent_and_flags_the_rest() -> None:
    def decide(kind: str, status: str, found: bool, **payload: object) -> str:
        return sweep_decision({"kind": kind, "status": status, "payload": payload}, found)

    assert decide("submit", "sent", True) == "attach"
    assert decide("submit", "claimed", False) == "requeue"
    assert decide("submit", "sent", False) == "fail"
    assert decide("cancel", "sent", False) == "requeue"
    assert decide("submit", "sent", False, algo={"name": "schedule"}) == "stopped"


def test_a_command_finishes_with_its_order() -> None:
    assert command_outcome(order("accepted", False), None) == ("sent", None, None)
    assert command_outcome(order("denied", True), "too big") == ("denied", None, "too big")
    assert command_outcome(order("rejected", True), "no funds") == ("failed", None, "no funds")
    done, result, error = command_outcome(order("filled", True, "2", 101.5), None)
    assert (done, error) == ("done", None)
    assert result == {"status": "filled", "filled_qty": 2.0, "avg_px": 101.5}


def test_each_command_type_becomes_the_matching_nautilus_order() -> None:
    from nautilus_trader.model import ClientOrderId, OrderType, TimeInForce, TrailingOffsetType

    from trading_node.alpaca_adapter.tests.test_alpaca_parsing import FACTORY, crypto
    from trading_node.strategies.commands import build_order

    base = {"symbol": "BTC/USD", "side": "sell", "qty": 0.5}

    def made(**payload: object) -> Any:
        return build_order(
            FACTORY, {**base, **payload}, crypto(), ClientOrderId(f"CMD-{len(payload)}")
        )

    market = made(type="market")
    assert (market.order_type, market.time_in_force) == (OrderType.MARKET, TimeInForce.GTC)
    stop = made(type="stop", stop_price=45000)
    assert stop.order_type == OrderType.STOP_MARKET and float(stop.trigger_price) == 45000
    capped = made(type="stop_limit", stop_price=45000, limit_price=44000)
    assert (float(capped.trigger_price), float(capped.price)) == (45000, 44000)
    trail = made(type="trailing_stop", trail_percent=1.5)
    assert trail.trailing_offset_type == TrailingOffsetType.BASIS_POINTS
    assert float(trail.trailing_offset) == 150
    opening = made(type="limit", limit_price=50000, time_in_force="opg")
    assert opening.time_in_force == TimeInForce.AT_THE_OPEN
