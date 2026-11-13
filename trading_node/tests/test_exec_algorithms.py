from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Any, Self

from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.common import LoggerConfig, LogLevel
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import (
    AccountType,
    Bar,
    ClientOrderId,
    Money,
    OmsType,
    OrderStatus,
    TraderId,
)
from nautilus_trader.trading import Strategy, StrategyConfig

from trading_node.alpaca_adapter.constants import ALPACA_VENUE
from trading_node.alpaca_adapter.parsing import USD
from trading_node.exec_algorithms.family import (
    CANCELED,
    ENDED,
    STOP_TOPIC,
    children,
    family_outcome,
)
from trading_node.exec_algorithms.iceberg import IcebergAlgorithm
from trading_node.exec_algorithms.scheduled import ScheduledAlgorithm
from trading_node.strategies.commands import build_order
from trading_node.strategies.plan import daily_bar_type
from trading_node.tests.support import bars_of, instrument_for, rows

SPUS = instrument_for("SPUS")
PRICE = 50.0
CLIENT_ID = ClientOrderId("CMD-0123456789abcdef0123456789abcdef")


class Sender(Strategy):
    """Sends one order on the first bar, built from a command payload as the node builds it."""

    def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
        return super().__new__(cls)

    def __init__(self, payload: dict[str, Any], stop_after: timedelta | None) -> None:
        super().__init__(StrategyConfig(order_id_tag="CMD"))
        self.payload = payload
        self.stop_after = stop_after
        self.sent = False

    def on_start(self) -> None:
        self.subscribe_bars(daily_bar_type(SPUS.id))

    def on_bar(self, bar: Bar) -> None:
        if self.sent:
            return
        self.sent = True
        now = self.clock.utc_now()
        algo = dict(self.payload.get("algo") or {})
        if "offsets" in algo:
            offsets = algo.pop("offsets")
            algo["intervals"] = [
                ((now + timedelta(minutes=minute)).isoformat(), 1.0) for minute in offsets
            ]
            end = algo.pop("end_after", offsets[-1] + 1 if offsets else 1)
            algo["end"] = (now + timedelta(minutes=end)).isoformat()
        payload = {**self.payload, "algo": algo}
        self.submit_order(build_order(self.order_factory, payload, SPUS, CLIENT_ID))
        if self.stop_after is not None:
            self.clock.set_time_alert(
                "stop", now + self.stop_after, callback=lambda _: self.stop_algorithm()
            )

    def stop_algorithm(self) -> None:
        self.publish_message(STOP_TOPIC, str(CLIENT_ID))


@dataclass
class Outcome:
    primary: Any
    children: list[Any]
    command: tuple[str, dict[str, Any] | None, str | None]


def run(payload: dict[str, Any], stop_after: timedelta | None = None) -> Outcome:
    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-004"), logging=LoggerConfig(stdout_level=LogLevel.ERROR)
        )
    )
    engine.add_venue(
        venue=ALPACA_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(100_000, USD)],
        fee_model=MakerTakerFeeModel(maker_rate=Decimal(0), taker_rate=Decimal(0)),
    )
    engine.add_instrument(SPUS)
    engine.add_data(bars_of(SPUS, rows(PRICE)))
    strategy = Sender(payload, stop_after)
    engine.add_strategy(strategy)
    engine.add_exec_algorithm(ScheduledAlgorithm())
    engine.add_exec_algorithm(IcebergAlgorithm())
    engine.run()
    primary = engine.cache.order(CLIENT_ID)
    assert primary is not None
    reason = next(
        (str(event.reason) for event in primary.events() if hasattr(event, "reason")), None
    )
    found = Outcome(
        primary,
        children(engine.cache, CLIENT_ID),
        family_outcome(engine.cache, primary, reason),
    )
    engine.dispose()
    return found


def schedule(*offsets: int, end_after: float | None = None, **order: object) -> dict[str, Any]:
    algo: dict[str, Any] = {"name": "schedule", "offsets": list(offsets)}
    if end_after is not None:
        algo["end_after"] = end_after
    return {"symbol": "SPUS", "side": "buy", "qty": 30, **order, "algo": algo}


def iceberg(qty: int, display: int, limit: float) -> dict[str, Any]:
    return {
        "symbol": "SPUS",
        "side": "buy",
        "qty": qty,
        "type": "limit",
        "limit_price": limit,
        "time_in_force": "gtc",
        "algo": {"name": "iceberg", "display_qty": display},
    }


def test_a_twap_sends_equal_slices_and_the_primary_takes_the_last() -> None:
    found = run(schedule(1, 2, 3))

    assert [str(child.filled_qty) for child in found.children] == ["10", "10"]
    assert found.primary.status == OrderStatus.FILLED
    assert str(found.primary.filled_qty) == "10"
    status, result, _ = found.command
    assert status == "done"
    assert result == {"status": "filled", "filled_qty": 30.0, "avg_px": PRICE, "detail": None}


def test_slices_priced_out_of_the_market_end_the_schedule_with_nothing_filled() -> None:
    found = run(schedule(1, 2, 3, type="limit", limit_price=1.0))

    assert all(child.status == OrderStatus.CANCELED for child in found.children)
    assert found.primary.is_closed
    status, result, _ = found.command
    assert status == "done"
    assert result is not None
    assert (result["status"], result["filled_qty"]) == ("ended", 0.0)


def test_a_stop_cancels_the_schedule_and_keeps_what_filled() -> None:
    found = run(schedule(1, 2, 3, 4, 5), stop_after=timedelta(minutes=2, seconds=30))

    assert found.primary.status == OrderStatus.DENIED
    status, result, _ = found.command
    assert status == "done"
    assert result is not None
    assert result["filled_qty"] == 12.0
    assert result["detail"] == CANCELED


def test_the_end_of_the_schedule_leaves_the_rest_unsent() -> None:
    found = run(schedule(1, 2, 3, end_after=2.5))

    assert found.primary.status == OrderStatus.DENIED
    status, result, _ = found.command
    assert status == "done"
    assert result is not None
    assert result["filled_qty"] == 20.0
    assert result["detail"] == f"{ENDED} 10 of 30 not filled"


def test_an_order_with_an_empty_schedule_is_denied() -> None:
    found = run(schedule())

    assert found.primary.status == OrderStatus.DENIED
    assert found.children == []
    assert found.command == ("denied", None, "Bad schedule: the schedule has no intervals")


def test_an_iceberg_shows_one_slice_at_a_time_until_all_is_filled() -> None:
    found = run(iceberg(25, 10, PRICE + 1))

    assert [str(child.quantity) for child in found.children] == ["10", "10"]
    assert found.primary.status == OrderStatus.FILLED
    assert str(found.primary.quantity) == "5"
    status, result, _ = found.command
    assert (status, result and result["filled_qty"]) == ("done", 25.0)


def test_a_stopped_iceberg_cancels_the_slice_on_show() -> None:
    found = run(iceberg(25, 10, 1.0), stop_after=timedelta(minutes=1))

    assert [child.status for child in found.children] == [OrderStatus.CANCELED]
    assert found.primary.status == OrderStatus.DENIED
    status, result, _ = found.command
    assert status == "done"
    assert result is not None
    assert (result["filled_qty"], result["detail"]) == (0.0, CANCELED)
