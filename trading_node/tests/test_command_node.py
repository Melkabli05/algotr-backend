from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import sqlalchemy as sa
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.common import LoggerConfig, LogLevel
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import AccountType, Bar, Money, OmsType, Quantity, TraderId

from app.bridge import commands as bridge
from app.core.tables import compliance, node_orders, node_status
from trading_node.alpaca_adapter.constants import ALPACA_VENUE
from trading_node.alpaca_adapter.parsing import USD
from trading_node.exec_algorithms.iceberg import IcebergAlgorithm
from trading_node.exec_algorithms.scheduled import ScheduledAlgorithm
from trading_node.state_writer import StateWriter
from trading_node.strategies.commands import CommandStrategy
from trading_node.strategies.plan import daily_bar_type
from trading_node.tests.support import AAPL, BTC, instrument_for

SPUS = instrument_for("SPUS")
START = pd.Timestamp("2024-01-02 15:00", tz="UTC")
PRICES = {SPUS: 50.0, AAPL: 180.0, BTC: 60_000.0}


def bars(instrument: Any, price: float, minutes: int = 3) -> list[Bar]:
    """A bar a minute, so the node's timers run for a few minutes of market time."""
    found = []
    for minute in range(minutes):
        nanos = (START + pd.Timedelta(minutes=minute)).value
        value = instrument.make_price(price)
        found.append(
            Bar(
                bar_type=daily_bar_type(instrument.id),
                open=value,
                high=value,
                low=value,
                close=value,
                volume=Quantity(1e7, instrument.size_precision),
                ts_event=nanos,
                ts_init=nanos,
            )
        )
    return found


def run_node(database: sa.engine.Engine, between: Callable[[], None] | None = None) -> None:
    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-005"), logging=LoggerConfig(stdout_level=LogLevel.ERROR)
        )
    )
    engine.add_venue(
        venue=ALPACA_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(100_000, USD)],
        fee_model=MakerTakerFeeModel(maker_rate=Decimal(0), taker_rate=Decimal(0)),
    )
    for instrument, price in PRICES.items():
        engine.add_instrument(instrument)
        engine.add_data(bars(instrument, price))
    writer = StateWriter(database)
    writer.start()
    engine.add_strategy(CommandStrategy(writer, database, "test-node"))
    engine.add_exec_algorithm(ScheduledAlgorithm())
    engine.add_exec_algorithm(IcebergAlgorithm())
    if between is None:
        engine.run()
    else:
        middle = (START + pd.Timedelta(minutes=1)).value
        engine.run(end=middle)
        writer.flush()
        between()
        engine.run(start=middle)
    writer.flush()
    writer.stop()
    engine.dispose()


def allow(database: sa.engine.Engine, symbol: str) -> None:
    with database.begin() as connection:
        connection.execute(
            sa.insert(compliance).values(
                symbol=symbol, allowed=True, reason="passes", as_of=datetime.now(UTC)
            )
        )


def command(database: sa.engine.Engine, kind: Any, payload: dict[str, Any]) -> bridge.Command:
    with database.begin() as connection:
        return bridge.submit(connection, kind, payload)


def reread(database: sa.engine.Engine, command_id: Any) -> bridge.Command:
    with database.connect() as connection:
        found = bridge.get(connection, command_id)
    assert found is not None
    return found


def test_a_submit_command_is_placed_filled_and_written_back(database: sa.engine.Engine) -> None:
    allow(database, "SPUS")
    placed = command(database, "submit", {"symbol": "SPUS", "side": "buy", "qty": 10})

    run_node(database)

    finished = reread(database, placed.id)
    assert finished.status == "done"
    assert finished.result == {"status": "filled", "filled_qty": 10.0, "avg_px": 50.0}
    with database.connect() as connection:
        [row] = connection.execute(sa.select(node_orders)).all()
    assert (row.client_order_id, row.status) == (placed.client_order_id, "filled")


def test_a_buy_without_a_shariah_verdict_is_denied_before_nautilus(
    database: sa.engine.Engine,
) -> None:
    placed = command(database, "submit", {"symbol": "AAPL", "side": "buy", "qty": 1})

    run_node(database)

    finished = reread(database, placed.id)
    assert finished.status == "denied"
    assert finished.error == "AAPL has no Shariah verdict on record; the screen fails closed"


def test_a_rebalance_sells_first_then_buys_and_reports_each_leg(
    database: sa.engine.Engine,
) -> None:
    allow(database, "SPUS")
    orders = [
        {"symbol": "SPUS", "side": "buy", "qty": 100},
        {"symbol": "BTC/USD", "side": "buy", "qty": 0.5},
        {"symbol": "AAPL", "side": "buy", "qty": 1},
        {"symbol": "SPUS", "side": "buy", "qty": 0.4},
        {"symbol": "BTC/USD", "side": "sell", "qty": 0.1},
    ]
    placed = command(database, "rebalance", {"orders": orders, "cancel": ["SPUS"]})

    run_node(database)

    finished = reread(database, placed.id)
    assert finished.status == "done"
    assert finished.result is not None
    legs = {(leg["symbol"], leg["side"], leg["qty"]): leg for leg in finished.result["orders"]}
    assert legs[("SPUS", "buy", 100.0)]["status"] == "filled"
    assert legs[("SPUS", "buy", 100.0)]["order_id"] == f"{placed.client_order_id}-B1"
    assert legs[("BTC/USD", "buy", 0.5)]["status"] == "filled"
    assert legs[("AAPL", "buy", 1.0)]["status"].startswith("denied: AAPL has no Shariah verdict")
    assert legs[("SPUS", "buy", 0.4)]["status"] == "skipped: below the smallest tradable size"
    sell = legs[("BTC/USD", "sell", 0.1)]
    assert sell["order_id"] == f"{placed.client_order_id}-S1"
    with database.connect() as connection:
        rows = connection.execute(sa.select(node_orders.c.client_order_id, node_orders.c.created))
        created = {row.client_order_id: row.created for row in rows}
    assert created[f"{placed.client_order_id}-S1"] <= created[f"{placed.client_order_id}-B1"]


def test_the_node_says_it_is_running_once_its_strategies_start(
    database: sa.engine.Engine,
) -> None:
    run_node(database)

    with database.connect() as connection:
        row = connection.execute(sa.select(node_status).where(node_status.c.name == "node")).one()
    assert row.value == {"state": "running", "node": "test-node", "mode": "paper"}


def test_a_close_sells_the_whole_position_as_a_one_leg_rebalance(
    database: sa.engine.Engine,
) -> None:
    allow(database, "SPUS")
    command(database, "submit", {"symbol": "SPUS", "side": "buy", "qty": 10})
    closing: list[bridge.Command] = []

    def between() -> None:
        closing.append(command(database, "close_position", {"symbol": "SPUS"}))

    run_node(database, between=between)

    finished = reread(database, closing[0].id)
    assert finished.status == "done"
    assert finished.result is not None
    [leg] = finished.result["orders"]
    assert (leg["symbol"], leg["side"], leg["qty"], leg["status"]) == (
        "SPUS",
        "sell",
        10.0,
        "filled",
    )
