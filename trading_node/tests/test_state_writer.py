from datetime import UTC, datetime
from decimal import Decimal

import sqlalchemy as sa

from app.core.tables import (
    node_fills,
    node_orders,
    node_positions,
    node_reconciliation,
    node_status,
)
from trading_node.state_writer import StateWriter

NOW = datetime(2026, 11, 7, 15, tzinfo=UTC)


def order(status: str, filled: str) -> dict[str, object]:
    return {
        "client_order_id": "CMD-1",
        "venue_order_id": "v-1",
        "command_id": None,
        "primary_id": None,
        "exec_algorithm": None,
        "strategy_id": "CommandStrategy-CMD",
        "instrument_id": "SPUS.ALPACA",
        "side": "buy",
        "order_type": "market",
        "time_in_force": "day",
        "quantity": Decimal(2),
        "filled_qty": Decimal(filled),
        "avg_px": None,
        "price": None,
        "trigger_price": None,
        "status": status,
        "is_open": status != "filled",
        "is_closed": status == "filled",
        "created": NOW,
        "updated": NOW,
        "last_event": "Order",
        "reason": None,
    }


def test_orders_are_upserted_fills_kept_once_and_positions_replaced(
    database: sa.engine.Engine,
) -> None:
    writer = StateWriter(database)
    writer.start()
    writer.order(order("accepted", "0"))
    writer.order(order("filled", "2"))
    fill = {
        "trade_id": "t-1",
        "client_order_id": "CMD-1",
        "venue_order_id": "v-1",
        "instrument_id": "SPUS.ALPACA",
        "side": "buy",
        "quantity": Decimal(2),
        "price": Decimal("55.5"),
        "commission": None,
        "commission_currency": None,
        "liquidity_side": "taker",
        "time": NOW,
    }
    writer.fill(fill)
    writer.fill(fill)
    held = {"side": "long", "avg_px_open": Decimal("55.5"), "realized_pnl": None, "updated": NOW}
    writer.positions([{"instrument_id": "SPUS.ALPACA", "quantity": Decimal(2), **held}])
    writer.positions([{"instrument_id": "HLAL.ALPACA", "quantity": Decimal(1), **held}])
    writer.status("heartbeat", {"node": "n"})
    writer.reconciliation("policy_resolution", "critical", "not found", client_order_id="CMD-1")
    writer.flush()
    writer.stop()
    with database.connect() as connection:
        [row] = connection.execute(sa.select(node_orders)).all()
        assert (row.status, row.filled_qty, row.is_closed) == ("filled", Decimal(2), True)
        assert (
            connection.execute(sa.select(sa.func.count()).select_from(node_fills)).scalar_one() == 1
        )
        assert [p.instrument_id for p in connection.execute(sa.select(node_positions))] == [
            "HLAL.ALPACA"
        ]
        assert connection.execute(sa.select(node_status.c.value)).scalar_one() == {"node": "n"}
        assert (
            connection.execute(sa.select(node_reconciliation.c.severity)).scalar_one() == "critical"
        )
