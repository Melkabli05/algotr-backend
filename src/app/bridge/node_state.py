from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.engine import Connection

from app.core.tables import node_fills, node_orders, node_reconciliation, node_status

HEARTBEAT_STALE = timedelta(seconds=15)


class NodeOrder(BaseModel):
    """One order as the trading node last wrote it; the node is its only writer."""

    client_order_id: str
    venue_order_id: str | None
    command_id: Any | None
    primary_id: str | None
    exec_algorithm: str | None
    strategy_id: str
    instrument_id: str
    side: str
    order_type: str
    time_in_force: str
    quantity: Decimal
    filled_qty: Decimal
    avg_px: Decimal | None
    price: Decimal | None
    trigger_price: Decimal | None
    status: str
    is_open: bool
    is_closed: bool
    created: datetime
    updated: datetime
    last_event: str
    reason: str | None

    @property
    def symbol(self) -> str:
        return self.instrument_id.rsplit(".", 1)[0]

    @property
    def is_primary(self) -> bool:
        return self.exec_algorithm is not None and self.primary_id == self.client_order_id


class NodeFill(BaseModel):
    trade_id: str
    client_order_id: str
    instrument_id: str
    side: str
    quantity: Decimal
    price: Decimal
    time: datetime


class Reconciliation(BaseModel):
    id: int
    time: datetime
    kind: str
    severity: str
    instrument_id: str | None
    client_order_id: str | None
    detail: str


class NodeHealth(BaseModel):
    """Whether the trading node is running, from its last heartbeat and its own state."""

    up: bool
    state: str
    mode: str | None
    heartbeat: datetime | None
    open_orders: int | None
    waiting: int | None


def orders(
    connection: Connection,
    limit: int = 50,
    *,
    algorithm: str | None = None,
    open_only: bool = False,
) -> list[NodeOrder]:
    query = sa.select(node_orders).order_by(node_orders.c.created.desc()).limit(limit)
    if algorithm is not None:
        query = query.where(
            node_orders.c.exec_algorithm == algorithm,
            node_orders.c.primary_id == node_orders.c.client_order_id,
        )
    if open_only:
        query = query.where(node_orders.c.is_closed.is_(False))
    return [NodeOrder.model_validate(row._mapping) for row in connection.execute(query)]


def order(connection: Connection, client_order_id: str) -> NodeOrder | None:
    row = connection.execute(
        sa.select(node_orders).where(node_orders.c.client_order_id == client_order_id)
    ).one_or_none()
    return None if row is None else NodeOrder.model_validate(row._mapping)


def family(connection: Connection, primary_id: str) -> list[NodeOrder]:
    """An algorithm's primary and the orders it spawned, oldest first."""
    rows = connection.execute(
        sa.select(node_orders)
        .where(node_orders.c.primary_id == primary_id)
        .order_by(node_orders.c.created)
    )
    return [NodeOrder.model_validate(row._mapping) for row in rows]


def fills(connection: Connection, client_order_ids: list[str]) -> list[NodeFill]:
    rows = connection.execute(
        sa.select(node_fills)
        .where(node_fills.c.client_order_id.in_(client_order_ids))
        .order_by(node_fills.c.time)
    )
    return [NodeFill.model_validate(row._mapping) for row in rows]


def reconciliation(connection: Connection, limit: int = 100) -> list[Reconciliation]:
    rows = connection.execute(
        sa.select(node_reconciliation).order_by(node_reconciliation.c.id.desc()).limit(limit)
    )
    return [Reconciliation.model_validate(row._mapping) for row in rows]


def health(connection: Connection, now: datetime | None = None) -> NodeHealth:
    rows = {row.name: row for row in connection.execute(sa.select(node_status))}
    beat, node = rows.get("heartbeat"), rows.get("node")
    state = node.value.get("state", "unknown") if node else "never started"
    heartbeat = beat.updated if beat else None
    fresh = heartbeat is not None and (now or datetime.now(UTC)) - heartbeat < HEARTBEAT_STALE
    return NodeHealth(
        up=fresh and state != "stopped",
        state=state if fresh or state == "stopped" else "silent",
        mode=node.value.get("mode") if node else None,
        heartbeat=heartbeat,
        open_orders=beat.value.get("open_orders") if beat else None,
        waiting=beat.value.get("waiting") if beat else None,
    )
