import uuid
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from app.core import breakers
from app.core.tables import compliance, order_command_events, order_commands, trading_state

FINISHED = ("done", "denied", "failed")
UNFINISHED = ("claimed", "sent", "unknown")


def claim(connection: Connection, node: str, limit: int = 10) -> list[dict[str, Any]]:
    """Take the oldest pending commands; SKIP LOCKED lets another node never take the same."""
    waiting = (
        sa.select(order_commands.c.id)
        .where(order_commands.c.status == "pending")
        .order_by(order_commands.c.created)
        .limit(limit)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )
    rows = connection.execute(
        sa.update(order_commands)
        .where(order_commands.c.id.in_(waiting))
        .values(
            status="claimed",
            claimed_by=node,
            claimed_at=sa.func.now(),
            attempts=order_commands.c.attempts + 1,
        )
        .returning(*order_commands.c)
    ).all()
    for row in rows:
        record(connection, row.id, "claimed", {"node": node})
    return sorted((dict(row._mapping) for row in rows), key=lambda row: row["created"])


def record(
    connection: Connection, command_id: uuid.UUID, status: str, detail: dict[str, Any] | None = None
) -> None:
    connection.execute(
        sa.insert(order_command_events).values(command_id=command_id, status=status, detail=detail)
    )


def mark(
    connection: Connection,
    command_id: uuid.UUID,
    status: str,
    detail: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    values: dict[str, Any] = {"status": status}
    if result is not None:
        values["result"] = result
    if error is not None:
        values["error"] = error
    connection.execute(
        sa.update(order_commands).where(order_commands.c.id == command_id).values(**values)
    )
    record(connection, command_id, status, detail)


def unfinished(connection: Connection) -> list[dict[str, Any]]:
    """Commands a node took and did not finish: the startup sweep decides what each one is."""
    rows = connection.execute(
        sa.select(order_commands)
        .where(order_commands.c.status.in_(UNFINISHED))
        .order_by(order_commands.c.created)
    )
    return [dict(row._mapping) for row in rows]


def requeue(connection: Connection, command_id: uuid.UUID, why: str) -> None:
    connection.execute(
        sa.update(order_commands)
        .where(order_commands.c.id == command_id)
        .values(status="pending", claimed_by=None, claimed_at=None)
    )
    record(connection, command_id, "pending", {"requeued": why})


def is_halted(connection: Connection) -> tuple[bool, str | None]:
    """Halted by the kill switch flag, a global or account kill switch, or the system breaker."""
    row = connection.execute(sa.select(trading_state).where(trading_state.c.id == 1)).one_or_none()
    if row is not None and row.halted:
        return True, row.reason
    stopped = breakers.stopped(connection)
    return (stopped is not None), stopped


def verdict(connection: Connection, symbol: str) -> tuple[bool, str] | None:
    row = connection.execute(
        sa.select(compliance).where(compliance.c.symbol == symbol)
    ).one_or_none()
    return None if row is None else (bool(row.allowed), row.reason)


def command_id_of(client_order_id: str) -> uuid.UUID | None:
    """`CMD-<hex>` and its spawned children `CMD-<hex>-<n>` belong to command <hex>."""
    if not client_order_id.startswith("CMD-"):
        return None
    try:
        return uuid.UUID(hex=client_order_id[4:].split("-")[0])
    except ValueError:
        return None


def now() -> datetime:
    return datetime.now(UTC)
