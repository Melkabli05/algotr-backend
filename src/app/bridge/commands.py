import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.engine import Connection

from app.core.tables import order_command_events, order_commands, trading_state

SUPPORTED_TYPES = ("market", "limit", "stop", "stop_limit", "trailing_stop")
SUPPORTED_CLASSES = ("simple", "bracket")
ALGORITHMS = ("schedule", "iceberg")


def unsupported(payload: Mapping[str, Any]) -> str | None:
    """Why the trading node cannot build this order, or None; the API and the node both ask."""
    if payload.get("type", "market") not in SUPPORTED_TYPES:
        return f"{payload.get('type')} orders are not supported by the Alpaca adapter yet"
    order_class = payload.get("order_class") or "simple"
    if order_class not in SUPPORTED_CLASSES:
        return (
            f"{order_class} orders are not built by Nautilus' order factory in this version; "
            "a bracket is"
        )
    if payload.get("stop_loss_limit") is not None:
        return "A stop-limit exit is not built by Nautilus' bracket factory in this version"
    algo = payload.get("algo")
    if algo and algo.get("name") not in ALGORITHMS:
        return f"{algo.get('name')} is not an execution algorithm the node runs"
    if algo and order_class != "simple":
        return "An execution algorithm works a simple order"
    return None


Kind = Literal["submit", "cancel", "modify", "cancel_all", "close_position", "rebalance"]
Source = Literal["manual", "autopilot", "agents", "kill_switch", "api", "arbiter", "breaker"]
CHANNEL = "order_commands"


class Command(BaseModel):
    id: uuid.UUID
    created: datetime
    kind: Kind
    payload: dict[str, Any]
    source: str
    status: str
    attempts: int
    client_order_id: str | None
    result: dict[str, Any] | None
    error: str | None


class CommandEvent(BaseModel):
    time: datetime
    status: str
    detail: dict[str, Any] | None


def client_order_id_for(command_id: uuid.UUID) -> str:
    """Fixed when the command is written, so a resent command is recognisable at Alpaca."""
    return f"CMD-{command_id.hex}"


def submit(
    connection: Connection,
    kind: Kind,
    payload: Mapping[str, Any],
    source: Source = "manual",
) -> Command:
    """Write a command the trading node will carry out; durable before anything reaches Alpaca."""
    command_id = uuid.uuid4()
    places = kind in ("submit", "rebalance", "close_position")
    client_order_id = client_order_id_for(command_id) if places else None
    row = connection.execute(
        sa.insert(order_commands)
        .values(
            id=command_id,
            kind=kind,
            payload=dict(payload),
            source=source,
            client_order_id=client_order_id,
        )
        .returning(*order_commands.c)
    ).one()
    connection.execute(
        sa.insert(order_command_events).values(command_id=command_id, status="pending")
    )
    connection.execute(
        sa.text("select pg_notify(:channel, :id)"), {"channel": CHANNEL, "id": str(command_id)}
    )
    return Command.model_validate(row._mapping)


def get(connection: Connection, command_id: uuid.UUID) -> Command | None:
    row = connection.execute(
        sa.select(order_commands).where(order_commands.c.id == command_id)
    ).one_or_none()
    return None if row is None else Command.model_validate(row._mapping)


def recent(connection: Connection, limit: int = 100) -> list[Command]:
    rows = connection.execute(
        sa.select(order_commands).order_by(order_commands.c.created.desc()).limit(limit)
    )
    return [Command.model_validate(row._mapping) for row in rows]


def events(connection: Connection, command_id: uuid.UUID) -> list[CommandEvent]:
    rows = connection.execute(
        sa.select(order_command_events)
        .where(order_command_events.c.command_id == command_id)
        .order_by(order_command_events.c.id)
    )
    return [CommandEvent.model_validate(row._mapping) for row in rows]


def withdraw(connection: Connection, command_id: uuid.UUID, reason: str) -> bool:
    """Take back a command no node has claimed yet; False once one has."""
    taken = connection.execute(
        sa.update(order_commands)
        .where(order_commands.c.id == command_id, order_commands.c.status == "pending")
        .values(status="denied", error=reason)
        .returning(order_commands.c.id)
    ).one_or_none()
    if taken is None:
        return False
    connection.execute(
        sa.insert(order_command_events).values(
            command_id=command_id, status="denied", detail={"withdrawn": reason}
        )
    )
    return True


def waiting(connection: Connection) -> int:
    """Commands written but not yet taken by the trading node."""
    return int(
        connection.execute(
            sa.select(sa.func.count()).where(order_commands.c.status == "pending")
        ).scalar_one()
    )


def set_halted(connection: Connection, halted: bool, reason: str | None = None) -> None:
    """The kill switch's flag; the trading node refuses new orders while it is set."""
    statement = sa.dialects.postgresql.insert(trading_state).values(
        id=1, halted=halted, reason=reason, updated=sa.func.now()
    )
    connection.execute(
        statement.on_conflict_do_update(
            index_elements=[trading_state.c.id],
            set_={"halted": halted, "reason": reason, "updated": sa.func.now()},
        )
    )


def halted(connection: Connection) -> tuple[bool, str | None]:
    row = connection.execute(sa.select(trading_state).where(trading_state.c.id == 1)).one_or_none()
    return (False, None) if row is None else (bool(row.halted), row.reason)
