import json
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from app.bridge.fanout import EVENTS, MARKET
from app.core.tables import (
    node_fills,
    node_orders,
    node_positions,
    node_reconciliation,
    node_status,
)

logger = logging.getLogger(__name__)

NOTIFY_LIMIT = 7900
Work = Callable[[Connection], dict[str, Any] | None]


class StateWriter:
    """Writes the node's state to Postgres on its own thread, so the node's loop never waits.

    Each piece of work runs in one transaction; what it returns is sent on the events channel
    in the same transaction, so a listener never hears of a row it cannot read.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.queue: queue.Queue[Work | None] = queue.Queue()
        self.thread = threading.Thread(target=self.run, name="node-state-writer", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self.queue.put(None)
        self.thread.join(timeout)

    def put(self, work: Work) -> None:
        self.queue.put(work)

    def run(self) -> None:
        while (work := self.queue.get()) is not None:
            try:
                with self.engine.begin() as connection:
                    event = work(connection)
                    if event is not None:
                        notify(connection, event)
            except Exception:
                logger.exception("The node could not write its state")

    def flush(self, timeout: float = 5.0) -> None:
        done = threading.Event()

        def work(_: Connection) -> None:
            done.set()

        self.put(work)
        done.wait(timeout)

    def market(self, events: list[dict[str, Any]]) -> None:
        """Market data goes out on its own channel, in one transaction per batch, unstored."""

        def work(connection: Connection) -> None:
            for event in events:
                notify(connection, event, MARKET)

        self.put(work)

    def order(self, row: dict[str, Any]) -> None:
        def work(connection: Connection) -> dict[str, Any]:
            statement = insert(node_orders).values(**row)
            keep = {
                key: value
                for key, value in row.items()
                if key not in ("client_order_id", "created")
            }
            connection.execute(
                statement.on_conflict_do_update(
                    index_elements=[node_orders.c.client_order_id], set_=keep
                )
            )
            return {"type": "order", **jsonable(row)}

        self.put(work)

    def fill(self, row: dict[str, Any]) -> None:
        def work(connection: Connection) -> dict[str, Any] | None:
            inserted = connection.execute(
                insert(node_fills)
                .values(**row)
                .on_conflict_do_nothing(index_elements=[node_fills.c.trade_id])
            )
            return {"type": "fill", **jsonable(row)} if inserted.rowcount else None

        self.put(work)

    def positions(self, rows: list[dict[str, Any]]) -> None:
        def work(connection: Connection) -> dict[str, Any]:
            held = [row["instrument_id"] for row in rows]
            connection.execute(
                sa.delete(node_positions).where(node_positions.c.instrument_id.not_in(held))
            )
            for row in rows:
                statement = insert(node_positions).values(**row)
                connection.execute(
                    statement.on_conflict_do_update(
                        index_elements=[node_positions.c.instrument_id],
                        set_={key: value for key, value in row.items() if key != "instrument_id"},
                    )
                )
            return {"type": "positions", "count": len(rows)}

        self.put(work)

    def status(self, name: str, value: dict[str, Any], announce: bool = False) -> None:
        def work(connection: Connection) -> dict[str, Any] | None:
            statement = insert(node_status).values(
                name=name, value=jsonable(value), updated=sa.func.now()
            )
            connection.execute(
                statement.on_conflict_do_update(
                    index_elements=[node_status.c.name],
                    set_={"value": jsonable(value), "updated": sa.func.now()},
                )
            )
            return {"type": "status", "name": name, **jsonable(value)} if announce else None

        self.put(work)

    def reconciliation(self, kind: str, severity: str, detail: str, **ids: str | None) -> None:
        def work(connection: Connection) -> dict[str, Any]:
            connection.execute(
                sa.insert(node_reconciliation).values(
                    kind=kind, severity=severity, detail=detail, **ids
                )
            )
            return {"type": "reconciliation", "kind": kind, "severity": severity, "detail": detail}

        self.put(work)


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def notify(connection: Connection, event: dict[str, Any], channel: str = EVENTS) -> None:
    payload = json.dumps(event, default=str)
    if len(payload) > NOTIFY_LIMIT:
        payload = json.dumps({"type": event.get("type"), "truncated": True})
    connection.execute(
        sa.text("select pg_notify(:channel, :payload)"), {"channel": channel, "payload": payload}
    )
