from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest
from fastapi.testclient import TestClient

from app.bridge.commands import Command
from app.features.orders.iceberg import IcebergDesk
from app.features.orders.router import get_icebergs
from app.features.orders.schemas import Algo, Iceberg
from app.main import app
from conftest import NodeOrderMaker

Queued = Callable[[], list[Command]]
OPEN = {
    "client_order_id": "CMD-open",
    "instrument_id": "F.ALPACA",
    "order_type": "limit",
    "quantity": Decimal(2),
    "price": Decimal(1),
    "time_in_force": "gtc",
    "status": "accepted",
}


def change(client: TestClient, **fields: Any) -> Any:
    return client.patch("/orders/CMD-open", json=fields)


def test_a_limit_price_change_is_a_modify_command_and_the_order_keeps_its_id(
    client: TestClient, node_order: NodeOrderMaker, queued: Queued, node_running: None
) -> None:
    node_order(**OPEN)

    response = change(client, limit_price=1.1)

    assert response.status_code == 200
    body = response.json()
    assert (body["id"], body["status"]) == ("CMD-open", "pending_replace")
    [command] = queued()
    assert (command.kind, command.payload) == (
        "modify",
        {"client_order_id": "CMD-open", "limit_price": 1.1},
    )


def test_the_quantity_can_change(
    client: TestClient, node_order: NodeOrderMaker, queued: Queued
) -> None:
    node_order(**OPEN)
    assert change(client, qty=3).status_code == 200
    assert queued()[0].payload["qty"] == 3


@pytest.mark.parametrize(
    ("order", "fields", "message"),
    [
        ({}, {"stop_price": 5}, "stop_price does not apply to a limit order"),
        (
            {"order_type": "stop_market", "trigger_price": Decimal(9), "price": None},
            {"limit_price": 5},
            "does not apply to a stop order",
        ),
        ({"order_type": "market", "price": None}, {"limit_price": 5}, "a market order"),
        ({"status": "submitted"}, {"limit_price": 1.1}, "it can be changed once it is new"),
        ({"status": "pending_update"}, {"limit_price": 1.1}, "it can be changed once it is new"),
        ({"status": "filled"}, {"limit_price": 1.1}, "so it can no longer be changed"),
        ({"status": "canceled"}, {"qty": 4}, "so it can no longer be changed"),
        ({"quantity": Decimal("1.5")}, {"qty": 2}, "quantity of a fractional order cannot change"),
        ({}, {"qty": 2.5}, "only change to a whole number of shares"),
        (
            {"filled_qty": Decimal(2), "status": "partially_filled"},
            {"qty": 2},
            "already filled",
        ),
        ({}, {"time_in_force": "day"}, "The trading node changes price and quantity"),
        ({}, {"time_in_force": "opg"}, "Input should be"),
    ],
)
def test_a_change_that_would_be_refused_never_becomes_a_command(
    client: TestClient,
    node_order: NodeOrderMaker,
    queued: Queued,
    order: dict[str, Any],
    fields: dict[str, Any],
    message: str,
) -> None:
    node_order(**{**OPEN, **order})
    response = change(client, **fields)
    assert response.status_code == 422 and message in response.text
    assert queued() == []


def test_a_change_must_change_something(client: TestClient) -> None:
    response = change(client)
    assert response.status_code == 422 and "at least one" in response.text


def test_a_trailing_stops_trail_is_not_changed_by_the_node(
    client: TestClient, node_order: NodeOrderMaker, queued: Queued
) -> None:
    node_order(**{**OPEN, "order_type": "trailing_stop_market", "price": None})
    response = change(client, trail=10)
    assert response.status_code == 422 and "cancel and place anew" in response.text
    assert queued() == []


def test_a_change_to_an_order_the_node_does_not_have_is_not_found(
    client: TestClient, queued: Queued
) -> None:
    assert change(client, limit_price=1.2).status_code == 404
    assert queued() == []


@pytest.mark.parametrize(
    ("status", "shown"),
    [
        ("filled", "filled"),
        ("canceled", "canceled"),
        ("expired", "expired"),
        ("denied", "rejected"),
    ],
)
def test_canceling_a_finished_order_says_where_it_stands_and_sends_nothing(
    client: TestClient, node_order: NodeOrderMaker, queued: Queued, status: str, shown: str
) -> None:
    node_order(**{**OPEN, "status": status, "is_open": False, "is_closed": True})
    body = client.delete("/orders/CMD-open").json()
    assert (body["status"], body["canceled"], body["detail"]) == (
        shown,
        shown == "canceled",
        f"Already {shown}",
    )
    assert queued() == []


def test_a_cancel_while_the_node_is_down_waits_and_says_so(
    client: TestClient, node_order: NodeOrderMaker, queued: Queued
) -> None:
    node_order(**OPEN)
    body = client.delete("/orders/CMD-open").json()
    assert body["detail"].startswith("Queued: the trading node is not running")
    assert queued()[0].kind == "cancel"


NOW = datetime(2026, 11, 3, tzinfo=UTC)


def working(iceberg_id: str) -> Iceberg:
    return Iceberg(
        id=iceberg_id,
        created=NOW,
        symbol="F",
        side="buy",
        qty=3,
        display_qty=1,
        limit_price=1.0,
        time_in_force="gtc",
    )


@pytest.fixture
def desk() -> MagicMock:
    mock: MagicMock = create_autospec(IcebergDesk, instance=True)
    app.dependency_overrides[get_icebergs] = lambda: mock
    return mock


def test_canceling_everything_is_one_command_and_counts_what_the_node_has_working(
    client: TestClient, node_order: NodeOrderMaker, queued: Queued, desk: MagicMock
) -> None:
    desk.icebergs.return_value = []
    node_order(client_order_id="CMD-a")
    node_order(client_order_id="CMD-b")
    node_order(client_order_id="CMD-c", status="filled", is_open=False, is_closed=True)
    common = {"exec_algorithm": "ICEBERG", "primary_id": "CMD-i", "is_open": False}
    node_order(client_order_id="CMD-i", status="initialized", **common)

    body = client.delete("/orders").json()

    assert body == {"canceled": 2, "failed": [], "icebergs": 1, "algos": 0}
    assert [command.kind for command in queued()] == ["cancel_all"]


def test_canceling_everything_with_nothing_open_is_not_an_error(
    client: TestClient, queued: Queued, desk: MagicMock
) -> None:
    desk.icebergs.return_value = []
    assert client.delete("/orders").json() == {
        "canceled": 0,
        "failed": [],
        "icebergs": 0,
        "algos": 0,
    }


def test_canceling_everything_also_stops_the_algorithm_orders_before_the_orders(
    client: TestClient, queued: Queued, desk: MagicMock, algos: MagicMock
) -> None:
    calls: list[str] = []
    desk.icebergs.return_value = []
    running = Algo(
        id="x",
        created=NOW,
        algo="twap",
        symbol="F",
        side="buy",
        qty=5,
        duration=10,
        start=NOW,
        end=NOW,
        intervals=[],
    )
    finished = running.model_copy(update={"id": "y", "status": "done"})
    algos.algos.return_value = [running, finished]
    algos.cancel.side_effect = lambda algo_id: calls.append(f"algo {algo_id}")

    body = client.delete("/orders").json()
    assert calls == ["algo x"]
    assert body["algos"] == 1
    assert [command.kind for command in queued()] == ["cancel_all"]
