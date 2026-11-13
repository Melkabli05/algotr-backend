from collections.abc import Callable
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.bridge.commands import Command
from conftest import NodeOrderMaker

Queued = Callable[[], list[Command]]


def test_orders_are_read_from_the_node_in_the_platform_shape(
    client: TestClient, node_order: NodeOrderMaker
) -> None:
    node_order(
        client_order_id="CMD-a",
        order_type="limit",
        price=Decimal("410.5"),
        status="partially_filled",
        filled_qty=Decimal(2),
        avg_px=Decimal("410.25"),
    )
    node_order(client_order_id="CMD-b", order_type="stop_market", status="accepted")
    node_order(client_order_id="CMD-c", time_in_force="at_the_open", status="denied")

    orders = {order["id"]: order for order in client.get("/orders").json()}

    assert (orders["CMD-a"]["type"], orders["CMD-a"]["limit_price"]) == ("limit", 410.5)
    assert (orders["CMD-a"]["status"], orders["CMD-a"]["filled_avg_price"]) == (
        "partially_filled",
        410.25,
    )
    assert (orders["CMD-b"]["type"], orders["CMD-b"]["status"]) == ("stop", "new")
    assert (orders["CMD-c"]["time_in_force"], orders["CMD-c"]["status"]) == ("opg", "rejected")
    assert orders["CMD-a"]["symbol"] == "MSFT"


def test_an_algorithms_primary_is_hidden_until_it_is_sent_itself(
    client: TestClient, node_order: NodeOrderMaker
) -> None:
    common = {"exec_algorithm": "SCHEDULE", "primary_id": "CMD-p"}
    node_order(client_order_id="CMD-p", venue_order_id=None, status="initialized", **common)
    node_order(client_order_id="CMD-p-E1", **common)

    assert [order["id"] for order in client.get("/orders").json()] == ["CMD-p-E1"]


def test_limit_is_validated(client: TestClient) -> None:
    assert client.get("/orders", params={"limit": 0}).status_code == 422
    assert client.get("/orders", params={"limit": 501}).status_code == 422


def test_an_order_is_written_as_a_command_for_the_node(
    client: TestClient, queued: Queued, node_running: None
) -> None:
    response = client.post(
        "/orders", json={"symbol": "msft", "side": "buy", "type": "market", "qty": 5}
    )

    assert response.status_code == 201
    [command] = queued()
    assert (command.kind, command.status, command.source) == ("submit", "pending", "manual")
    assert command.payload == {
        "symbol": "MSFT",
        "side": "buy",
        "type": "market",
        "qty": 5,
        "time_in_force": "day",
        "order_class": "simple",
    }
    body = response.json()
    assert body["id"] == command.client_order_id
    assert (body["status"], body["note"]) == ("pending_new", None)


def test_an_order_placed_while_the_node_is_down_waits_and_says_so(
    client: TestClient, queued: Queued
) -> None:
    body = {"symbol": "BTC/USD", "side": "sell", "type": "limit", "qty": 0.5}
    response = client.post("/orders", json={**body, "limit_price": 90000, "time_in_force": "gtc"})

    assert response.status_code == 201
    assert response.json()["note"].startswith("Queued: the trading node is not running")
    [command] = queued()
    assert command.payload["limit_price"] == 90000


@pytest.mark.parametrize(
    "body",
    [
        {"symbol": "AAPL", "side": "buy", "type": "limit", "qty": 1},
        {"symbol": "AAPL", "side": "buy", "type": "market", "qty": 0},
        {"symbol": "AAPL", "side": "hold", "type": "market", "qty": 1},
        {"symbol": "", "side": "buy", "type": "market", "qty": 1},
        {"symbol": "AAPL", "side": "buy", "type": "limit", "qty": 1, "limit_price": -1},
    ],
)
def test_invalid_orders_never_become_commands(
    client: TestClient, queued: Queued, body: dict[str, object]
) -> None:
    assert client.post("/orders", json=body).status_code == 422
    assert queued() == []


def test_an_order_the_node_cannot_build_is_refused_before_it_is_written(
    client: TestClient, queued: Queued
) -> None:
    body = {
        "symbol": "F",
        "side": "buy",
        "type": "limit",
        "qty": 2,
        "limit_price": 10,
        "order_class": "oto",
        "stop_loss": 8,
    }

    response = client.post("/orders", json=body)

    assert response.status_code == 422
    assert "oto orders are not built by Nautilus" in response.text
    assert queued() == []


def test_a_cancel_is_a_command_for_the_order(
    client: TestClient, node_order: NodeOrderMaker, queued: Queued, node_running: None
) -> None:
    node_order(client_order_id="CMD-a")

    response = client.delete("/orders/CMD-a")

    assert response.json() == {
        "id": "CMD-a",
        "status": "pending_cancel",
        "canceled": True,
        "detail": "Cancel sent to the trading node",
    }
    [command] = queued()
    assert (command.kind, command.payload) == ("cancel", {"client_order_id": "CMD-a"})


def test_a_cancel_for_an_order_the_node_does_not_have_is_not_found(
    client: TestClient, queued: Queued
) -> None:
    assert client.delete("/orders/CMD-missing").status_code == 404
    assert queued() == []


@pytest.mark.parametrize("symbol", ["../orders", "AAPL/../x", "A B", "A" * 11, "BTC/USD/X"])
def test_invalid_symbols_never_become_commands(
    client: TestClient, queued: Queued, symbol: str
) -> None:
    body = {"symbol": symbol, "side": "buy", "type": "market", "qty": 1}
    assert client.post("/orders", json=body).status_code == 422
    assert queued() == []


def test_symbols_are_normalized(client: TestClient, queued: Queued) -> None:
    body = {"symbol": " btc/usd ", "side": "buy", "type": "market", "qty": 1}
    assert client.post("/orders", json={**body, "time_in_force": "gtc"}).status_code == 201
    assert queued()[0].payload["symbol"] == "BTC/USD"


def test_the_order_rules_are_served_for_the_ticket(client: TestClient) -> None:
    shares = client.get("/orders/rules", params={"symbol": "f", "qty": 10}).json()
    assert shares["asset"] == "stock"
    assert shares["time_in_force"]["limit"] == ["day", "gtc", "ioc", "fok"]
    assert shares["trail_percent"] == [0.1, 25.0]
    part = client.get("/orders/rules", params={"symbol": "F", "qty": 1.5}).json()
    assert (part["asset"], part["time_in_force"]["stop"]) == ("fractional", ["day"])
    coin = client.get("/orders/rules", params={"symbol": "BTC/USD"}).json()
    assert coin["asset"] == "crypto" and "bracket" not in coin["time_in_force"]
    assert client.get("/orders/rules", params={"symbol": "F", "qty": 0}).status_code == 422


def test_a_bracket_is_one_command_with_both_exits(client: TestClient, queued: Queued) -> None:
    body = {
        "symbol": "F",
        "side": "buy",
        "type": "limit",
        "qty": 2,
        "limit_price": 10,
        "order_class": "bracket",
        "take_profit": 12,
        "stop_loss": 8,
    }

    response = client.post("/orders", json=body)

    assert response.status_code == 201
    assert response.json()["order_class"] == "bracket"
    payload = queued()[0].payload
    assert (payload["order_class"], payload["take_profit"], payload["stop_loss"]) == (
        "bracket",
        12,
        8,
    )


def test_an_order_alpaca_would_refuse_is_stopped_before_it_is_written(
    client: TestClient, queued: Queued
) -> None:
    body = {"symbol": "F", "side": "buy", "type": "stop", "qty": 1, "stop_price": 12}
    response = client.post("/orders", json={**body, "time_in_force": "fok"})
    assert response.status_code == 422
    assert "Stop orders for whole shares accept day, gtc" in response.text
    assert queued() == []
