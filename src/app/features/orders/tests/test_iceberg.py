import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import PositionIntent
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.tests.support import api_error, until
from app.features.orders import iceberg as module
from app.features.orders.iceberg import IcebergDesk
from app.features.orders.router import get_icebergs
from app.features.orders.schemas import IcebergRequest
from app.main import app


class Broker:
    """An Alpaca stand-in that keeps orders by id and by client order id."""

    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {}
        self.by_client: dict[str, str] = {}
        self.requests: list[Any] = []
        self.client: MagicMock = create_autospec(TradingClient, instance=True)
        self.client.submit_order.side_effect = self.submit
        self.client.get_order_by_id.side_effect = lambda order_id: self.orders[order_id]
        self.client.get_order_by_client_id.side_effect = lambda key: self.orders[
            self.by_client[key]
        ]
        self.client.cancel_order_by_id.side_effect = self.cancel

    def submit(self, request: Any) -> dict[str, Any]:
        if request.client_order_id in self.by_client:
            raise api_error("client_order_id must be unique", 422)
        order_id = f"order-{len(self.orders) + 1}"
        self.requests.append(request)
        self.by_client[request.client_order_id] = order_id
        self.orders[order_id] = {
            "id": order_id,
            "symbol": request.symbol,
            "side": request.side.value,
            "type": "limit",
            "status": "new",
            "qty": str(request.qty),
            "filled_qty": "0",
            "filled_avg_price": None,
            "submitted_at": None,
        }
        return self.orders[order_id]

    def fill(self, order_id: str, qty: float | None = None, price: float = 10.0) -> None:
        order = self.orders[order_id]
        total = float(order["qty"])
        done = total if qty is None else qty
        order["filled_qty"] = str(done)
        order["filled_avg_price"] = str(price)
        order["status"] = "filled" if done >= total else "partially_filled"

    def cancel(self, order_id: str) -> None:
        self.orders[order_id]["status"] = "canceled"


def request(**fields: Any) -> IcebergRequest:
    body = {"symbol": "F", "side": "buy", "qty": 10, "display_qty": 3, "limit_price": 10, **fields}
    return IcebergRequest.model_validate(body)


@pytest.fixture
def broker() -> Broker:
    return Broker()


@pytest.fixture
def desk(broker: Broker, tmp_path: Path) -> IcebergDesk:
    return IcebergDesk(broker.client, tmp_path / "icebergs.json")


def test_placing_an_iceberg_shows_only_the_first_slice(desk: IcebergDesk, broker: Broker) -> None:
    item = desk.place(request())
    [sent] = broker.requests
    assert (sent.qty, sent.limit_price, sent.symbol) == (3, 10, "F")
    assert sent.client_order_id == f"iceberg-{item.id}-1"
    assert sent.position_intent == PositionIntent.BUY_TO_OPEN
    assert (item.status, len(item.slices), item.filled_qty) == ("working", 1, 0)


def test_each_filled_slice_is_followed_by_the_next_until_the_total_is_filled(
    desk: IcebergDesk, broker: Broker
) -> None:
    desk.place(request())
    for number, (size, price) in enumerate([(3, 10.0), (3, 11.0), (3, 12.0), (1, 13.0)], start=1):
        assert broker.requests[-1].qty == size
        broker.fill(f"order-{number}", price=price)
        desk.tick()
    [item] = desk.icebergs()
    assert [slice_.qty for slice_ in item.slices] == [3, 3, 3, 1]
    assert (item.status, item.filled_qty) == ("done", 10)
    assert item.average_price == pytest.approx((3 * 10 + 3 * 11 + 3 * 12 + 13) / 10)
    assert len(broker.requests) == 4


def test_a_partly_filled_slice_waits_for_the_rest(desk: IcebergDesk, broker: Broker) -> None:
    desk.place(request())
    broker.fill("order-1", qty=1)
    desk.tick()
    [item] = desk.icebergs()
    assert (item.status, item.filled_qty, len(broker.requests)) == ("working", 1, 1)


def test_canceling_cancels_the_slice_on_show_and_keeps_what_filled(
    desk: IcebergDesk, broker: Broker
) -> None:
    item = desk.place(request())
    broker.fill("order-1", qty=1)
    stopped = desk.cancel(item.id)
    assert (stopped.status, stopped.filled_qty) == ("canceled", 1)
    assert broker.orders["order-1"]["status"] == "canceled"
    broker.fill("order-1", qty=1)
    broker.orders["order-1"]["status"] = "canceled"
    desk.tick()
    assert len(broker.requests) == 1


def test_a_slice_canceled_elsewhere_stops_the_iceberg(desk: IcebergDesk, broker: Broker) -> None:
    desk.place(request())
    broker.orders["order-1"]["status"] = "canceled"
    desk.tick()
    [item] = desk.icebergs()
    assert (item.status, item.detail) == ("canceled", "Slice 1 canceled")


def test_a_slice_that_expires_expires_the_iceberg(desk: IcebergDesk, broker: Broker) -> None:
    desk.place(request(time_in_force="day"))
    broker.orders["order-1"]["status"] = "expired"
    desk.tick()
    assert desk.icebergs()[0].status == "expired"


def test_a_refused_slice_fails_the_iceberg_with_alpacas_reason(
    desk: IcebergDesk, broker: Broker
) -> None:
    desk.place(request())
    broker.fill("order-1")
    broker.client.submit_order.side_effect = api_error("insufficient buying power", 403)
    desk.tick()
    [item] = desk.icebergs()
    assert (item.status, item.detail) == ("failed", "insufficient buying power")
    assert item.filled_qty == 3


def test_a_busy_broker_is_retried_on_the_next_tick(desk: IcebergDesk, broker: Broker) -> None:
    desk.place(request())
    broker.fill("order-1")
    real = broker.client.submit_order.side_effect
    broker.client.submit_order.side_effect = api_error("try again", 503)
    desk.tick()
    assert desk.icebergs()[0].status == "working"
    broker.client.submit_order.side_effect = real
    desk.tick()
    assert len(desk.icebergs()[0].slices) == 2


def test_a_slice_placed_just_before_a_crash_is_adopted_not_placed_twice(
    broker: Broker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "icebergs.json"
    first = IcebergDesk(broker.client, path)
    first.place(request())
    broker.fill("order-1")

    def crash() -> None:
        raise RuntimeError("the process died after the order was sent")

    monkeypatch.setattr(first, "_save", crash)
    with pytest.raises(RuntimeError):
        first.tick()
    assert len(broker.requests) == 2

    second = IcebergDesk(broker.client, path)
    second.tick()
    [item] = second.icebergs()
    assert len(broker.requests) == 2
    assert [slice_.order for slice_ in item.slices] == ["order-1", "order-2"]


def test_a_slice_replaced_from_the_order_table_is_followed_to_its_new_order(
    desk: IcebergDesk, broker: Broker
) -> None:
    desk.place(request())
    broker.orders["order-1"].update(status="replaced", replaced_by="order-9")
    broker.orders["order-9"] = {
        **broker.orders["order-1"],
        "id": "order-9",
        "status": "new",
        "limit_price": "11.5",
    }
    desk.tick()
    [item] = desk.icebergs()
    assert (item.status, item.limit_price, item.slices[-1].order) == ("working", 11.5, "order-9")
    broker.fill("order-9")
    desk.tick()
    assert broker.requests[-1].limit_price == 11.5
    assert [slice_.order for slice_ in desk.icebergs()[0].slices] == ["order-9", "order-3"]


def test_icebergs_survive_a_restart(broker: Broker, tmp_path: Path) -> None:
    path = tmp_path / "icebergs.json"
    placed = IcebergDesk(broker.client, path).place(request())
    again = IcebergDesk(broker.client, path)
    assert [item.id for item in again.icebergs()] == [placed.id]
    broker.fill("order-1")
    again.tick()
    assert len(again.icebergs()[0].slices) == 2


def test_a_finished_iceberg_costs_no_further_calls(desk: IcebergDesk, broker: Broker) -> None:
    item = desk.place(request(qty=6))
    desk.cancel(item.id)
    calls = broker.client.get_order_by_id.call_count
    desk.tick()
    desk.tick()
    assert broker.client.get_order_by_id.call_count == calls


def test_working_icebergs_are_never_dropped_and_finished_ones_are_capped(
    broker: Broker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "KEPT_FINISHED", 2)
    desk = IcebergDesk(broker.client, tmp_path / "icebergs.json")
    ids = [desk.place(request()).id for _ in range(5)]
    for number in ids[:4]:
        desk.cancel(number)
    kept = [item.id for item in desk.icebergs()]
    assert ids[4] in kept and len(kept) == 3
    assert ids[0] not in kept and ids[1] not in kept


def test_cancelling_an_unknown_iceberg_is_an_error(desk: IcebergDesk) -> None:
    with pytest.raises(module.IcebergNotFoundError):
        desk.cancel("nope")


def test_the_scheduler_places_the_next_slice_on_its_own(
    broker: Broker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "EVERY", timedelta(milliseconds=50))

    async def scenario() -> int:
        desk = IcebergDesk(broker.client, tmp_path / "icebergs.json")
        desk.place(request())
        broker.fill("order-1")
        desk.start()
        await until(lambda: len(desk.icebergs()[0].slices) == 2)
        desk.stop()
        return len(desk.icebergs()[0].slices)

    assert asyncio.run(scenario()) == 2


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"display_qty": 10}, "display_qty must be below qty"),
        ({"display_qty": 12}, "display_qty must be below qty"),
        (
            {"symbol": "BTC/USD", "qty": 1, "display_qty": 0.25, "time_in_force": "day"},
            "accept gtc",
        ),
        ({"qty": 10.5}, "Limit orders for fractional shares accept day"),
        ({"display_qty": 1.5, "time_in_force": "gtc"}, "accept day"),
        ({"time_in_force": "ioc"}, "Input should be 'day' or 'gtc'"),
        ({"limit_price": 0}, "greater than 0"),
    ],
)
def test_an_iceberg_alpaca_could_not_slice_is_refused(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError) as caught:
        request(**fields)
    assert message in str(caught.value)


def test_a_coin_iceberg_and_a_day_iceberg_are_accepted() -> None:
    request(symbol="BTC/USD", qty=1, display_qty=0.25, time_in_force="gtc")
    request(time_in_force="day")


@pytest.fixture
def served(desk: IcebergDesk) -> IcebergDesk:
    app.dependency_overrides[get_icebergs] = lambda: desk
    return desk


BODY = {"symbol": "f", "side": "buy", "qty": 10, "display_qty": 3, "limit_price": 10}


def test_an_iceberg_is_placed_listed_and_canceled_through_the_api(
    client: TestClient, served: IcebergDesk, broker: Broker
) -> None:
    created = client.post("/orders/iceberg", json=BODY)
    assert created.status_code == 201
    body = created.json()
    assert (body["symbol"], body["status"], body["filled_qty"]) == ("F", "working", 0)
    assert body["slices"][0]["qty"] == 3
    assert [item["id"] for item in client.get("/orders/iceberg").json()] == [body["id"]]
    stopped = client.delete(f"/orders/iceberg/{body['id']}")
    assert stopped.status_code == 200 and stopped.json()["status"] == "canceled"
    assert client.delete("/orders/iceberg/nope").status_code == 404


def test_the_newest_iceberg_is_listed_first(client: TestClient, served: IcebergDesk) -> None:
    first = client.post("/orders/iceberg", json=BODY).json()["id"]
    second = client.post("/orders/iceberg", json=BODY).json()["id"]
    assert [item["id"] for item in client.get("/orders/iceberg").json()] == [second, first]


def test_a_refused_first_slice_is_a_422_and_nothing_is_kept(
    client: TestClient, served: IcebergDesk, broker: Broker
) -> None:
    broker.client.submit_order.side_effect = api_error("insufficient buying power", 403)
    response = client.post("/orders/iceberg", json=BODY)
    assert response.status_code == 422
    assert "insufficient buying power" in response.json()["detail"]
    assert client.get("/orders/iceberg").json() == []


def test_a_bad_iceberg_never_reaches_the_broker(
    client: TestClient, served: IcebergDesk, broker: Broker
) -> None:
    response = client.post("/orders/iceberg", json={**BODY, "display_qty": 10})
    assert response.status_code == 422
    assert not broker.requests
