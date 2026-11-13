from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.tests.support import api_error
from app.features.ledger.router import get_ledger
from app.features.ledger.service import Ledger
from app.features.ledger.tests.support import Broker, fill, position
from app.main import app

STOCKS = [
    fill("2026-10-28T14:00:00", "o1", "buy", "10", "50"),
    fill("2026-10-29T14:00:00", "o2", "buy", "10", "60"),
    fill("2026-10-30T14:00:00", "o3", "sell", "5", "70"),
]


@pytest.fixture
def broker() -> Broker:
    return Broker([*STOCKS], [position("SPUS", "15", "55", price="65")])


@pytest.fixture
def served(broker: Broker, tmp_path: Path) -> Iterator[Ledger]:
    desk = Ledger(broker.client, tmp_path / "ledger.sqlite")
    app.dependency_overrides[get_ledger] = lambda: desk
    yield desk


def test_nothing_is_known_before_the_first_sync(client: TestClient, served: Ledger) -> None:
    status = client.get("/ledger/status").json()
    assert (status["activities"], status["reconciliation"], status["error"]) == (0, None, None)
    assert client.get("/ledger/reconciliation").json() is None
    assert client.get("/ledger/positions").json() == []


def test_a_sync_reports_whether_the_ledger_matches_the_broker(
    client: TestClient, served: Ledger
) -> None:
    body = client.post("/ledger/sync").json()
    assert (body["ok"], body["differences"], body["activities"]) == (True, [], 3)
    assert client.get("/ledger/reconciliation").json()["ok"] is True
    assert client.get("/ledger/status").json()["activities"] == 3


def test_positions_follow_the_method_in_the_query(client: TestClient, served: Ledger) -> None:
    client.post("/ledger/sync")
    [fifo] = client.get("/ledger/positions", params={"method": "fifo"}).json()
    [average] = client.get("/ledger/positions", params={"method": "average"}).json()
    assert (fifo["basis"], fifo["unrealized"], fifo["realized"]) == (850, 125, 100)
    assert (average["basis"], average["unrealized"], average["realized"]) == (825, 150, 75)
    assert client.get("/ledger/positions", params={"method": "lifo"}).status_code == 422


def test_lots_and_realized_rows_can_be_filtered(client: TestClient, served: Ledger) -> None:
    client.post("/ledger/sync")
    lots = client.get("/ledger/lots", params={"symbol": "spus"}).json()
    assert [(lot["id"], lot["qty"]) for lot in lots] == [("o1", 5), ("o2", 10)]
    assert client.get("/ledger/lots", params={"symbol": "AAPL"}).json() == []
    realized = client.get("/ledger/realized").json()
    assert (realized["summary"]["count"], realized["summary"]["gain"]) == (1, 100)
    assert client.get("/ledger/realized", params={"days": 1}).json()["rows"] == []


def test_the_lots_a_sale_closes_are_chosen_and_cleared(client: TestClient, served: Ledger) -> None:
    client.post("/ledger/sync")
    body = {"lots": [{"lot": "o2", "qty": 5}]}
    assert client.put("/ledger/picks/o3", json=body).json() == body
    assert client.get("/ledger/picks/o3").json() == body
    gain = client.get("/ledger/realized", params={"method": "specific"}).json()["summary"]["gain"]
    assert gain == 50
    assert client.delete("/ledger/picks/o3").status_code == 204
    assert client.get("/ledger/picks/o3").json() == {"lots": []}
    bad = {"lots": [{"lot": "o2", "qty": 0}]}
    assert client.put("/ledger/picks/o3", json=bad).status_code == 422


def test_adopting_makes_the_ledger_hold_what_alpaca_holds(
    client: TestClient, served: Ledger, broker: Broker
) -> None:
    assert client.post("/ledger/sync").json()["ok"] is True
    assert client.post("/ledger/adopt/SPUS").status_code == 409
    broker.positions[:] = [position("SPUS", "20", "57")]
    assert client.post("/ledger/sync").json()["ok"] is False
    adopted = client.post("/ledger/adopt/spus")
    assert adopted.status_code == 200 and adopted.json()["ok"] is True


def test_a_symbol_with_a_slash_can_be_adopted_too(
    client: TestClient, served: Ledger, broker: Broker
) -> None:
    broker.positions[:] = [position("BTCUSD", "0.5", "80000")]
    assert client.post("/ledger/sync").json()["ok"] is False
    assert client.post("/ledger/adopt/BTC/USD").status_code == 200


def test_a_broker_error_during_a_sync_is_reported_not_swallowed(
    client: TestClient, served: Ledger, broker: Broker
) -> None:
    broker.client.get.side_effect = api_error("service unavailable", 503)
    response = client.post("/ledger/sync")
    assert response.status_code == 502 and "service unavailable" in response.json()["detail"]
    assert "service unavailable" in client.get("/ledger/status").json()["error"]


def test_the_stored_fills_are_listed_and_can_be_narrowed(
    client: TestClient, served: Ledger
) -> None:
    client.post("/ledger/sync")
    rows = client.get("/ledger/fills").json()
    assert [row["order"] for row in rows] == ["o3", "o2", "o1"]
    assert rows[0]["value"] == 350 and rows[0]["partial"] is False
    assert [row["order"] for row in client.get("/ledger/fills", params={"limit": 1}).json()] == [
        "o3"
    ]
    assert client.get("/ledger/fills", params={"symbol": "AAPL"}).json() == []
    assert client.get("/ledger/fills", params={"limit": 0}).status_code == 422


def test_the_purification_statement_uses_the_ratios_set_through_the_api(
    client: TestClient, served: Ledger, broker: Broker
) -> None:
    broker.rows.append(
        {
            "id": "20260320000000000::div",
            "activity_type": "DIV",
            "date": "2026-03-20",
            "net_amount": "50.00",
            "symbol": "SPUS",
        }
    )
    broker.client.get_account.return_value = {"equity": "80000"}
    client.post("/ledger/sync")
    first = client.get("/ledger/purification", params={"year": 2026}).json()
    assert first["missing"] == ["SPUS"] and first["owed"] == "0"
    saved = client.put("/ledger/purification/ratios", json={"spus": "0.02"}).json()
    assert saved == {"SPUS": "0.02"}
    assert client.get("/ledger/purification/ratios").json() == {"SPUS": "0.02"}
    body = client.get("/ledger/purification", params={"year": 2026}).json()
    assert body["owed"] == "1.0000"
    assert body["zakat"] == "2000.000"
    assert client.put("/ledger/purification/ratios", json={"SPUS": None}).json() == {}
    assert client.put("/ledger/purification/ratios", json={"SPUS": "1.5"}).status_code == 422


def test_the_zakat_estimate_is_left_out_when_the_account_cannot_be_read(
    client: TestClient, served: Ledger, broker: Broker
) -> None:
    broker.client.get_account.side_effect = api_error("down", 500)
    assert client.get("/ledger/purification").json()["zakat"] is None


def test_a_loss_bought_back_within_30_days_is_reported_as_a_wash_sale(
    client: TestClient, served: Ledger, broker: Broker
) -> None:
    broker.rows[:] = [
        fill("2026-09-01T14:00:00", "w1", "buy", "10", "50"),
        fill("2026-09-20T14:00:00", "w2", "sell", "10", "40"),
        fill("2026-09-25T14:00:00", "w3", "buy", "10", "42"),
    ]
    broker.positions[:] = [position("SPUS", "10", "42", price="43")]
    client.post("/ledger/sync")
    realized = client.get("/ledger/realized").json()
    assert realized["summary"]["gain"] == -100
    assert (realized["summary"]["disallowed"], realized["summary"]["reported"]) == (100, 0)
    [lot] = client.get("/ledger/lots").json()
    assert (lot["basis"], lot["wash"]) == (520, 100)
    assert client.post("/ledger/sync").json()["ok"] is True
