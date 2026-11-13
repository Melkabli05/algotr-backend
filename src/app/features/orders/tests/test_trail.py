from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine

from app.core import audit
from app.features.orders.iceberg import IcebergDesk
from app.features.orders.router import get_icebergs
from app.features.orders.tests.test_lifecycle import OPEN
from app.main import app
from conftest import NodeOrderMaker

BODY = {"symbol": "F", "side": "buy", "type": "limit", "qty": 2, "limit_price": 12.5}


def entries() -> list[audit.Entry]:
    return audit.AuditLog(audit.AUDIT_PATH).entries()


def last() -> dict[str, Any]:
    found = entries()[-1]
    assert found.kind == "order"
    return found.payload


def test_an_accepted_order_is_recorded_with_what_was_asked_and_what_came_back(
    client: TestClient, database: Engine
) -> None:
    response = client.post("/orders", json=BODY)
    assert response.status_code == 201
    entry = last()
    assert entry["action"] == "submit" and entry["outcome"] == "accepted"
    assert entry["request"]["symbol"] == "F" and entry["request"]["limit_price"] == 12.5
    assert entry["result"].pop("ms") >= 0
    assert entry["result"] == {"id": response.json()["id"], "status": "pending_new", "usage": {}}
    assert entry["reason"] is None


def test_an_order_refused_before_it_reaches_the_routes_is_recorded_too(
    client: TestClient, trading: MagicMock
) -> None:
    response = client.post("/orders", json={**BODY, "type": "stop", "limit_price": None})
    assert response.status_code == 422
    entry = last()
    assert (entry["action"], entry["outcome"]) == ("submit", "refused")
    assert "stop_price is required for stop orders" in entry["reason"]
    assert entry["request"]["symbol"] == "F"


def test_a_buy_that_breaks_the_cash_rule_is_recorded_as_refused(
    client: TestClient, trading: MagicMock
) -> None:
    trading.get_account.return_value = {"cash": "10"}
    response = client.post("/orders", json=BODY)
    assert response.status_code == 422 and "does not borrow" in response.json()["detail"]
    assert "does not borrow" in last()["reason"]


def test_a_change_is_recorded_with_the_order(
    client: TestClient, node_order: NodeOrderMaker
) -> None:
    node_order(**OPEN)
    assert client.patch("/orders/CMD-open", json={"limit_price": 1.1}).status_code == 200
    entry = last()
    assert (entry["action"], entry["outcome"]) == ("change", "accepted")
    assert entry["request"] == {"id": "CMD-open", "limit_price": 1.1}
    assert entry["result"] == {
        "id": "CMD-open",
        "status": "pending_replace",
        "ms": entry["result"]["ms"],
    }


def test_a_change_that_is_refused_is_recorded_whether_it_fails_here_or_in_the_routes(
    client: TestClient, node_order: NodeOrderMaker
) -> None:
    node_order(**{**OPEN, "status": "filled", "is_open": False, "is_closed": True})
    assert client.patch("/orders/CMD-open", json={"limit_price": 1.1}).status_code == 422
    assert (last()["outcome"], "no longer be changed" in last()["reason"]) == ("refused", True)
    assert client.patch("/orders/CMD-open", json={}).status_code == 422
    entry = last()
    assert (entry["action"], entry["outcome"], entry["request"]) == ("change", "refused", {})


def test_a_cancel_and_a_cancel_all_are_recorded(
    client: TestClient, node_order: NodeOrderMaker
) -> None:
    node_order(**{**OPEN, "status": "filled", "is_open": False, "is_closed": True})
    client.delete("/orders/CMD-open")
    entry = last()
    assert (entry["action"], entry["request"]) == ("cancel", {"id": "CMD-open"})
    assert entry["result"]["detail"] == "Already filled"
    node_order(client_order_id="CMD-working")

    desk = MagicMock(spec=IcebergDesk)
    desk.icebergs.return_value = []
    app.dependency_overrides[get_icebergs] = lambda: desk
    client.delete("/orders")
    entry = last()
    assert (entry["action"], entry["result"]["canceled"]) == ("cancel-all", 1)


def test_the_hash_chain_stays_valid_across_every_kind_of_attempt(
    client: TestClient, database: Engine
) -> None:
    client.post("/orders", json=BODY)
    client.post("/orders", json={**BODY, "qty": 0})
    client.post("/orders", json={**BODY, "type": "stop", "limit_price": None})
    log = audit.AuditLog(audit.AUDIT_PATH)
    assert log.verify().valid and len(log.entries()) == 3


def test_reading_orders_and_other_validation_errors_leave_no_entry(
    client: TestClient, database: Engine
) -> None:
    assert client.get("/orders", params={"limit": 0}).status_code == 422
    assert client.get("/orders").status_code == 200
    assert entries() == []


def test_a_fault_in_the_audit_log_does_not_undo_or_block_an_order(
    client: TestClient, database: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: audit.AuditLog, kind: str, payload: dict[str, Any]) -> audit.Entry:
        raise OSError("disk full")

    monkeypatch.setattr(audit.AuditLog, "record", broken)
    assert client.post("/orders", json=BODY).status_code == 201
