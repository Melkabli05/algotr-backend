from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.core import audit
from app.features.orders.algo import AlgoNotFoundError
from app.features.orders.schemas import Algo, AlgoPlan, Interval, PlannedSlice
from app.features.orders.service import AlgoRefusedError

NOW = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)
BODY = {"algo": "twap", "symbol": "F", "side": "buy", "qty": 10, "duration": 10, "slices": 5}
SPAN = Interval(start=NOW, weight=1.0)
ALGO = Algo(
    id="abc",
    created=NOW,
    algo="twap",
    symbol="F",
    side="buy",
    qty=10,
    duration=10,
    start=NOW,
    end=NOW + timedelta(minutes=10),
    arrival_price=10.0,
    intervals=[SPAN],
)
PLAN = AlgoPlan(
    algo="twap",
    start=NOW,
    end=NOW + timedelta(minutes=10),
    arrival_price=10.0,
    intervals=[SPAN],
    slices=[PlannedSlice(due=NOW, qty=10)],
)


def last() -> dict[str, Any]:
    found = audit.AuditLog(audit.AUDIT_PATH).entries()[-1]
    assert found.kind == "order"
    return found.payload


def test_a_preview_returns_the_plan_and_places_nothing(
    client: TestClient, algos: MagicMock
) -> None:
    algos.plan.return_value = PLAN
    body = client.post("/orders/algo/preview", json=BODY).json()
    assert (body["arrival_price"], body["slices"][0]["qty"]) == (10.0, 10)
    algos.place.assert_not_called()


def test_a_preview_that_cannot_run_says_why(client: TestClient, algos: MagicMock) -> None:
    algos.plan.side_effect = AlgoRefusedError("The market is closed; an algorithm runs in hours")
    response = client.post("/orders/algo/preview", json=BODY)
    assert response.status_code == 422
    assert "market is closed" in response.json()["detail"]


def test_an_order_is_started_and_recorded(client: TestClient, algos: MagicMock) -> None:
    algos.place.return_value = ALGO
    response = client.post("/orders/algo", json=BODY)
    assert response.status_code == 201
    assert (response.json()["id"], response.json()["filled_qty"]) == ("abc", 0)
    request = algos.place.call_args.args[0]
    assert (request.algo, request.symbol, request.slices) == ("twap", "F", 5)
    entry = last()
    assert (entry["action"], entry["outcome"]) == ("algo", "accepted")
    assert entry["request"]["duration"] == 10
    assert entry["result"].pop("ms") >= 0
    assert entry["result"] == {"id": "abc", "status": "working"}


def test_a_buy_above_the_cash_is_refused_before_anything_starts(
    client: TestClient, trading: MagicMock, algos: MagicMock
) -> None:
    trading.get_account.return_value = {"cash": "10"}
    response = client.post("/orders/algo", json={**BODY, "limit_price": 12.5})
    assert response.status_code == 422
    assert "this account does not borrow" in response.json()["detail"]
    algos.place.assert_not_called()
    assert last()["outcome"] == "refused"


def test_a_refusal_from_the_desk_is_a_422_and_recorded(
    client: TestClient, algos: MagicMock
) -> None:
    algos.place.side_effect = AlgoRefusedError("You hold no F to sell")
    response = client.post("/orders/algo", json={**BODY, "side": "sell"})
    assert response.status_code == 422 and response.json()["detail"] == "You hold no F to sell"
    entry = last()
    assert (entry["outcome"], entry["reason"]) == ("refused", "You hold no F to sell")


def test_a_request_that_cannot_run_is_refused_before_the_routes_and_recorded(
    client: TestClient, algos: MagicMock
) -> None:
    response = client.post("/orders/algo", json={**BODY, "slices": None})
    assert response.status_code == 422
    entry = last()
    assert (entry["action"], entry["outcome"]) == ("algo", "refused")
    assert "A TWAP needs slices" in entry["reason"]
    algos.place.assert_not_called()


def test_the_orders_are_listed_with_their_totals(client: TestClient, algos: MagicMock) -> None:
    algos.algos.return_value = [ALGO]
    [item] = client.get("/orders/algo").json()
    assert (item["id"], item["filled_qty"], item["average_price"]) == ("abc", 0, None)
    assert item["slippage_bps"] is None


def test_an_order_is_canceled_and_recorded(client: TestClient, algos: MagicMock) -> None:
    algos.cancel.return_value = ALGO.model_copy(update={"status": "canceled"})
    assert client.delete("/orders/algo/abc").json()["status"] == "canceled"
    entry = last()
    assert entry["action"] == "algo-cancel"
    assert entry["result"].pop("ms") >= 0
    assert entry["result"] == {"id": "abc", "status": "canceled"}


def test_canceling_an_unknown_order_is_a_404(client: TestClient, algos: MagicMock) -> None:
    algos.cancel.side_effect = AlgoNotFoundError("nope")
    assert client.delete("/orders/algo/nope").status_code == 404


def test_a_vwap_is_asked_for_without_slices(client: TestClient, algos: MagicMock) -> None:
    algos.plan.return_value = PLAN
    body = {**BODY, "algo": "vwap", "slices": None, "duration": 30}
    assert client.post("/orders/algo/preview", json=body).status_code == 200
    assert algos.plan.call_args.args[0].algo == "vwap"


def test_a_shortfall_carries_its_risk_aversion_to_the_desk(
    client: TestClient, algos: MagicMock
) -> None:
    algos.plan.return_value = PLAN
    body = {**BODY, "algo": "shortfall", "risk_aversion": 3.5}
    assert client.post("/orders/algo/preview", json=body).status_code == 200
    assert algos.plan.call_args.args[0].risk_aversion == 3.5


def test_a_coin_order_is_not_judged_whole_because_its_slices_are_judged_as_they_go(
    client: TestClient, algos: MagicMock, guard: MagicMock, crypto_data: MagicMock
) -> None:
    algos.place.return_value = ALGO
    crypto_data.get_crypto_latest_trade.return_value = {"BTC/USD": {"p": 80000.0}}
    body = {**BODY, "symbol": "BTC/USD", "qty": 0.01}
    assert client.post("/orders/algo", json=body).status_code == 201
    guard.refusal.assert_not_called()
