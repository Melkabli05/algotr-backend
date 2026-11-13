from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.features.book.router import get_book
from app.features.book.service import BookDesk
from app.features.book.tests.support import snapshot
from app.main import app


@pytest.fixture
def desk(tmp_path: Path) -> Iterator[BookDesk]:
    found = BookDesk(["BTC/USD", "ETH/USD"], tmp_path / "book")
    found.keeper.handle(snapshot("BTC/USD"))
    app.dependency_overrides[get_book] = lambda: found
    yield found
    app.dependency_overrides.pop(get_book, None)


def test_the_view_has_levels_totals_and_the_spread(client: TestClient, desk: BookDesk) -> None:
    body = client.get("/book/view", params={"symbol": "btc/usd", "levels": 2}).json()
    assert body["symbol"] == "BTC/USD" and body["mid"] == 100.25
    assert [level["total"] for level in body["bids"]] == [1.0, 3.0]
    assert body["spread_bps"] == pytest.approx(0.5 / 100.25 * 1e4)


def test_a_stock_has_no_book_and_the_answer_says_why(client: TestClient, desk: BookDesk) -> None:
    response = client.get("/book/view", params={"symbol": "AAPL"})
    assert response.status_code == 404
    assert "Alpaca streams one for coins only" in response.json()["detail"]


def test_a_book_without_a_snapshot_is_503(client: TestClient, desk: BookDesk) -> None:
    response = client.get("/book/view", params={"symbol": "ETH/USD"})
    assert response.status_code == 503 and "not ready" in response.json()["detail"]


def test_levels_are_limited_to_the_twenty_five_that_are_kept(
    client: TestClient, desk: BookDesk
) -> None:
    assert client.get("/book/view", params={"symbol": "BTC/USD", "levels": 26}).status_code == 422
    assert client.get("/book/view", params={"symbol": "BTC/USD", "levels": 0}).status_code == 422


def test_the_status_lists_the_books_and_the_feed_age(client: TestClient, desk: BookDesk) -> None:
    body = client.get("/book/status").json()
    assert {book["symbol"]: book["ready"] for book in body["books"]} == {
        "BTC/USD": True,
        "ETH/USD": False,
    }
    assert body["stale"] is False and body["feed_age"] >= 0


def test_the_history_is_empty_before_anything_is_stored(client: TestClient, desk: BookDesk) -> None:
    assert client.get("/book/history", params={"symbol": "BTC/USD"}).json() == []
    assert client.get("/book/history", params={"symbol": "AAPL"}).status_code == 404
    assert (
        client.get("/book/history", params={"symbol": "BTC/USD", "minutes": 0}).status_code == 422
    )


def test_the_flow_endpoint_lists_windows_and_refuses_a_stock(
    client: TestClient, desk: BookDesk
) -> None:
    assert client.get("/book/flow", params={"symbol": "BTC/USD"}).json() == []
    assert client.get("/book/flow", params={"symbol": "AAPL"}).status_code == 404
    assert client.get("/book/flow", params={"symbol": "BTC/USD", "minutes": 0}).status_code == 422


def test_the_impact_of_a_market_order_is_walked_through_the_book(
    client: TestClient, desk: BookDesk
) -> None:
    body = client.get("/book/impact", params={"symbol": "btc/usd", "side": "buy", "qty": 3}).json()
    assert body["symbol"] == "BTC/USD" and body["side"] == "buy"
    assert (body["filled"], body["levels"], body["complete"]) == (3.0, 2, True)
    assert body["average"] == pytest.approx(302.5 / 3)
    assert body["advice"] == "market" and body["reason"] is None


def test_the_impact_says_when_to_use_a_limit_order(client: TestClient, desk: BookDesk) -> None:
    desk.max_slippage_bps = 100.0
    body = client.get("/book/impact", params={"symbol": "BTC/USD", "side": "buy", "qty": 7}).json()
    assert body["advice"] == "limit" and "would pay about 124.7 bps" in body["reason"]


def test_the_impact_of_a_stock_or_a_book_not_ready_is_refused_with_a_reason(
    client: TestClient, desk: BookDesk
) -> None:
    stock = client.get("/book/impact", params={"symbol": "AAPL", "side": "buy", "qty": 1})
    assert stock.status_code == 404
    waiting = client.get("/book/impact", params={"symbol": "ETH/USD", "side": "buy", "qty": 1})
    assert waiting.status_code == 503 and "not ready" in waiting.json()["detail"]


def test_the_impact_needs_a_side_and_a_positive_size(client: TestClient, desk: BookDesk) -> None:
    params: dict[str, str | int] = {"symbol": "BTC/USD", "side": "buy", "qty": 1}
    assert client.get("/book/impact", params={**params, "side": "hold"}).status_code == 422
    assert client.get("/book/impact", params={**params, "qty": 0}).status_code == 422
    assert client.get("/book/impact", params={**params, "qty": -1}).status_code == 422
    assert (
        client.get("/book/impact", params={"symbol": "BTC/USD", "side": "buy"}).status_code == 422
    )
