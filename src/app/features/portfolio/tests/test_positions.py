from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

POSITION = {
    "symbol": "AAPL",
    "side": "long",
    "qty": "10",
    "avg_entry_price": "150.5",
    "current_price": "160",
    "market_value": "1600",
    "unrealized_pl": "95",
    "asset_id": "ignored",
}


def test_positions_are_parsed_and_trimmed(client: TestClient, trading: MagicMock) -> None:
    trading.get_all_positions.return_value = [POSITION]
    [position] = client.get("/positions").json()
    assert position["symbol"] == "AAPL"
    assert position["unrealized_pl"] == 95.0
    assert "asset_id" not in position


def test_cost_basis_and_the_day_s_profit_come_through_when_alpaca_sends_them(
    client: TestClient, trading: MagicMock
) -> None:
    full = {
        **POSITION,
        "cost_basis": "1505",
        "unrealized_plpc": "0.0631",
        "unrealized_intraday_pl": "12.5",
        "qty_available": "8",
    }
    trading.get_all_positions.return_value = [full, POSITION]
    detailed, plain = client.get("/positions").json()
    assert (detailed["cost_basis"], detailed["unrealized_plpc"]) == (1505.0, 0.0631)
    assert (detailed["unrealized_intraday_pl"], detailed["qty_available"]) == (12.5, 8.0)
    assert plain["cost_basis"] is None and plain["unrealized_intraday_pl"] is None


def test_no_positions(client: TestClient, trading: MagicMock) -> None:
    trading.get_all_positions.return_value = []
    assert client.get("/positions").json() == []


def test_close_position(client: TestClient, trading: MagicMock) -> None:
    assert client.delete("/positions/AAPL").status_code == 204
    trading.close_position.assert_called_once_with("AAPL")


def test_close_position_with_slash_in_symbol(client: TestClient, trading: MagicMock) -> None:
    assert client.delete("/positions/BTC/USD").status_code == 204
    trading.close_position.assert_called_once_with("BTC/USD")


def test_close_position_normalizes_the_symbol(client: TestClient, trading: MagicMock) -> None:
    assert client.delete("/positions/btcusd").status_code == 204
    trading.close_position.assert_called_once_with("BTCUSD")


@pytest.mark.parametrize("path", ["%2E%2E/orders", "..%2Forders", "AAPL%2F..%2Forders", "A%20B"])
def test_close_position_rejects_paths_that_could_reach_other_endpoints(
    client: TestClient, trading: MagicMock, path: str
) -> None:
    assert client.delete(f"/positions/{path}").status_code == 422
    trading.close_position.assert_not_called()
