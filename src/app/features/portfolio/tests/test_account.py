from unittest.mock import MagicMock

from fastapi.testclient import TestClient

ACCOUNT = {
    "status": "ACTIVE",
    "currency": "USD",
    "equity": "101000",
    "last_equity": "100000",
    "cash": "90000.5",
    "buying_power": "400000",
    "trading_blocked": False,
    "account_number": "ignored",
}


def test_account_parses_numbers_and_computes_day_pl(client: TestClient, trading: MagicMock) -> None:
    trading.get_account.return_value = ACCOUNT
    assert client.get("/account").json() == {
        "status": "ACTIVE",
        "currency": "USD",
        "equity": 101000.0,
        "last_equity": 100000.0,
        "cash": 90000.5,
        "buying_power": 400000.0,
        "trading_blocked": False,
        "paper": True,
        "day_pl": 1000.0,
        "day_pl_pct": 1.0,
    }


def test_day_pl_pct_is_zero_without_prior_equity(client: TestClient, trading: MagicMock) -> None:
    trading.get_account.return_value = {**ACCOUNT, "last_equity": "0"}
    assert client.get("/account").json()["day_pl_pct"] == 0.0
