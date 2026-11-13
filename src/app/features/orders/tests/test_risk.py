from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from alpaca.data.enums import DataFeed
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine

from app.bridge import commands
from app.core import audit
from app.core.tests.support import api_error
from app.features.orders.iceberg import IcebergDesk
from app.features.orders.router import get_icebergs
from app.main import app

OPEN = {
    "id": "ae4f1bdb-be73-469d-a883-51a274bfd4bd",
    "symbol": "F",
    "side": "buy",
    "type": "limit",
    "status": "new",
    "qty": "2",
    "filled_qty": "0",
    "filled_avg_price": None,
    "submitted_at": "2026-11-03T10:00:00Z",
    "limit_price": "1",
    "time_in_force": "gtc",
}
LIMIT = {"symbol": "F", "side": "buy", "type": "limit", "qty": 10, "limit_price": 12.5}


def place(client: TestClient, **fields: object) -> int:
    response = client.post("/orders", json={**LIMIT, **fields})
    return int(response.status_code)


@pytest.fixture(autouse=True)
def accepted(trading: MagicMock, database: Engine) -> None:
    """Placed orders become commands in the test database."""


def nothing_queued(database: Engine) -> bool:
    with database.connect() as connection:
        return commands.recent(connection) == []


def test_a_buy_within_the_cash_is_placed(client: TestClient, trading: MagicMock) -> None:
    trading.get_account.return_value = {"cash": "125"}
    assert place(client) == 201


def test_a_buy_above_the_cash_is_refused_with_the_amounts(
    client: TestClient, trading: MagicMock, database: Engine
) -> None:
    trading.get_account.return_value = {"cash": "100"}
    response = client.post("/orders", json=LIMIT)
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "A buy costing $125.00 is more than the $100.00 of cash left after open buy orders; "
        "this account does not borrow"
    )
    assert nothing_queued(database)


def test_open_buy_orders_with_a_price_reduce_the_cash(
    client: TestClient, trading: MagicMock
) -> None:
    trading.get_account.return_value = {"cash": "200"}
    resting = {**OPEN, "qty": "8", "filled_qty": "2", "limit_price": "10", "status": "new"}
    trading.get_orders.return_value = [resting, {**resting, "side": "sell", "qty": "100"}]
    assert place(client, limit_price=12.5, qty=12) == 422
    assert place(client, limit_price=12.5, qty=11) == 201


def test_a_sell_is_never_stopped_by_the_cash_rule(client: TestClient, trading: MagicMock) -> None:
    trading.get_account.return_value = {"cash": "0"}
    assert place(client, side="sell") == 201


def test_a_market_buy_is_priced_at_the_last_trade(
    client: TestClient, trading: MagicMock, stock_data: MagicMock
) -> None:
    trading.get_account.return_value = {"cash": "100"}
    stock_data.get_stock_latest_trade.return_value = {"F": {"p": 12.5}}
    market = {"type": "market", "limit_price": None}
    assert place(client, qty=8, **market) == 201
    assert place(client, qty=9, **market) == 422


@pytest.mark.parametrize("feed", [DataFeed.IEX, DataFeed.SIP])
def test_the_last_trade_is_read_from_the_chosen_feed(
    client: TestClient,
    stock_data: MagicMock,
    choose_feed: Callable[[DataFeed], None],
    feed: DataFeed,
) -> None:
    choose_feed(feed)
    stock_data.get_stock_latest_trade.return_value = {"F": {"p": 5.0}}
    place(client, type="market", limit_price=None)
    assert stock_data.get_stock_latest_trade.call_args.args[0].feed == feed


def test_a_stop_buy_is_priced_at_its_stop_and_a_trailing_buy_at_the_last_trade(
    client: TestClient, trading: MagicMock, stock_data: MagicMock
) -> None:
    trading.get_account.return_value = {"cash": "100"}
    assert place(client, type="stop", limit_price=None, stop_price=11, qty=10) == 422
    stock_data.get_stock_latest_trade.return_value = {"F": {"p": 5.0}}
    trailing = {"type": "trailing_stop", "limit_price": None, "trail_percent": 5}
    assert place(client, qty=20, **trailing) == 201
    assert place(client, qty=21, **trailing) == 422


def test_a_coin_buy_is_priced_by_the_crypto_trade(
    client: TestClient, trading: MagicMock, crypto_data: MagicMock
) -> None:
    trading.get_account.return_value = {"cash": "1000"}
    crypto_data.get_crypto_latest_trade.return_value = {"BTC/USD": {"p": 80000.0}}
    coin = {"symbol": "BTC/USD", "type": "market", "limit_price": None, "time_in_force": "gtc"}
    assert place(client, qty=0.0125, **coin) == 201
    assert place(client, qty=0.02, **coin) == 422


def test_a_buy_that_cannot_be_priced_is_refused(
    client: TestClient,
    trading: MagicMock,
    stock_data: MagicMock,
    caplog: pytest.LogCaptureFixture,
    database: Engine,
) -> None:
    trading.get_account.return_value = {"cash": "1000000"}
    stock_data.get_stock_latest_trade.side_effect = api_error("no data", 404)
    assert place(client, type="market", limit_price=None) == 422
    assert "The last price of F could not be read" in caplog.text
    assert "no data" in caplog.text
    stock_data.get_stock_latest_trade.side_effect = None
    stock_data.get_stock_latest_trade.return_value = {}
    assert place(client, type="market", limit_price=None) == 422
    assert nothing_queued(database)


def test_a_sell_that_cannot_be_priced_is_placed(
    client: TestClient, trading: MagicMock, stock_data: MagicMock
) -> None:
    stock_data.get_stock_latest_trade.side_effect = api_error("no data", 404)
    assert place(client, type="market", limit_price=None, side="sell") == 201


def test_a_bracket_is_checked_on_its_entry(client: TestClient, trading: MagicMock) -> None:
    trading.get_account.return_value = {"cash": "100"}
    bracket = {"order_class": "bracket", "take_profit": 20, "stop_loss": 10}
    assert place(client, **bracket) == 422
    assert place(client, qty=8, **bracket) == 201


def test_an_iceberg_is_checked_on_its_whole_size(client: TestClient, trading: MagicMock) -> None:

    desk = MagicMock(spec=IcebergDesk)
    app.dependency_overrides[get_icebergs] = lambda: desk
    trading.get_account.return_value = {"cash": "100"}
    body = {"symbol": "F", "side": "buy", "qty": 10, "display_qty": 1, "limit_price": 12.5}
    response = client.post("/orders/iceberg", json=body)
    assert response.status_code == 422 and "does not borrow" in response.json()["detail"]
    desk.place.assert_not_called()


COIN = {"symbol": "BTC/USD", "type": "market", "limit_price": None, "time_in_force": "gtc"}
SLIPPAGE = "A market buy of 2 BTC/USD would pay about 124.7 bps over the mid; use a limit order"


def test_a_market_order_on_a_coin_is_refused_with_the_reason_of_its_book(
    client: TestClient, trading: MagicMock, guard: MagicMock, database: Engine
) -> None:
    guard.refusal.return_value = SLIPPAGE
    response = client.post("/orders", json={**LIMIT, **COIN, "qty": 2})
    assert response.status_code == 422
    assert response.json()["detail"] == SLIPPAGE
    guard.refusal.assert_called_once_with("BTC/USD", "buy", 2)
    assert nothing_queued(database)


def test_the_refusal_of_the_book_is_in_the_audit_trail(
    client: TestClient, guard: MagicMock
) -> None:
    guard.refusal.return_value = SLIPPAGE
    client.post("/orders", json={**LIMIT, **COIN, "qty": 2})
    entry = audit.AuditLog(audit.AUDIT_PATH).entries()[-1].payload
    assert (entry["action"], entry["outcome"], entry["reason"]) == ("submit", "refused", SLIPPAGE)
    assert entry["request"]["symbol"] == "BTC/USD"


def test_a_market_order_on_a_coin_that_its_book_allows_is_placed(
    client: TestClient, crypto_data: MagicMock, guard: MagicMock
) -> None:
    crypto_data.get_crypto_latest_trade.return_value = {"BTC/USD": {"p": 80000.0}}
    assert place(client, qty=0.01, **COIN) == 201
    guard.refusal.assert_called_once_with("BTC/USD", "buy", 0.01)


def test_a_market_sell_of_a_coin_is_put_to_the_book_too(
    client: TestClient, guard: MagicMock
) -> None:
    assert place(client, side="sell", qty=1, **COIN) == 201
    guard.refusal.assert_called_once_with("BTC/USD", "sell", 1)


def test_a_coin_order_with_a_price_never_asks_the_book(
    client: TestClient, guard: MagicMock
) -> None:
    coin = {"symbol": "BTC/USD", "limit_price": 80000.0, "time_in_force": "gtc"}
    assert place(client, qty=0.01, **coin) == 201
    guard.refusal.assert_not_called()


def test_a_market_order_on_a_stock_never_asks_the_book(
    client: TestClient, stock_data: MagicMock, guard: MagicMock
) -> None:
    stock_data.get_stock_latest_trade.return_value = {"F": {"p": 5.0}}
    assert place(client, type="market", limit_price=None) == 201
    guard.refusal.assert_not_called()
