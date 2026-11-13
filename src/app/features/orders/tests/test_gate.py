from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from fastapi.testclient import TestClient

from app.core.compliance import KillState, Screening
from app.core.tests.support import api_error
from app.features.orders import gate
from app.features.orders.risk import RiskRefusedError
from app.main import app

LIMITS = gate.Limits(position_share=0.25, daily_loss=0.03, adv_share=0.01, spread_bps=50)


class Lists:
    def screen(self, symbol: str) -> Screening:
        allowed = symbol != "JPM"
        return Screening(symbol=symbol, allowed=allowed, reason="held by SPUS" if allowed else "x")

    def listed(self) -> list[str]:
        return ["AAPL"]


class Switch:
    def __init__(self, state: KillState = "normal") -> None:
        self.state = state

    def kill_state(self) -> KillState:
        return self.state


def bars(price: float, volume: float) -> dict[str, list[dict[str, float]]]:
    return {"AAPL": [{"c": price, "v": volume}] * 20}


def built(**overrides: Any) -> tuple[gate.PreTrade, MagicMock, MagicMock]:
    trading = create_autospec(TradingClient, instance=True)
    trading.get_account.return_value = {"equity": "100000", "last_equity": "100000"}
    trading.get_open_position.side_effect = api_error("position does not exist", 404)
    stocks = create_autospec(StockHistoricalDataClient, instance=True)
    stocks.get_stock_bars.return_value = bars(100.0, 10_000_000)
    stocks.get_stock_latest_quote.return_value = {"AAPL": {"bp": 99.99, "ap": 100.01}}
    stocks.get_stock_latest_trade.return_value = {"AAPL": {"p": 100.0}}
    crypto = create_autospec(CryptoHistoricalDataClient, instance=True)
    crypto.get_crypto_latest_trade.return_value = {"BTC/USD": {"p": 80000.0}}
    checks = gate.PreTrade(
        trading,
        stocks,
        crypto,
        DataFeed.IEX,
        overrides.get("limits", LIMITS),
        Lists(),
        overrides.get("kill", Switch()),
    )
    return checks, trading, stocks


def test_a_buy_within_every_limit_passes_and_a_sell_is_never_stopped() -> None:
    checks, trading, _ = built()
    checks.check("AAPL", "buy", 10, None)
    checks.check("JPM", "sell", 10_000, None)
    trading.get_account.side_effect = ConnectionError("down")
    checks.check("AAPL", "sell", 1, None)


def test_a_stock_off_the_shariah_lists_is_refused() -> None:
    checks, _, _ = built()
    with pytest.raises(RiskRefusedError, match="JPM fails the Shariah screen"):
        checks.check("JPM", "buy", 1, 10.0)


@pytest.mark.parametrize("state", ["defensive", "flatten"])
def test_the_kill_switch_stops_new_buys_from_defensive_on(state: KillState) -> None:
    checks, _, _ = built(kill=Switch(state))
    with pytest.raises(RiskRefusedError, match=f"kill switch is at {state}"):
        checks.check("AAPL", "buy", 1, 100.0)
    built(kill=Switch("reduce"))[0].check("AAPL", "buy", 1, 100.0)


def test_a_day_down_past_the_limit_stops_buys() -> None:
    checks, trading, _ = built()
    trading.get_account.return_value = {"equity": "96000", "last_equity": "100000"}
    with pytest.raises(RiskRefusedError, match="down 4.0% today"):
        checks.check("AAPL", "buy", 1, 100.0)


def test_a_position_past_its_share_of_the_account_is_refused() -> None:
    checks, trading, _ = built()
    trading.get_open_position.side_effect = None
    trading.get_open_position.return_value = {"market_value": "24000"}
    checks.check("AAPL", "buy", 9, 100.0)
    with pytest.raises(RiskRefusedError, match="more than the 25% of account value"):
        checks.check("AAPL", "buy", 11, 100.0)


def test_an_order_past_its_share_of_daily_trading_is_refused() -> None:
    checks, _, stocks = built()
    stocks.get_stock_bars.return_value = bars(100.0, 10_000)
    with pytest.raises(RiskRefusedError, match="1.0% of the"):
        checks.check("AAPL", "buy", 200, 100.0)
    request = stocks.get_stock_bars.call_args.args[0]
    assert request.feed == DataFeed.SIP


def test_a_market_buy_across_a_wide_spread_is_refused_but_a_limit_buy_is_not() -> None:
    checks, _, stocks = built()
    stocks.get_stock_latest_quote.return_value = {"AAPL": {"bp": 99.0, "ap": 101.0}}
    with pytest.raises(RiskRefusedError, match="200 bps"):
        checks.check("AAPL", "buy", 1, None)
    checks.check("AAPL", "buy", 1, 100.0)


def test_a_coin_skips_the_stock_volume_and_spread_checks() -> None:
    checks, _, stocks = built()
    checks.check("BTC/USD", "buy", 0.01, None)
    stocks.get_stock_bars.assert_not_called()


@pytest.mark.parametrize(
    "broken",
    [
        lambda t, s: setattr(t.get_account, "return_value", {}),
        lambda t, s: setattr(t.get_open_position, "side_effect", api_error("boom", 500)),
        lambda t, s: setattr(s.get_stock_bars, "return_value", {}),
        lambda t, s: setattr(s.get_stock_latest_quote, "return_value", {}),
        lambda t, s: setattr(s.get_stock_latest_trade, "return_value", {}),
    ],
)
def test_a_check_that_cannot_read_its_data_refuses_the_buy(broken: Any) -> None:
    checks, trading, stocks = built()
    broken(trading, stocks)
    with pytest.raises(RiskRefusedError):
        checks.check("AAPL", "buy", 1, None)


def test_order_entry_refuses_what_the_gate_refuses(
    client: TestClient, gate: MagicMock, trading: MagicMock
) -> None:
    gate.check.side_effect = RiskRefusedError("JPM fails the Shariah screen: not held")
    order = {"symbol": "JPM", "side": "buy", "type": "limit", "qty": 1, "limit_price": 10}
    response = client.post("/orders", json=order)
    assert response.status_code == 422
    assert "Shariah" in response.json()["detail"]
    trading.submit_order.assert_not_called()
    gate.check.assert_called_once_with("JPM", "buy", 1.0, 10.0)


def test_the_gate_is_built_from_the_settings_and_the_apps_screen_and_switch() -> None:
    from types import SimpleNamespace

    from app.features.orders.router import get_gate

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(scanner=Lists(), automation=Switch()))
    )
    found = get_gate(request)  # type: ignore[arg-type]
    assert isinstance(found, gate.PreTrade)
    assert found.limits.position_share == 0.25
    app.dependency_overrides.clear()


def test_an_order_is_checked_without_being_sent(
    client: TestClient, gate: MagicMock, trading: MagicMock, stock_data: MagicMock
) -> None:
    gate.held_value.return_value = 1_000.0
    gate.equity.return_value = (100_000.0, 100_000.0)
    trading.get_account.return_value = {"cash": "100000"}
    order = {"symbol": "SPUS", "side": "buy", "type": "limit", "qty": 10, "limit_price": 50}
    body = client.post("/orders/check", json=order).json()
    assert body == {
        "allowed": True,
        "reason": None,
        "price": 50.0,
        "cost": 500.0,
        "position_value": 1500.0,
        "position_share": 0.015,
    }
    trading.submit_order.assert_not_called()
    gate.check.side_effect = RiskRefusedError("The kill switch is at defensive")
    refused = client.post("/orders/check", json=order).json()
    assert refused["allowed"] is False and "defensive" in refused["reason"]


def test_the_limits_and_kill_state_are_readable(client: TestClient, gate: MagicMock) -> None:
    gate.limits = LIMITS
    gate.kill = Switch("watch")
    body = client.get("/orders/limits").json()
    assert body["limits"]["position_share"] == 0.25 and body["kill_state"] == "watch"


class Sectors:
    def sector_of(self, symbol: str) -> str:
        return {"AAPL": "Information Technology", "MSFT": "Information Technology"}.get(
            symbol, "Fund"
        )


def test_a_stock_buy_past_its_sectors_share_is_refused() -> None:
    checks, trading, _ = built(limits=LIMITS.model_copy(update={"sector_share": 0.4}))
    checks.sectors = Sectors()
    trading.get_all_positions.return_value = [
        {"symbol": "MSFT", "market_value": "35000", "asset_class": "us_equity"},
        {"symbol": "SPUS", "market_value": "50000", "asset_class": "us_equity"},
    ]
    checks.check("AAPL", "buy", 40, 100.0)
    with pytest.raises(RiskRefusedError, match="Information Technology would be"):
        checks.check("AAPL", "buy", 60, 100.0)
