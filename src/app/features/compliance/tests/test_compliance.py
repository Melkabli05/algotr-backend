from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.core.compliance import Screening
from app.features.compliance import service
from app.features.compliance.router import get_screen
from app.main import app


class Lists:
    def screen(self, symbol: str) -> Screening:
        return Screening(symbol=symbol, allowed=symbol in {"SPUS", "BTC/USD"}, reason="test")

    def listed(self) -> list[str]:
        return ["AAPL", "SPUS"]


def held(symbol: str, asset_class: str = "us_equity") -> dict[str, Any]:
    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "side": "long",
        "qty": "2",
        "avg_entry_price": "10",
        "current_price": "11",
        "market_value": "22",
        "unrealized_pl": "2",
    }


def test_each_holding_carries_todays_verdict_with_coins_read_as_pairs() -> None:
    trading = MagicMock()
    trading.get_all_positions.return_value = [held("SPUS"), held("XOM"), held("BTCUSD", "crypto")]
    found = service.holdings(trading, Lists())
    assert [(h.symbol, h.screening.symbol, h.screening.allowed) for h in found] == [
        ("SPUS", "SPUS", True),
        ("XOM", "XOM", False),
        ("BTCUSD", "BTC/USD", True),
    ]


def test_the_routes_screen_a_symbol_and_the_holdings(
    client: TestClient, trading: MagicMock
) -> None:
    app.dependency_overrides[get_screen] = Lists
    trading.get_all_positions.return_value = [held("XOM")]
    assert client.get("/compliance/screen/BTC/USD").json()["allowed"] is True
    [row] = client.get("/compliance/holdings").json()
    assert (row["symbol"], row["screening"]["allowed"]) == ("XOM", False)


def test_the_universe_is_every_listed_name_with_its_verdict(client: TestClient) -> None:
    app.dependency_overrides[get_screen] = Lists
    found = client.get("/compliance/universe").json()
    assert [(item["symbol"], item["allowed"]) for item in found] == [
        ("AAPL", False),
        ("SPUS", True),
    ]
