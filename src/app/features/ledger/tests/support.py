from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock, create_autospec

from alpaca.trading.client import TradingClient


def row(*fields: object) -> tuple[object, ...]:
    return fields


def digits(stamp: str) -> str:
    return "".join(char for char in stamp if char.isdigit())[:14]


def fill(
    stamp: str,
    order: str,
    side: str,
    qty: str,
    price: str,
    symbol: str = "SPUS",
    partial: bool = False,
) -> dict[str, Any]:
    return {
        "id": f"{digits(stamp)}::{order}-{qty}-{price}",
        "activity_type": "FILL",
        "transaction_time": f"{stamp}Z",
        "type": "partial_fill" if partial else "fill",
        "price": price,
        "qty": qty,
        "side": side,
        "symbol": symbol,
        "order_id": order,
        "cum_qty": qty,
        "leaves_qty": "0",
    }


def coin_fee(stamp: str, qty: str, price: str, symbol: str = "BTCUSD") -> dict[str, Any]:
    return {
        "id": f"{stamp[:10].replace('-', '')}000000000::fee-{stamp}-{qty}",
        "activity_type": "CFEE",
        "date": stamp[:10],
        "created_at": f"{stamp}Z",
        "net_amount": "0",
        "description": "Coin Pair Transaction Fee (Non USD)",
        "symbol": symbol,
        "qty": qty,
        "price": price,
        "status": "executed",
    }


def usd_fee(stamp: str, amount: str) -> dict[str, Any]:
    return {
        "id": f"{stamp[:10].replace('-', '')}000000000::usd-{stamp}",
        "activity_type": "CFEE",
        "date": stamp[:10],
        "created_at": f"{stamp}Z",
        "net_amount": amount,
        "description": "Coin Pair Transaction Fee (USD)",
        "status": "executed",
    }


def position(symbol: str, qty: str, average: str, price: str | None = None) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "qty": qty,
        "avg_entry_price": average,
        "current_price": price or average,
    }


def served_on(row: dict[str, Any]) -> str:
    """The date Alpaca lists a row under: fills on their day, fees and other rows a day later."""
    if row["activity_type"] == "FILL":
        return str(row["transaction_time"])[:10]
    return (date.fromisoformat(row["date"]) + timedelta(days=1)).isoformat()


class Broker:
    """An Alpaca stand-in serving activities by page or by date, and a list of positions."""

    def __init__(self, rows: list[dict[str, Any]], positions: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.positions = positions
        self.calls: list[dict[str, Any]] = []
        self.client: MagicMock = create_autospec(TradingClient, instance=True)
        self.client.get.side_effect = self.get
        self.client.get_all_positions.side_effect = lambda: self.positions

    def get(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        assert path == "/account/activities"
        params = params or {}
        self.calls.append(params)
        ordered = sorted(self.rows, key=lambda row: row["id"])
        if "date" in params:
            return [row for row in ordered if served_on(row) == params["date"]]
        ids = [row["id"] for row in ordered]
        start = ids.index(params["page_token"]) + 1 if "page_token" in params else 0
        return ordered[start : start + params["page_size"]]
