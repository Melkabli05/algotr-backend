from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import requests
from alpaca.common.exceptions import APIError
from alpaca.data.models import Bar, Orderbook, Quote, Trade
from alpaca.trading.models import Asset, Order, Position, TradeAccount, TradeUpdate
from nautilus_trader.core import dt_to_unix_nanos

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
NOW_NS = dt_to_unix_nanos(NOW)
ORDER_ID = UUID("11111111-1111-1111-1111-111111111111")
ASSET_ID = UUID("22222222-2222-2222-2222-222222222222")
OTHER_ORDER_ID = UUID("33333333-3333-3333-3333-333333333333")
FILL_ID_ONE = UUID("44444444-4444-4444-4444-444444444444")
FILL_ID_TWO = UUID("55555555-5555-5555-5555-555555555555")


RAW_BAR = {
    "t": NOW.isoformat(),
    "o": 80990.0,
    "h": 81010.5,
    "l": 80980.25,
    "c": 81000.5,
    "v": 12.5,
    "n": 7,
    "vw": 81000.0,
}


def asset(symbol: str = "AAPL", **overrides: Any) -> Asset:
    crypto = "/" in symbol
    return Asset.model_validate(
        {
            "id": ASSET_ID,
            "class": "crypto" if crypto else "us_equity",
            "exchange": "CRYPTO" if crypto else "NASDAQ",
            "symbol": symbol,
            "status": "active",
            "tradable": True,
            "marginable": not crypto,
            "shortable": not crypto,
            "easy_to_borrow": not crypto,
            "fractionable": True,
            **(
                {
                    "min_order_size": 0.000012372,
                    "min_trade_increment": 0.000000001,
                    "price_increment": 0.000000001,
                }
                if crypto
                else {}
            ),
            **overrides,
        }
    )


def order(symbol: str = "BTC/USD", **overrides: Any) -> Order:
    return Order.model_validate(
        {
            "id": ORDER_ID,
            "client_order_id": "O-1",
            "created_at": NOW,
            "updated_at": NOW,
            "submitted_at": NOW,
            "order_class": "simple",
            "time_in_force": "gtc",
            "status": "new",
            "extended_hours": False,
            "symbol": symbol,
            "asset_class": "crypto" if "/" in symbol else "us_equity",
            "qty": "0.5",
            "filled_qty": "0",
            "side": "buy",
            "order_type": "limit",
            "type": "limit",
            "limit_price": "80000",
            **overrides,
        }
    )


def position(symbol: str = "BTCUSD", **overrides: Any) -> Position:
    crypto = symbol.endswith("USD")
    return Position.model_validate(
        {
            "asset_id": ASSET_ID,
            "symbol": symbol,
            "exchange": "CRYPTO" if crypto else "NASDAQ",
            "asset_class": "crypto" if crypto else "us_equity",
            "avg_entry_price": "81000.5",
            "qty": "0.25",
            "side": "long",
            "cost_basis": "20250.125",
            **overrides,
        }
    )


def account(**overrides: Any) -> TradeAccount:
    return TradeAccount.model_validate(
        {
            "id": ASSET_ID,
            "account_number": "PA000000",
            "status": "ACTIVE",
            "cash": "100000",
            "non_marginable_buying_power": "99000",
            "buying_power": "400000",
            "equity": "100500",
            **overrides,
        }
    )


def trade_update(event: str = "new", **overrides: Any) -> TradeUpdate:
    fields: dict[str, Any] = {"event": event, "order": order(), "timestamp": NOW, **overrides}
    return TradeUpdate.model_validate(fields)


def quote(symbol: str = "BTC/USD") -> Quote:
    raw = {"t": NOW.isoformat(), "bp": 81000.1, "bs": 1.5, "ap": 81001.9, "as": 2.25}
    return Quote(symbol, {**raw, "bx": "", "ax": ""})


def book(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    reset: bool,
    symbol: str = "BTC/USD",
) -> Orderbook:
    def levels(found: list[tuple[float, float]]) -> list[dict[str, float]]:
        return [{"p": price, "s": size} for price, size in found]

    raw = {"t": NOW.isoformat(), "b": levels(bids), "a": levels(asks), "r": reset}
    return Orderbook(symbol, raw)


def trade(symbol: str = "BTC/USD", size: float = 0.01) -> Trade:
    return Trade(symbol, {"t": NOW.isoformat(), "p": 81000.5, "s": size, "i": 42, "x": ""})


def bar(symbol: str = "BTC/USD") -> Bar:
    return Bar(symbol, RAW_BAR)


def api_error(status: int, message: str = "boom") -> APIError:
    response = requests.Response()
    response.status_code = status
    return APIError(f'{{"message": "{message}"}}', requests.HTTPError(response=response))
