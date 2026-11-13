from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient

from app.core.settings import Settings

SETTINGS = Settings(alpaca_api_key="key", alpaca_api_secret="secret", alpaca_paper=True)
LIVE_SETTINGS = Settings(alpaca_api_key="key", alpaca_api_secret="secret", alpaca_paper=False)
ACCOUNT = {
    "status": "ACTIVE",
    "currency": "USD",
    "equity": "100000",
    "last_equity": "100000",
    "cash": "100000",
    "buying_power": "100000",
    "trading_blocked": False,
}


def closes(symbols: list[str], daily_move: float, days: int = 400, seed: int = 0) -> pd.DataFrame:
    """Prices drifting `daily_move` per day with a little noise, ending yesterday."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=days)
    frame = {}
    for offset, symbol in enumerate(symbols):
        noise = rng.normal(0, 0.002, days)
        frame[symbol] = (100 + 10 * offset) * np.exp(np.cumsum(daily_move + noise))
    return pd.DataFrame(frame, index=index)


def uptrend(symbols: list[str], days: int = 400) -> pd.DataFrame:
    return closes(symbols, 0.003, days)


def downtrend(symbols: list[str], days: int = 400) -> pd.DataFrame:
    return closes(symbols, -0.003, days)


def position(symbol: str, qty: float, price: float) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "side": "long",
        "qty": str(qty),
        "avg_entry_price": str(price),
        "current_price": str(price),
        "market_value": str(qty * price),
        "unrealized_pl": "0",
    }


def trading_with(positions: list[dict[str, Any]] | None = None, /, **account: Any) -> MagicMock:
    mock: MagicMock = create_autospec(TradingClient, instance=True)
    mock.get_account.return_value = {**ACCOUNT, **account}
    mock.get_all_positions.return_value = positions or []
    mock.get_orders.return_value = []
    mock.submit_order.side_effect = lambda request: {
        "id": f"order-{request.symbol}",
        "symbol": request.symbol,
        "side": request.side.value,
        "type": "market",
        "status": "accepted",
        "qty": str(request.qty),
        "filled_qty": "0",
        "filled_avg_price": None,
        "submitted_at": None,
    }
    return mock


def serve(monkeypatch: Any, closes: Callable[..., pd.DataFrame]) -> None:
    """Serve the daily closes `closes` returns, each day opening at the close before it."""
    from app.quant import history

    def daily_prices(
        stocks: object, crypto: object, symbols: list[str], years: float
    ) -> history.Prices:
        frame = closes(stocks, crypto, symbols, years)
        return history.Prices(frame, frame.shift(1).fillna(frame))

    monkeypatch.setattr(history, "daily_prices", daily_prices)


def fake_history(monkeypatch: Any, **direction: str) -> dict[str, str]:
    """Serve synthetic daily closes; `direction` maps a symbol to "up" or "down" (default up)."""

    directions = {"default": "up", **direction}

    def daily_closes(_s: object, _c: object, symbols: list[str], _y: float) -> pd.DataFrame:
        parts = [
            (downtrend if directions.get(symbol, directions["default"]) == "down" else uptrend)(
                [symbol], 1200
            )
            for symbol in symbols
        ]
        return pd.concat(parts, axis=1)

    serve(monkeypatch, daily_closes)
    return directions
