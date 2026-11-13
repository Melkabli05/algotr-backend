from typing import Any

from alpaca.data.models import Orderbook

AT = "2026-11-03T16:00:00.500000Z"
BIDS = [(100.0, 1.0), (99.5, 2.0), (99.0, 4.0), (98.0, 8.0)]
ASKS = [(100.5, 1.0), (101.0, 2.0), (102.0, 4.0), (104.0, 8.0)]


def levels(pairs: list[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"p": price, "s": size} for price, size in pairs]


def message(
    symbol: str = "BTC/USD",
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
    reset: bool = False,
    at: str = AT,
) -> Orderbook:
    raw: dict[str, Any] = {"t": at, "b": levels(bids or []), "a": levels(asks or []), "r": reset}
    return Orderbook(symbol, raw)


def snapshot(symbol: str = "BTC/USD") -> Orderbook:
    return message(symbol, BIDS, ASKS, reset=True)
