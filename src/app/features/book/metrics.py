from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

THRESHOLD = 0.3
LEVEL_COUNTS = (5, 10, 25)
BAND_BPS = (10, 25, 50, 100)
BPS = 1e4
Pressure = Literal["buying", "selling", "balanced"]
Levels = Sequence[tuple[float, float]]
Best = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class Walk:
    """What an order of a given size finds in the book: how much it fills, at what average price."""

    filled: float
    average: float | None
    levels: int


def walk(levels: Levels, qty: float) -> Walk:
    """Take `qty` from `levels` (best first) and report the fill."""
    left, cost, used = qty, 0.0, 0
    for price, size in levels:
        take = min(size, left)
        if take <= 0:
            break
        cost += take * price
        left -= take
        used += 1
    filled = qty - left
    return Walk(filled, cost / filled if filled else None, used)


def slippage(side: str, average: float, mid: float) -> float:
    """How far an average fill price is from the mid in basis points, positive when it costs."""
    gap = average - mid if side == "buy" else mid - average
    return gap / mid * BPS


def max_size(levels: Levels, side: str, reference: float, bps: float) -> float:
    """The largest quantity whose average fill stays within `bps` of the `reference` price."""
    sign = 1.0 if side == "buy" else -1.0
    limit = sign * reference * (1 + sign * bps / BPS)
    qty = cost = 0.0
    for price, size in levels:
        signed = sign * price
        if signed <= limit:
            qty, cost = qty + size, cost + size * signed
            continue
        fits = max(0.0, (limit * qty - cost) / (signed - limit))
        if fits < size:
            return qty + fits
        qty, cost = qty + size, cost + size * signed
    return qty


def within(levels: Levels, mid: float, bps: int, side: str) -> tuple[float, float]:
    """The size and the value in quote currency of a side's levels within `bps` of the mid."""
    reach = bps / BPS
    near = [
        (price, size)
        for price, size in levels
        if (price >= mid * (1 - reach) if side == "bid" else price <= mid * (1 + reach))
    ]
    return sum(size for _, size in near), sum(price * size for price, size in near)


def imbalance(bid: float, ask: float) -> float | None:
    """(bid - ask) / (bid + ask), from +1 (all on the bid) to -1 (all on the ask); None if empty."""
    total = bid + ask
    return (bid - ask) / total if total > 0 else None


def order_flow(before: Best, after: Best) -> float:
    """The order flow imbalance of one change of the best bid and ask (Cont, Kukanov, Stoikov)."""
    (bid0, bid_size0), (ask0, ask_size0) = before
    (bid1, bid_size1), (ask1, ask_size1) = after
    flow = 0.0
    if bid1 >= bid0:
        flow += bid_size1
    if bid1 <= bid0:
        flow -= bid_size0
    if ask1 <= ask0:
        flow -= ask_size1
    if ask1 >= ask0:
        flow += ask_size0
    return flow


def pressure(value: float | None) -> Pressure | None:
    if value is None:
        return None
    if value > THRESHOLD:
        return "buying"
    return "selling" if value < -THRESHOLD else "balanced"


def by_levels(bids: Levels, asks: Levels, count: int) -> float | None:
    """The imbalance of the volume in the best `count` levels of each side."""
    return imbalance(sum(size for _, size in bids[:count]), sum(size for _, size in asks[:count]))


def by_band(bids: Levels, asks: Levels, mid: float, bps: int) -> float | None:
    """The imbalance of the volume within `bps` basis points of the mid on each side."""
    low, high = mid * (1 - bps / BPS), mid * (1 + bps / BPS)
    return imbalance(
        sum(size for price, size in bids if price >= low),
        sum(size for price, size in asks if price <= high),
    )
