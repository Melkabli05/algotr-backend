import math
from datetime import UTC, datetime
from statistics import NormalDist
from typing import Any, Literal

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from pydantic import BaseModel

from app.features.orders.shortfall import PERMANENT_SHARE, market_of
from app.quant import history, volume

BPS = 1e4
IMPACT_RANGE = (0.5, 1.5)
NORMAL = NormalDist()


class Estimate(BaseModel):
    """What an order should cost against the mid, and for a resting limit, its chance to trade."""

    price: float
    basis: Literal["book", "model", "none"]
    marketable: bool
    spread_bps: float | None
    impact_bps: tuple[float, float] | None
    permanent_bps: float | None
    total_bps: tuple[float, float] | None
    total_dollars: tuple[float, float] | None
    participation: float | None
    fill_probability: float | None
    horizon: str | None
    note: str


def touch_probability(distance: float, sigma: float, days: float) -> float:
    """Chance a driftless random walk with daily deviation `sigma` reaches `distance` in `days`."""
    if distance <= 0:
        return 1.0
    if sigma <= 0 or days <= 0:
        return 0.0
    return min(1.0, 2 * (1 - NORMAL.cdf(distance / (sigma * math.sqrt(days)))))


def square_root_bps(sigma_share: float, participation: float) -> tuple[float, float]:
    """Temporary impact by the square-root law, with its constant between 0.5 and 1.5."""
    core = sigma_share * math.sqrt(max(participation, 0.0)) * BPS
    return core * IMPACT_RANGE[0], core * IMPACT_RANGE[1]


def session_left(profile: volume.Profile, now: datetime) -> float:
    """The share of today's session still to come, for a day order; at least one bucket."""
    buckets = profile.session.buckets
    index = profile.session.bucket_of(now)
    return max(buckets - max(index, 0), 1) / buckets


def estimate(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    book: Any,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    limit: float | None,
    day_order: bool,
    now: datetime | None = None,
) -> Estimate:
    moment = now or datetime.now(UTC)
    sign = 1 if side == "buy" else -1
    marketable = limit is None or (limit - price) * sign >= 0
    impact_found = None
    if "/" in symbol and book is not None:
        try:
            impact_found = book.impact(symbol, side, qty)
        except Exception:
            impact_found = None
    try:
        curve = volume.profile(stocks, crypto, symbol, moment)
    except history.NotEnoughHistoryError:
        curve = None
    sigma_share = None
    participation = None
    spread = None
    if curve is not None:
        quoted = (
            impact_found.spread_bps * price / BPS
            if impact_found is not None
            else volume.spread(stocks, symbol, moment)
            if "/" not in symbol
            else None
        )
        if quoted is not None:
            spread = quoted
        market = market_of(curve, spread or 0.0)
        sigma_share = market.sigma / price if price else None
        participation = qty / market.volume if market.volume else None
    probability = None
    horizon = None
    if not marketable and curve is not None and sigma_share is not None and limit is not None:
        days = session_left(curve, moment) if day_order and "/" not in symbol else 1.0
        horizon = "rest of the session" if day_order and "/" not in symbol else "next 24 hours"
        probability = touch_probability(abs(price - limit) / price, sigma_share, days)
    if not marketable:
        return Estimate(
            price=price,
            basis="model" if probability is not None else "none",
            marketable=False,
            spread_bps=None,
            impact_bps=None,
            permanent_bps=None,
            total_bps=None,
            total_dollars=None,
            participation=participation,
            fill_probability=probability,
            horizon=horizon,
            note="A resting limit pays no spread; the chance is that the price touches it",
        )
    half = spread / 2 / price * BPS if spread else None
    if impact_found is not None and impact_found.slippage_bps is not None:
        walk = max(impact_found.slippage_bps - (half or 0.0), 0.0)
        impact = (walk, walk)
        basis: Literal["book", "model", "none"] = "book"
        permanent = None
        note = "Walked through the live order book"
    elif sigma_share is not None and participation is not None:
        impact = square_root_bps(sigma_share, participation)
        basis = "model"
        permanent = (
            (spread or 0.0) / (PERMANENT_SHARE * qty / participation) * qty / 2 / price * BPS
            if participation
            else None
        )
        note = "Square-root impact from 20 sessions of volatility and volume"
    else:
        return Estimate(
            price=price,
            basis="none",
            marketable=True,
            spread_bps=half,
            impact_bps=None,
            permanent_bps=None,
            total_bps=None,
            total_dollars=None,
            participation=None,
            fill_probability=None,
            horizon=None,
            note="Not enough history to estimate impact",
        )
    low = (half or 0.0) + impact[0] + (permanent or 0.0)
    high = (half or 0.0) + impact[1] + (permanent or 0.0)
    notional = qty * price
    return Estimate(
        price=price,
        basis=basis,
        marketable=True,
        spread_bps=half,
        impact_bps=impact,
        permanent_bps=permanent,
        total_bps=(low, high),
        total_dollars=(low * notional / BPS, high * notional / BPS),
        participation=participation,
        fill_probability=1.0,
        horizon=None,
        note=note,
    )
