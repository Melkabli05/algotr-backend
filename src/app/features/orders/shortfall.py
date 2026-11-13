import math
from dataclasses import dataclass

from app.quant.volume import Profile

RISK_UNIT = 1e-6
TEMPORARY_SHARE = 0.01
PERMANENT_SHARE = 0.1


@dataclass(frozen=True)
class Market:
    """What the trajectory needs of a stock: volatility, volume and spread."""

    sigma: float
    volume: float
    spread: float


def market_of(profile: Profile, spread: float) -> Market:
    """A stock's volatility per root trading day and daily volume, from its profile, and spread."""
    sigma = profile.step * math.sqrt(profile.session.buckets)
    return Market(sigma=sigma, volume=profile.volume, spread=spread)


def held(kappa: float, total: float, at: float) -> float:
    """sinh(kappa (total - at)) / sinh(kappa total), without overflowing for a large kappa."""
    if kappa == 0:
        return (total - at) / total
    return (
        math.exp(-kappa * at)
        * math.expm1(-2 * kappa * (total - at))
        / math.expm1(-2 * kappa * total)
    )


def decay(risk: float, days: float, count: int, market: Market) -> float:
    """Kappa of Almgren and Chriss: how fast the position is worked down, per trading day."""
    step = days / count
    temporary = market.spread / (TEMPORARY_SHARE * market.volume)
    permanent = market.spread / (PERMANENT_SHARE * market.volume)
    effective = temporary - permanent * step / 2
    squared = risk * RISK_UNIT * market.sigma**2 / effective
    return math.acosh(step**2 * squared / 2 + 1) / step


def trajectory(qty: float, days: float, count: int, risk: float, market: Market) -> list[float]:
    """The quantity still held at each of `count` + 1 times, from `qty` down to nothing."""
    kappa = decay(risk, days, count, market)
    return [qty * held(kappa, days, days * index / count) for index in range(count + 1)]


def weights(qty: float, days: float, count: int, risk: float, market: Market) -> list[float]:
    """The quantity to trade in each of `count` intervals."""
    path = trajectory(qty, days, count, risk, market)
    return [before - after for before, after in zip(path, path[1:], strict=False)]
