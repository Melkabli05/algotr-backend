import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import pandas as pd

UNCAPPED = ("Fund", "Crypto", "Unclassified")
TRADING_STAGES = ("canary", "live")
MIN_TRADE_VALUE = 10.0
COIN_STEP = 1e-6
TARGETS_TOPIC = "arbiter.targets"


@dataclass(frozen=True)
class StrategyTarget:
    """What one strategy wants held, as shares of the capital it was given; never an order."""

    strategy: str
    stage: str
    allocation: float
    weights: Mapping[str, float]
    version_id: str | None = None
    session: str | None = None
    closes: Mapping[str, list[float]] = field(default_factory=dict)
    inputs: str = "closes"


@dataclass(frozen=True)
class Limits:
    sector_share: float
    correlation: float
    gross: float = 1.0
    net: float = 1.0


@dataclass
class Arbitration:
    weights: dict[str, float]
    notes: list[str]
    orders: list[dict[str, object]]


def combine(targets: Iterable[StrategyTarget]) -> dict[str, float]:
    """The portfolio's weight in each symbol: each strategy's wish times its allocation."""
    found: dict[str, float] = {}
    for target in targets:
        for symbol, weight in target.weights.items():
            found[symbol] = found.get(symbol, 0.0) + target.allocation * weight
    return {symbol: weight for symbol, weight in found.items() if weight}


def scale(weights: dict[str, float], symbols: Iterable[str], cap: float) -> float:
    """Cut the named symbols together to `cap`; returns the factor applied."""
    names = list(symbols)
    total = sum(weights[name] for name in names)
    if total <= cap or total <= 0:
        return 1.0
    factor = cap / total
    for name in names:
        weights[name] *= factor
    return factor


def clusters(returns: pd.DataFrame, limit: float) -> list[set[str]]:
    """Groups of symbols linked by a correlation above `limit`, two or more each."""
    if returns.shape[1] < 2:
        return []
    matrix = returns.corr()
    parent = {name: name for name in matrix.columns}

    def root(name: str) -> str:
        while parent[name] != name:
            name = parent[name]
        return name

    for left in matrix.columns:
        for right in matrix.columns:
            if left < right and matrix.loc[left, right] > limit:
                parent[root(right)] = root(left)
    groups: dict[str, set[str]] = {}
    for name in matrix.columns:
        groups.setdefault(root(name), set()).add(name)
    return [group for group in groups.values() if len(group) > 1]


def constrain(
    weights: Mapping[str, float],
    limits: Limits,
    sector_of: Callable[[str], str],
    returns: pd.DataFrame | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Apply the portfolio's limits no strategy can override: sector, correlation, gross, net.

    Funds, coins and unclassified symbols carry no sector cap, as in the pre-trade checks, and a
    correlated group made only of them is not capped either.
    """
    found = dict(weights)
    notes: list[str] = []
    sectors: dict[str, list[str]] = {}
    for symbol in found:
        sectors.setdefault(sector_of(symbol), []).append(symbol)
    for sector, names in sorted(sectors.items()):
        if sector in UNCAPPED:
            continue
        factor = scale(found, names, limits.sector_share)
        if factor < 1:
            notes.append(f"{sector} cut to {limits.sector_share:.0%} of the portfolio")
    if returns is not None:
        held = [name for name in found if name in returns.columns]
        for group in clusters(returns[held].dropna(), limits.correlation):
            capped = [name for name in group if sector_of(name) not in UNCAPPED]
            if not capped:
                continue
            factor = scale(found, capped, limits.sector_share)
            if factor < 1:
                shown = ", ".join(sorted(capped))
                notes.append(
                    f"{shown} move together above {limits.correlation:.2f}; cut to "
                    f"{limits.sector_share:.0%} together"
                )
    gross = sum(abs(weight) for weight in found.values())
    if gross > limits.gross:
        factor = limits.gross / gross
        found = {name: weight * factor for name, weight in found.items()}
        notes.append(f"Gross exposure cut from {gross:.0%} to {limits.gross:.0%}")
    net = sum(found.values())
    if net > limits.net:
        factor = limits.net / net
        found = {name: weight * factor for name, weight in found.items()}
        notes.append(f"Net exposure cut from {net:.0%} to {limits.net:.0%}")
    return found, notes


def quantity(symbol: str, value: float, price: float) -> float:
    """Whole shares of a stock, coins to a millionth, always rounded towards zero."""
    raw = value / price
    if "/" in symbol:
        return math.floor(abs(raw) / COIN_STEP) * COIN_STEP * (1 if raw >= 0 else -1)
    return float(math.floor(abs(raw)) * (1 if raw >= 0 else -1))


def net_orders(
    weights: Mapping[str, float],
    equity: float,
    held: Mapping[str, float],
    prices: Mapping[str, float],
) -> list[dict[str, object]]:
    """The one set of orders from today's holdings to the portfolio's targets, sells first."""
    orders: list[dict[str, object]] = []
    for symbol in sorted(set(weights) | set(held)):
        price = prices.get(symbol)
        if not price:
            continue
        current = held.get(symbol, 0.0)
        change = quantity(symbol, weights.get(symbol, 0.0) * equity - current * price, price)
        if weights.get(symbol, 0.0) == 0 and current > 0:
            change = -current
        if abs(change) * price < MIN_TRADE_VALUE or change == 0:
            continue
        side = "buy" if change > 0 else "sell"
        orders.append({"symbol": symbol, "side": side, "qty": round(abs(change), 9)})
    return sorted(orders, key=lambda order: order["side"] != "sell")


def returns_of(targets: Iterable[StrategyTarget]) -> pd.DataFrame:
    """Daily returns of every symbol the strategies sent closes for, aligned at the end."""
    series: dict[str, pd.Series] = {}
    for target in targets:
        for symbol, closes in target.closes.items():
            if len(closes) > 1 and symbol not in series:
                series[symbol] = pd.Series(closes).pct_change().dropna().reset_index(drop=True)
    if not series:
        return pd.DataFrame()
    length = min(len(values) for values in series.values())
    return pd.DataFrame({name: values.iloc[-length:].to_numpy() for name, values in series.items()})


def arbitrate(
    targets: Iterable[StrategyTarget],
    limits: Limits,
    sector_of: Callable[[str], str],
    equity: float,
    held: Mapping[str, float],
    prices: Mapping[str, float],
    cut: float = 1.0,
) -> Arbitration:
    """From the trading strategies' targets to one constrained portfolio and its net orders.

    `cut` scales every position after the limits: the portfolio breaker halves them.
    """
    trading = [target for target in targets if target.stage in TRADING_STAGES]
    weights, notes = constrain(combine(trading), limits, sector_of, returns_of(trading))
    if cut != 1.0:
        weights = {symbol: weight * cut for symbol, weight in weights.items()}
        notes.append(f"Portfolio breaker open: every position at {cut:.0%}")
    return Arbitration(weights, notes, net_orders(weights, equity, held, prices))
