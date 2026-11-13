import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from cachetools import LRUCache

from app.features.systematic.ensemble import Replay
from app.quant import regimes as markov

MIN_FIT = 250
CONFIDENT = 0.6
TOP_K = 2
MIN_REGIME_DAYS = 20
PERIODS = 252


@dataclass(frozen=True)
class Choice:
    weights: dict[str, float]
    regime: str | None
    confidence: float
    reason: str


FITS: LRUCache[tuple[object, ...], markov.Regimes] = LRUCache(maxsize=256)


def fitted(level: pd.Series) -> markov.Regimes:
    """The platform's regime model (`app.quant.regimes`) fitted on this exact history, kept:
    every seed and the latest choice reuse it."""
    values = level.to_numpy()
    key = (len(values), level.index[-1], hashlib.sha256(values.tobytes()).hexdigest())
    if key not in FITS:
        FITS[key] = markov.fit(level)
    return FITS[key]


def competence(returns: pd.DataFrame, labels: pd.Series, regime: str) -> dict[str, float]:
    """Each agent's annualised Sharpe on the past days of `regime`, where there are enough."""
    days = labels.index[labels == regime].intersection(returns.index)
    if len(days) < MIN_REGIME_DAYS:
        return {}
    found = {}
    for name in returns:
        values = returns.loc[days, name]
        spread = values.std()
        found[str(name)] = float(values.mean() / spread * np.sqrt(PERIODS)) if spread > 0 else 0.0
    return found


def select(returns: pd.DataFrame, labels: pd.Series, regime: str, confidence: float) -> Choice:
    """The agents competent in the current regime, weighted by that competence; equal weights
    at a smaller size when the regime is unclear; cash when nobody is competent in it."""
    names = [str(name) for name in returns]
    if confidence < CONFIDENT:
        share = confidence / len(names)
        reason = f"Regime unclear ({regime} at {confidence:.0%}): equal weights, smaller"
        return Choice(dict.fromkeys(names, share), regime, confidence, reason)
    scores = competence(returns, labels, regime)
    if not scores:
        reason = f"Too few {regime} days on record: equal weights"
        return Choice(dict.fromkeys(names, 1 / len(names)), regime, confidence, reason)
    chosen = sorted(
        (name for name in names if scores.get(name, 0.0) > 0), key=lambda n: -scores[n]
    )[:TOP_K]
    if not chosen:
        return Choice(dict.fromkeys(names, 0.0), regime, confidence, f"No agent earns in {regime}")
    total = sum(scores[name] for name in chosen)
    weights = {name: scores[name] / total if name in chosen else 0.0 for name in names}
    reason = f"{regime} at {confidence:.0%}: " + ", ".join(
        f"{name} Sharpe {scores[name]:.2f}" for name in chosen
    )
    return Choice(weights, regime, confidence, reason)


def choose(agents: pd.DataFrame, market: pd.Series, before: pd.Timestamp) -> Choice:
    """The choice for the period starting at `before`: the regime model fitted on the market
    before it, each past day labelled by its filtered regime, nothing after `before` used."""
    known = market.loc[market.index < before].dropna()
    past = agents.loc[agents.index < before]
    names = [str(name) for name in agents]
    if len(known) < MIN_FIT:
        return Choice(dict.fromkeys(names, 1 / len(names)), None, 0.0, "Too little history")
    found = fitted((1 + known).cumprod())
    last = found.probabilities.iloc[-1]
    return select(past, found.labels(), str(last.idxmax()), float(last.max()))


def replay(agents: pd.DataFrame, market: pd.Series) -> Replay:
    """The regime-aware ensemble over the agents' daily returns, chosen anew each month from
    only the months before; the weights left over are cash."""
    chosen: dict[pd.Period, list[float]] = {}
    parts: list[pd.Series] = []
    for month, block in agents.groupby(agents.index.to_period("M")):
        choice = choose(agents, market, block.index[0])
        mix = [choice.weights[str(name)] for name in agents]
        chosen[month] = mix
        parts.append(block @ np.array(mix))
    after = agents.index[-1] + pd.Timedelta(days=1) if len(agents) else pd.Timestamp.now()
    latest = choose(agents, market, after)
    return Replay(
        pd.concat(parts) if parts else pd.Series(dtype=float),
        pd.DataFrame(chosen, index=list(agents)).T,
        latest.weights,
    )
