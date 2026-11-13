from typing import NamedTuple

import numpy as np
import pandas as pd

from app.features.scanner import universe
from app.features.scanner.features import Features

SCORES = ("mom_12_1", "mom_6_1", "clenow_90", "rank_avg", "resmom_12_1")
TOP_N = (10, 20)
MIN_HISTORY = 253
MAX_GAP = 0.15
CAP = 0.10
CAP_PASSES = 5


class Variant(NamedTuple):
    score: str
    top_n: int

    @property
    def label(self) -> str:
        return f"{self.score} top {self.top_n}"


VARIANTS = tuple(Variant(score, top_n) for score in SCORES for top_n in TOP_N)


def eligible(features: Features) -> pd.DataFrame:
    """Per date and symbol whether the name may be held: trending, liquid, seasoned, no wild gap."""
    return (
        features.trend
        & (features.close >= universe.MIN_PRICE)
        & (features.adv20 >= universe.MIN_DOLLAR_VOLUME)
        & (features.history >= MIN_HISTORY)
        & (features.gap90 <= MAX_GAP)
        & features.mom_12_1.notna()
    )


def scores(features: Features, ok: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every score, with the rank average taken among the eligible names of each date."""
    ranked = [
        frame.where(ok).rank(axis=1, pct=True) for frame in (features.mom_12_1, features.mom_6_1)
    ]
    return {
        "mom_12_1": features.mom_12_1,
        "mom_6_1": features.mom_6_1,
        "clenow_90": features.clenow_90,
        "rank_avg": (ranked[0] + ranked[1]) / 2,
        "resmom_12_1": features.resmom_12_1,
    }


def select(score: pd.DataFrame, ok: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """The `top_n` best-scoring eligible names of each date."""
    return score.where(ok).rank(axis=1, ascending=False, method="first") <= top_n


def capped(weights: pd.DataFrame, cap: float = CAP) -> pd.DataFrame:
    """Weights limited to `cap` each, the excess shared among the names still under it."""
    for _ in range(CAP_PASSES):
        excess = (weights - cap).clip(lower=0).sum(axis=1)
        weights = weights.clip(upper=cap)
        room = weights.where(weights < cap)
        share = room.div(room.sum(axis=1), axis=0).mul(excess, axis=0)
        weights = weights + share.fillna(0.0)
    return weights.clip(upper=cap)


def weights(chosen: pd.DataFrame, volatility: pd.DataFrame, cap: float = CAP) -> pd.DataFrame:
    """Chosen names weighted by inverse volatility, no name above `cap`; what is left is cash."""
    inverse = (1 / volatility).where(chosen)
    inverse = inverse.where(np.isfinite(inverse))
    return capped(inverse.div(inverse.sum(axis=1), axis=0).fillna(0.0), cap)


def last_of(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    """True on the final session of each month ("M") or week ("W")."""
    dates = pd.Series(index, index=index)
    return dates.groupby(index.to_period(frequency)).transform("max") == dates


def targets(picks: pd.DataFrame, ok: pd.DataFrame) -> pd.DataFrame:
    """Holdings decided at each close: re-ranked at month end, weekly sale of names that fail.

    Weight freed by a weekly sale is not given to the others; it waits as cash for the next
    month-end ranking.
    """
    month_end = last_of(picks.index, "M")
    week_end = last_of(picks.index, "W")
    base = picks[month_end.to_numpy()].reindex(picks.index).ffill().fillna(0.0)
    segment = month_end.cumsum().to_numpy()
    broken = (~ok) & week_end.to_numpy()[:, None]
    alive = (~broken).astype(float).groupby(segment).cummin()
    return base * alive
