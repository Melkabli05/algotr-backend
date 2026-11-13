from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import empyrical
import numpy as np
import pandas as pd
from pydantic import BaseModel

FOLDS = 5
FOLD_SHARPE = 1.2
DIVERGENCE = 0.05
DECISION_DAYS = 30
SHADOW_DAYS = 30
SHADOW_SHARPE = 0.8
SHADOW_DRAWDOWN = 0.20
CANARY_DAYS = 60
CANARY_TOLERANCE = 0.20


class Gate(BaseModel):
    name: str
    passed: bool
    detail: str
    value: float | None = None


@dataclass(frozen=True)
class Track:
    """A strategy's returns over a stretch of days and how they are annualised."""

    returns: pd.Series
    periods_per_year: int

    @property
    def days(self) -> int:
        return len(self.returns)

    def sharpe(self) -> float:
        value = float(empyrical.sharpe_ratio(self.returns, annualization=self.periods_per_year))
        return value if np.isfinite(value) else 0.0

    def drawdown(self) -> float:
        value = float(empyrical.max_drawdown(self.returns)) if self.days else 0.0
        return -value if np.isfinite(value) else 0.0


def fold_sharpes(track: Track, folds: int = FOLDS) -> list[float]:
    """The Sharpe of each of `folds` consecutive, equal stretches of the history."""
    parts = np.array_split(np.arange(track.days), folds)
    return [
        Track(track.returns.iloc[part], track.periods_per_year).sharpe()
        for part in parts
        if len(part)
    ]


def research_gate(track: Track) -> Gate:
    """RESEARCH to VALIDATED: every walk-forward fold clears the Sharpe bar."""
    sharpes = fold_sharpes(track)
    worst = min(sharpes) if sharpes else 0.0
    shown = ", ".join(f"{value:.2f}" for value in sharpes)
    return Gate(
        name=f"Sharpe above {FOLD_SHARPE} in all {FOLDS} folds",
        passed=len(sharpes) == FOLDS and worst > FOLD_SHARPE,
        detail=f"Fold Sharpes {shown}",
        value=worst,
    )


def divergence(
    recorded: Mapping[date, Mapping[str, float]], replayed: Mapping[date, Mapping[str, float]]
) -> float | None:
    """How far the decisions taken live sit from the same rule replayed on today's history.

    Half the summed absolute weight difference, averaged over the days both cover: 0 is identical,
    1 is entirely different holdings. None when they share no day.
    """
    days = sorted(set(recorded) & set(replayed))
    if not days:
        return None
    gaps = []
    for day in days:
        live, again = recorded[day], replayed[day]
        names = set(live) | set(again)
        gaps.append(sum(abs(live.get(n, 0.0) - again.get(n, 0.0)) for n in names) / 2)
    return float(np.mean(gaps))


def validation_gate(
    recorded: Mapping[date, Mapping[str, float]], replayed: Mapping[date, Mapping[str, float]]
) -> Gate:
    """VALIDATED to SHADOW: a month of live decisions the backtest reproduces within 5%."""
    gap = divergence(recorded, replayed)
    enough = len(recorded) >= DECISION_DAYS
    passed = enough and gap is not None and gap < DIVERGENCE
    detail = (
        f"{len(recorded)} of {DECISION_DAYS} days recorded"
        if not enough
        else f"Live and replayed decisions differ by {gap:.1%}"
        if gap is not None
        else "No replayed day to compare"
    )
    return Gate(
        name=f"Live decisions match the backtest within {DIVERGENCE:.0%}",
        passed=passed,
        detail=detail,
        value=gap,
    )


def shadow_gate(track: Track) -> list[Gate]:
    """SHADOW to CANARY: thirty days on paper, no drawdown breach, Sharpe above 0.8."""
    sharpe, drawdown = track.sharpe(), track.drawdown()
    return [
        Gate(
            name=f"{SHADOW_DAYS} days in shadow",
            passed=track.days >= SHADOW_DAYS,
            detail=f"{track.days} days",
            value=float(track.days),
        ),
        Gate(
            name=f"No drawdown past {SHADOW_DRAWDOWN:.0%}",
            passed=drawdown < SHADOW_DRAWDOWN,
            detail=f"Worst drawdown {drawdown:.1%}",
            value=drawdown,
        ),
        Gate(
            name=f"Shadow Sharpe above {SHADOW_SHARPE}",
            passed=sharpe > SHADOW_SHARPE,
            detail=f"Sharpe {sharpe:.2f}",
            value=sharpe,
        ),
    ]


def canary_gate(track: Track, shadow_sharpe: float | None) -> list[Gate]:
    """CANARY to LIVE: sixty days at a tenth of the capital, Sharpe within 20% of shadow."""
    sharpe = track.sharpe()
    near = shadow_sharpe is not None and abs(sharpe - shadow_sharpe) <= CANARY_TOLERANCE * abs(
        shadow_sharpe
    )
    reference = "no shadow Sharpe on record" if shadow_sharpe is None else f"{shadow_sharpe:.2f}"
    return [
        Gate(
            name=f"{CANARY_DAYS} days in canary",
            passed=track.days >= CANARY_DAYS,
            detail=f"{track.days} days",
            value=float(track.days),
        ),
        Gate(
            name=f"Sharpe within {CANARY_TOLERANCE:.0%} of shadow",
            passed=near,
            detail=f"Canary Sharpe {sharpe:.2f} against shadow {reference}",
            value=sharpe,
        ),
    ]
