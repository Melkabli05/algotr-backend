from collections.abc import Sequence
from datetime import date
from typing import Protocol

import pandas as pd

ENSEMBLE = "ai-ensemble"


class AgentSignal(Protocol):
    """What the autopilot may ask of the AI agents' UCB1 ensemble."""

    def ensemble_eligible(self) -> bool:
        """Whether the latest evaluation found the ensemble passing every gate."""

    def ensemble_target(self) -> dict[str, float] | None:
        """The weights over its assets for tomorrow, cash left out; None until it has a record."""

    def ensemble_returns(self) -> pd.Series | None:
        """Its daily returns so far, for sizing it against the other strategies."""


class MarketMood(Protocol):
    def mood(self, sessions: Sequence[date]) -> pd.DataFrame:
        """Per session, the mean headline sentiment and risk of the screened market, or NaN."""
