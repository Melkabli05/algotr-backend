from typing import NamedTuple, Protocol

import pandas as pd


class ReturnStream(NamedTuple):
    """The daily returns of one thing to watch: a strategy, an agent or the ensemble."""

    name: str
    kind: str
    returns: pd.Series


class ReturnSource(Protocol):
    def streams(self) -> list[ReturnStream]:
        """The daily returns of what this part of the platform runs, oldest first."""


class SizingGuard(Protocol):
    def scale(self, name: str) -> float:
        """The multiplier on a strategy's positions: 1 normally, less while it has drifted."""


class StageGate(Protocol):
    def trading(self) -> list[str]:
        """The strategies whose lifecycle stage lets them trade."""

    def capital(self, name: str) -> float:
        """The share of its allocation a strategy's stage lets it trade with."""
