from datetime import date

import numpy as np
import pandas as pd

from app.core.drift import ReturnStream

START = date(2022, 1, 3)
BASE = 500
LIVE = 200


def series(
    live: tuple[float, float] | None = None,
    *,
    base: int = BASE,
    days: int = LIVE,
    seed: int = 1,
    name: str = "halal-etf-trend",
    kind: str = "strategy",
) -> ReturnStream:
    """A calm baseline of `base` days, then `days` more with the (mean, volatility) given."""
    rng = np.random.default_rng(seed)
    mean, spread = live or (0.0005, 0.01)
    values = np.concatenate([rng.normal(0.0005, 0.01, base), rng.normal(mean, spread, days)])
    index = pd.bdate_range(START, periods=len(values))
    return ReturnStream(name, kind, pd.Series(values, index=index))


def steps(
    segments: list[tuple[int, float]],
    *,
    base: int = BASE,
    seed: int = 1,
    name: str = "halal-etf-trend",
) -> ReturnStream:
    """A calm baseline, then stretches of the given length at a multiple of its volatility."""
    rng = np.random.default_rng(seed)
    parts = [rng.normal(0.0005, 0.01, base)]
    parts += [rng.normal(0.0005, 0.01 * factor, days) for days, factor in segments]
    values = np.concatenate(parts)
    return ReturnStream(
        name, "strategy", pd.Series(values, index=pd.bdate_range(START, periods=len(values)))
    )


class Source:
    def __init__(self, *streams: ReturnStream) -> None:
        self.found = list(streams)
        self.calls = 0

    def streams(self) -> list[ReturnStream]:
        self.calls += 1
        return self.found
