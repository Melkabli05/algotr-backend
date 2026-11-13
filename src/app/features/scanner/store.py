from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd

from app.features.scanner.bars import empty

PATH = Path("data/scanner/bars.parquet")
TOLERANCE = 1e-4


class BarStore:
    """The daily-bar panel on disk: one Parquet file in long layout, ordered by symbol then date."""

    def __init__(self, path: Path = PATH) -> None:
        self.path = path

    def read(self, symbols: Collection[str] | None = None) -> pd.DataFrame:
        """The whole panel, or only the rows of `symbols`, which Parquet filters while reading."""
        if not self.path.exists():
            return empty()
        filters = None if symbols is None else [("symbol", "in", sorted(symbols))]
        return pd.read_parquet(self.path, filters=filters)

    def write(self, panel: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        scratch = self.path.with_suffix(".parquet.tmp")
        panel.to_parquet(scratch, index=False)
        scratch.replace(self.path)


def diverged(stored: pd.DataFrame, recent: pd.DataFrame, tolerance: float = TOLERANCE) -> set[str]:
    """Symbols whose stored history cannot be extended: adjusted closes moved, or a gap opened.

    Alpaca rescales past prices when a split or dividend happens, so a symbol whose recent
    window no longer matches what was stored needs its whole history fetched again.
    """
    if stored.empty or recent.empty:
        return set()
    window = stored.loc[stored["date"] >= recent["date"].min(), ["symbol", "date", "close"]]
    both = window.merge(
        recent[["symbol", "date", "close"]], on=["symbol", "date"], suffixes=("_old", "_new")
    )
    moved = ~np.isclose(both["close_new"], both["close_old"], rtol=tolerance, atol=0.0)
    changed = set(both.loc[moved, "symbol"])
    gaps = set(stored["symbol"].unique()) & set(recent["symbol"].unique()) - set(both["symbol"])
    return changed | gaps


def combine(
    stored: pd.DataFrame, recent: pd.DataFrame, history: pd.DataFrame, stale: Collection[str] = ()
) -> pd.DataFrame:
    """The panel brought up to date: `stale` and refetched symbols replaced, the rest extended."""
    dropped = {*stale, *history["symbol"].unique()}
    kept = stored[~stored["symbol"].isin(dropped)]
    latest = kept.groupby("symbol")["date"].max().rename("latest")
    fresh = recent.merge(latest, left_on="symbol", right_index=True)
    fresh = fresh[fresh["date"] > fresh.pop("latest")]
    parts = [part for part in (kept, fresh, history) if not part.empty]
    if not parts:
        return empty()
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"], ignore_index=True)
