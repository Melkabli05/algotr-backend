from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from threading import Lock
from typing import NamedTuple

from app.features.scanner import lists, scan, universe
from app.features.scanner.store import BarStore


class Market(NamedTuple):
    analysis: scan.Analysis
    names: dict[str, str]
    lists: dict[str, list[str]]
    stale: list[str]
    versions: list[str]


class MarketView:
    """The market on its last stored session, rebuilt only when its inputs or the day change."""

    def __init__(self, store: BarStore, root: Path, sources: Sequence[lists.Source]) -> None:
        self.store = store
        self.root = root
        self.sources = sources
        self.lock = Lock()
        self.cache: tuple[tuple[int, date], Market] | None = None

    def stamp(self) -> int:
        """The newest change among the panel, the names and the list files."""
        files = [self.store.path, self.root / universe.NAMES, *(self.root / "lists").glob("*.csv")]
        return max(path.stat().st_mtime_ns for path in files if path.exists())

    def get(self) -> Market | None:
        if not self.store.path.exists():
            return None
        today = datetime.now(UTC).date()
        stamp = (self.stamp(), today)
        with self.lock:
            if self.cache is None or self.cache[0] != stamp:
                panel = self.store.read()
                if panel.empty:
                    return None
                snapshots = lists.current(self.sources, self.root / "lists")
                members = lists.listed(snapshots, set(panel["symbol"].unique()), today)
                stale = [s.name for s in snapshots if (today - s.as_of) > lists.STALE_AFTER]
                self.cache = (
                    stamp,
                    Market(
                        scan.analyse(panel, members),
                        universe.load_names(self.root),
                        {snapshot.name: snapshot.symbols for snapshot in snapshots},
                        stale,
                        lists.versions(snapshots),
                    ),
                )
            return self.cache[1]
