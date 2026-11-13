import sqlite3
import zlib
from collections import defaultdict
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path("data/book")
KEPT_DAYS = 3
LEVEL = np.dtype([("price", "<f8"), ("size", "<f4")])
SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    symbol TEXT NOT NULL,
    at INTEGER NOT NULL,
    mid REAL NOT NULL,
    spread REAL NOT NULL,
    bids BLOB NOT NULL,
    asks BLOB NOT NULL,
    PRIMARY KEY (symbol, at)
) WITHOUT ROWID
"""
FLOWS = """
CREATE TABLE IF NOT EXISTS flows (
    symbol TEXT NOT NULL,
    at INTEGER NOT NULL,
    span REAL NOT NULL,
    ofi REAL NOT NULL,
    PRIMARY KEY (symbol, at)
) WITHOUT ROWID
"""


@dataclass(frozen=True)
class Sample:
    """One coin's book at one moment: the 25 best levels of each side as (price, size)."""

    at: datetime
    symbol: str
    mid: float
    spread: float
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


@dataclass(frozen=True)
class Flow:
    """The order flow imbalance of one coin over a window of `span` seconds ending at `at`."""

    at: datetime
    symbol: str
    span: float
    ofi: float


def pack(levels: Sequence[tuple[float, float]]) -> bytes:
    """Levels as exact prices and single-precision sizes, compressed."""
    found = np.empty(len(levels), dtype=LEVEL)
    found["price"] = [price for price, _ in levels]
    found["size"] = [size for _, size in levels]
    return zlib.compress(found.tobytes(), 1)


def unpack(blob: bytes) -> list[tuple[float, float]]:
    found = np.frombuffer(zlib.decompress(blob), dtype=LEVEL)
    return [
        (float(price), float(size))
        for price, size in zip(found["price"], found["size"], strict=True)
    ]


def millis(moment: datetime) -> int:
    return round(moment.timestamp() * 1000)


class BookStore:
    """Samples in one SQLite file for each UTC day, so keeping a few days is deleting files."""

    def __init__(self, root: Path = ROOT, kept_days: int = KEPT_DAYS) -> None:
        self.root = root
        self.kept_days = kept_days

    def path(self, day: date) -> Path:
        return self.root / f"{day:%Y-%m-%d}.sqlite"

    def connect(self, day: date) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path(day))
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(SCHEMA)
        db.execute(FLOWS)
        return db

    def add(self, samples: Sequence[Sample]) -> None:
        by_day: dict[date, list[Sample]] = defaultdict(list)
        for sample in samples:
            by_day[sample.at.astimezone(UTC).date()].append(sample)
        for day, rows in by_day.items():
            with closing(self.connect(day)) as db, db:
                db.executemany(
                    "INSERT OR REPLACE INTO samples VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (s.symbol, millis(s.at), s.mid, s.spread, pack(s.bids), pack(s.asks))
                        for s in rows
                    ],
                )

    def add_flows(self, flows: Sequence[Flow]) -> None:
        by_day: dict[date, list[Flow]] = defaultdict(list)
        for flow in flows:
            by_day[flow.at.astimezone(UTC).date()].append(flow)
        for day, rows in by_day.items():
            with closing(self.connect(day)) as db, db:
                db.executemany(
                    "INSERT OR REPLACE INTO flows VALUES (?, ?, ?, ?)",
                    [(f.symbol, millis(f.at), f.span, f.ofi) for f in rows],
                )

    def read_flows(self, symbol: str, since: datetime, until: datetime | None = None) -> list[Flow]:
        end = until or datetime.now(UTC)
        found: list[Flow] = []
        day = since.astimezone(UTC).date()
        while day <= end.astimezone(UTC).date():
            if self.path(day).exists():
                with closing(self.connect(day)) as db:
                    rows = db.execute(
                        "SELECT at, span, ofi FROM flows"
                        " WHERE symbol = ? AND at >= ? AND at <= ? ORDER BY at",
                        (symbol, millis(since), millis(end)),
                    ).fetchall()
                found += [
                    Flow(datetime.fromtimestamp(at / 1000, UTC), symbol, span, ofi)
                    for at, span, ofi in rows
                ]
            day += timedelta(days=1)
        return found

    def spreads(self, symbol: str, since: datetime) -> list[float]:
        """The spread of each stored second of `symbol` since `since`, in basis points."""
        found: list[float] = []
        day = since.astimezone(UTC).date()
        while day <= datetime.now(UTC).date():
            if self.path(day).exists():
                with closing(self.connect(day)) as db:
                    found += [
                        row[0]
                        for row in db.execute(
                            "SELECT spread / mid * 10000 FROM samples WHERE symbol = ? AND at >= ?",
                            (symbol, millis(since)),
                        )
                    ]
            day += timedelta(days=1)
        return found

    def read(self, symbol: str, since: datetime, until: datetime | None = None) -> list[Sample]:
        """The samples of `symbol` from `since` to `until` (now), oldest first."""
        end = until or datetime.now(UTC)
        found: list[Sample] = []
        day = since.astimezone(UTC).date()
        while day <= end.astimezone(UTC).date():
            if self.path(day).exists():
                with closing(self.connect(day)) as db:
                    rows = db.execute(
                        "SELECT at, mid, spread, bids, asks FROM samples"
                        " WHERE symbol = ? AND at >= ? AND at <= ? ORDER BY at",
                        (symbol, millis(since), millis(end)),
                    ).fetchall()
                found += [
                    Sample(
                        datetime.fromtimestamp(at / 1000, UTC),
                        symbol,
                        mid,
                        spread,
                        unpack(bids),
                        unpack(asks),
                    )
                    for at, mid, spread, bids, asks in rows
                ]
            day += timedelta(days=1)
        return found

    def prune(self, today: date) -> int:
        """Delete the days older than the ones kept; returns how many files went."""
        gone = 0
        for path in self.root.glob("*.sqlite"):
            try:
                day = date.fromisoformat(path.stem)
            except ValueError:
                continue
            if (today - day).days >= self.kept_days:
                for part in (
                    path,
                    path.with_name(path.name + "-wal"),
                    path.with_name(path.name + "-shm"),
                ):
                    part.unlink(missing_ok=True)
                gone += 1
        return gone
