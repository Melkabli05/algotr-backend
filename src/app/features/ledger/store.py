import json
import logging
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from app.features.ledger.activities import Activity

logger = logging.getLogger(__name__)

PATH = Path("data/ledger.sqlite")
SCHEMA = """
CREATE TABLE IF NOT EXISTS activities (
    id TEXT PRIMARY KEY,
    day TEXT NOT NULL,
    raw TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS picks (
    sell_order TEXT NOT NULL,
    lot TEXT NOT NULL,
    qty TEXT NOT NULL,
    PRIMARY KEY (sell_order, lot)
);
CREATE TABLE IF NOT EXISTS adjustments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    symbol TEXT NOT NULL,
    kind TEXT NOT NULL,
    qty TEXT NOT NULL,
    price TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS purification (
    symbol TEXT PRIMARY KEY,
    ratio TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Adjustment:
    """A lot added or written off to make the ledger match the broker."""

    id: int
    time: datetime
    symbol: str
    kind: Literal["open", "drop"]
    qty: Decimal
    price: Decimal
    reason: str


def day_of(row: dict[str, Any]) -> str:
    stamp = row.get("transaction_time") or row.get("date") or row.get("created_at") or ""
    return str(stamp)[:10]


class Store:
    """The account's activity as Alpaca reported it, plus the choices and adjustments made here."""

    def __init__(self, path: Path = PATH) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        return connection

    def add(self, rows: Iterable[dict[str, Any]]) -> int:
        """Keep rows not seen before; the number of new ones comes back."""
        added = 0
        with closing(self.connect()) as connection, connection:
            for row in rows:
                done = connection.execute(
                    "INSERT OR IGNORE INTO activities VALUES (?, ?, ?)",
                    (row["id"], day_of(row), json.dumps(row)),
                )
                added += done.rowcount
        return added

    def activities(self) -> list[Activity]:
        with closing(self.connect()) as connection:
            rows = connection.execute("SELECT raw FROM activities ORDER BY id").fetchall()
        found = []
        for row in rows:
            try:
                found.append(Activity.model_validate_json(row["raw"]))
            except ValueError:
                logger.warning("Skipped an activity that could not be read: %s", row["raw"][:120])
        return found

    def ratios(self) -> dict[str, Decimal]:
        """The share of each holding's dividends to purify, as entered."""
        with closing(self.connect()) as connection:
            rows = connection.execute("SELECT symbol, ratio FROM purification").fetchall()
        return {row["symbol"]: Decimal(row["ratio"]) for row in rows}

    def set_ratios(self, ratios: dict[str, Decimal | None]) -> None:
        """Save each ratio; None forgets the symbol's."""
        with closing(self.connect()) as connection, connection:
            for symbol, ratio in ratios.items():
                if ratio is None:
                    connection.execute("DELETE FROM purification WHERE symbol = ?", (symbol,))
                else:
                    connection.execute(
                        "INSERT OR REPLACE INTO purification VALUES (?, ?)", (symbol, str(ratio))
                    )

    def count(self) -> int:
        with closing(self.connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM activities").fetchone()[0])

    def last_day(self) -> date | None:
        with closing(self.connect()) as connection:
            found = connection.execute("SELECT MAX(day) FROM activities").fetchone()[0]
        return date.fromisoformat(found) if found else None

    def picks(self) -> dict[str, list[tuple[str, Decimal]]]:
        with closing(self.connect()) as connection:
            rows = connection.execute("SELECT sell_order, lot, qty FROM picks").fetchall()
        found: dict[str, list[tuple[str, Decimal]]] = {}
        for row in rows:
            found.setdefault(row["sell_order"], []).append((row["lot"], Decimal(row["qty"])))
        return found

    def set_picks(self, sell_order: str, picks: Iterable[tuple[str, Decimal]]) -> None:
        with closing(self.connect()) as connection, connection:
            connection.execute("DELETE FROM picks WHERE sell_order = ?", (sell_order,))
            connection.executemany(
                "INSERT OR REPLACE INTO picks VALUES (?, ?, ?)",
                [(sell_order, lot, str(qty)) for lot, qty in picks],
            )

    def adjustments(self) -> list[Adjustment]:
        with closing(self.connect()) as connection:
            rows = connection.execute("SELECT * FROM adjustments ORDER BY id").fetchall()
        return [
            Adjustment(
                row["id"],
                datetime.fromisoformat(row["time"]),
                row["symbol"],
                row["kind"],
                Decimal(row["qty"]),
                Decimal(row["price"]),
                row["reason"],
            )
            for row in rows
        ]

    def adjust(
        self, symbol: str, kind: Literal["open", "drop"], qty: Decimal, price: Decimal, reason: str
    ) -> None:
        with closing(self.connect()) as connection, connection:
            connection.execute(
                "INSERT INTO adjustments (time, symbol, kind, qty, price, reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(UTC).isoformat(), symbol, kind, str(qty), str(price), reason),
            )
