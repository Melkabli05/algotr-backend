import json
import sqlite3
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from app.features.scanner.headlines import Event

PATH = Path("data/scanner/filings.sqlite")
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKERS = "https://www.sec.gov/files/company_tickers.json"
PAUSE = 0.12
RECENT = timedelta(days=5)
SOON = timedelta(days=7)
CADENCE = 91
HISTORY = 4
SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    accession TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    form TEXT NOT NULL,
    items TEXT NOT NULL,
    filed TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS filings_symbol ON filings(symbol, filed);
CREATE TABLE IF NOT EXISTS checked (symbol TEXT PRIMARY KEY, at TEXT NOT NULL);
"""
FOR_SYMBOLS = """
SELECT symbol, accession, form, items, filed FROM filings
WHERE symbol IN (SELECT value FROM json_each(?)) AND filed >= ?
ORDER BY filed DESC
"""
RESULTS = "2.02"
ITEMS: dict[str, Event] = {
    "1.03": Event(kind="halt", direction="negative", severity="high", label="Bankruptcy filing"),
    "3.01": Event(kind="halt", direction="negative", severity="high", label="Delisting notice"),
    "4.02": Event(kind="legal", direction="negative", severity="high", label="Restated financials"),
    "2.04": Event(
        kind="offering", direction="negative", severity="medium", label="Debt accelerated"
    ),
    "2.05": Event(kind="other", direction="negative", severity="medium", label="Exit costs"),
    "2.06": Event(kind="other", direction="negative", severity="medium", label="Impairment"),
    "1.02": Event(
        kind="contract", direction="negative", severity="medium", label="Agreement ended"
    ),
    "5.02": Event(
        kind="management", direction="neutral", severity="low", label="Officer or director change"
    ),
}


class Filing(BaseModel):
    symbol: str
    accession: str
    form: str
    items: list[str]
    filed: date


class FilingsBadge(BaseModel):
    last_results: date | None = None
    days_since_results: int | None = None
    next_results: date | None = None
    soon: bool = False
    events: list[Event] = []


def client(contact: str) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": contact, "Accept-Encoding": "gzip, deflate"},
        timeout=30,
        transport=httpx.HTTPTransport(retries=2),
    )


def read(symbol: str, payload: dict[str, Any]) -> list[Filing]:
    """The 8-K filings of one company from its submissions JSON, with their item numbers."""
    recent = payload["filings"]["recent"]
    rows = zip(
        recent["accessionNumber"],
        recent["form"],
        recent["items"],
        recent["filingDate"],
        strict=True,
    )
    return [
        Filing(
            symbol=symbol,
            accession=accession,
            form=form,
            items=[item for item in items.split(",") if item],
            filed=date.fromisoformat(filed),
        )
        for accession, form, items, filed in rows
        if form.startswith("8-K")
    ]


class FilingStore:
    def __init__(self, path: Path = PATH) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        return connection

    def add(self, symbol: str, filings: Iterable[Filing], now: datetime) -> None:
        with closing(self.connect()) as connection, connection:
            connection.executemany(
                "INSERT OR REPLACE INTO filings VALUES (?, ?, ?, ?, ?)",
                [
                    (f.accession, f.symbol, f.form, ",".join(f.items), f.filed.isoformat())
                    for f in filings
                ],
            )
            connection.execute(
                "INSERT OR REPLACE INTO checked VALUES (?, ?)", (symbol, now.isoformat())
            )

    def checked(self) -> dict[str, datetime]:
        with closing(self.connect()) as connection:
            rows = connection.execute("SELECT symbol, at FROM checked").fetchall()
        return {row["symbol"]: datetime.fromisoformat(row["at"]) for row in rows}

    def of(self, symbols: Sequence[str], since: date) -> dict[str, list[Filing]]:
        found: dict[str, list[Filing]] = defaultdict(list)
        if not symbols:
            return found
        with closing(self.connect()) as connection:
            for row in connection.execute(
                FOR_SYMBOLS, (json.dumps(list(symbols)), since.isoformat())
            ):
                found[row["symbol"]].append(
                    Filing(
                        symbol=row["symbol"],
                        accession=row["accession"],
                        form=row["form"],
                        items=[item for item in row["items"].split(",") if item],
                        filed=date.fromisoformat(row["filed"]),
                    )
                )
        return found


class Edgar:
    """SEC filings for the followed names, fetched slowly and only with a named contact."""

    def __init__(
        self,
        contact: str,
        store: FilingStore,
        http: httpx.Client | None = None,
        pause: float = PAUSE,
    ) -> None:
        self.store = store
        self.http = http or client(contact)
        self.pause = pause

    def ciks(self) -> dict[str, int]:
        response = self.http.get(TICKERS)
        response.raise_for_status()
        return {
            entry["ticker"].upper().replace("-", "."): int(entry["cik_str"])
            for entry in response.json().values()
        }

    def refresh(self, symbols: Sequence[str], now: datetime) -> int:
        """Fetch each symbol's recent 8-Ks; the number of companies updated comes back."""
        known = self.ciks()
        updated = 0
        for symbol in symbols:
            cik = known.get(symbol)
            if cik is None:
                continue
            response = self.http.get(SUBMISSIONS.format(cik=cik))
            if response.status_code == httpx.codes.NOT_FOUND:
                continue
            response.raise_for_status()
            self.store.add(symbol, read(symbol, response.json()), now)
            updated += 1
            time.sleep(self.pause)
        return updated


def badge(filings: list[Filing], today: date) -> FilingsBadge:
    """When results last came out and probably come next, and any serious item filed lately."""
    results = sorted({f.filed for f in filings if RESULTS in f.items}, reverse=True)
    last = results[0] if results else None
    gaps = [(a - b).days for a, b in zip(results, results[1 : HISTORY + 1], strict=False)]
    spacing = int(statistics.median(gaps)) if gaps else CADENCE
    following = None
    if last is not None:
        following = last + timedelta(days=spacing)
        while following <= today:
            following += timedelta(days=spacing)
    recent = today - RECENT
    events = [
        ITEMS[item]
        for filing in filings
        if filing.filed >= recent
        for item in filing.items
        if item in ITEMS
    ]
    return FilingsBadge(
        last_results=last,
        days_since_results=None if last is None else (today - last).days,
        next_results=following,
        soon=following is not None and following - today <= SOON,
        events=events,
    )
