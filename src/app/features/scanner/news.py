import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import closing
from datetime import UTC, datetime, timedelta
from itertools import batched
from pathlib import Path
from typing import Any, cast

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from pydantic import BaseModel

from app.features.scanner import headlines
from app.features.scanner.filings import FilingsBadge
from app.features.scanner.headlines import Event

PATH = Path("data/scanner/news.sqlite")
BACKFILL = timedelta(days=60)
OVERLAP = timedelta(days=1)
RECENT = timedelta(days=3)
VETO_WINDOW = timedelta(days=5)
BASELINE_MIN = 10
CHUNK = 20
MAX_TAGS_FOR_VETO = 3
SHOWN = 25
SUMMARY_CHARS = 600
SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY,
    created TEXT NOT NULL,
    source TEXT NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    url TEXT,
    tags INTEGER NOT NULL,
    kind TEXT NOT NULL,
    direction TEXT NOT NULL,
    severity TEXT NOT NULL,
    label TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tagged (
    article INTEGER NOT NULL REFERENCES articles(id),
    symbol TEXT NOT NULL,
    PRIMARY KEY (article, symbol)
);
CREATE INDEX IF NOT EXISTS tagged_symbol ON tagged(symbol, article);
CREATE TABLE IF NOT EXISTS tracked (symbol TEXT PRIMARY KEY, since TEXT NOT NULL);
"""
FOR_SYMBOLS = """
SELECT t.symbol AS symbol, a.id, a.created, a.source, a.headline, a.url, a.tags, a.kind,
       a.direction, a.severity, a.label
FROM tagged t JOIN articles a ON a.id = t.article
WHERE t.symbol IN (SELECT value FROM json_each(?)) AND a.created >= ?
ORDER BY a.created DESC
"""


class Sentiment(BaseModel):
    """What the model made of a headline, kept with the hash of its text for audit and backtests."""

    text_hash: str
    model: str
    score: float
    risk: int
    risk_basis: str = "rules"
    confidence: float
    positive: float
    negative: float
    neutral: float
    fallback: bool
    processed: datetime
    uncertainty: float | None = None


class Headline(BaseModel):
    id: int
    time: datetime
    source: str
    headline: str
    url: str | None
    tags: int
    event: Event
    sentiment: Sentiment | None = None


class NewsBadge(BaseModel):
    count: int
    attention: float | None = None
    worst: Event | None = None
    veto: Event | None = None


class NewsReport(BaseModel):
    symbol: str
    badge: NewsBadge
    headlines: list[Headline]
    filings: FilingsBadge | None = None


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def row_to_headline(row: sqlite3.Row) -> Headline:
    return Headline(
        id=row["id"],
        time=parse(row["created"]),
        source=row["source"],
        headline=row["headline"],
        url=row["url"],
        tags=row["tags"],
        event=Event(
            kind=row["kind"],
            direction=row["direction"],
            severity=row["severity"],
            label=row["label"],
        ),
    )


class NewsStore:
    """Headlines and the events typed from them, kept in SQLite so an update replaces its row."""

    def __init__(self, path: Path = PATH) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        return connection

    def add(self, articles: Iterable[dict[str, Any]]) -> int:
        """Store the articles; the number of ones not seen before comes back."""
        items = list(articles)
        if not items:
            return 0
        with closing(self.connect()) as connection, connection:
            known = connection.execute(
                "SELECT COUNT(*) FROM articles WHERE id IN (SELECT value FROM json_each(?))",
                (json.dumps([item["id"] for item in items]),),
            ).fetchone()[0]
            for item in items:
                tags = [symbol.upper() for symbol in item["symbols"]]
                event = headlines.classify(item["headline"], len(tags))
                connection.execute(
                    "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET headline = excluded.headline, "
                    "summary = excluded.summary, url = excluded.url, tags = excluded.tags, "
                    "kind = excluded.kind, direction = excluded.direction, "
                    "severity = excluded.severity, label = excluded.label",
                    (
                        item["id"],
                        item["created_at"],
                        item["source"],
                        item["headline"],
                        (item.get("summary") or "")[:SUMMARY_CHARS],
                        item.get("url"),
                        len(tags),
                        event.kind,
                        event.direction,
                        event.severity,
                        event.label,
                    ),
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO tagged VALUES (?, ?)", [(item["id"], s) for s in tags]
                )
        return len(items) - int(known)

    def retype(self) -> None:
        """Type every stored headline again, so a change to the rules reaches what was fetched."""
        with closing(self.connect()) as connection, connection:
            rows = connection.execute("SELECT id, headline, tags FROM articles").fetchall()
            connection.executemany(
                "UPDATE articles SET kind = ?, direction = ?, severity = ?, label = ? WHERE id = ?",
                [
                    (event.kind, event.direction, event.severity, event.label, row["id"])
                    for row in rows
                    for event in [headlines.classify(row["headline"], row["tags"])]
                ],
            )

    def latest(self) -> datetime | None:
        with closing(self.connect()) as connection:
            value = connection.execute("SELECT MAX(created) FROM articles").fetchone()[0]
        return None if value is None else parse(value)

    def untracked(self, symbols: Sequence[str]) -> list[str]:
        with closing(self.connect()) as connection:
            have = {row[0] for row in connection.execute("SELECT symbol FROM tracked")}
        return [symbol for symbol in symbols if symbol not in have]

    def track(self, symbols: Iterable[str], now: datetime) -> None:
        with closing(self.connect()) as connection, connection:
            connection.executemany(
                "INSERT OR IGNORE INTO tracked VALUES (?, ?)",
                [(symbol, now.isoformat()) for symbol in symbols],
            )

    def of(self, symbols: Sequence[str], since: datetime) -> dict[str, list[Headline]]:
        """Each symbol's headlines since `since`, newest first."""
        found: dict[str, list[Headline]] = defaultdict(list)
        if not symbols:
            return found
        with closing(self.connect()) as connection:
            for row in connection.execute(
                FOR_SYMBOLS, (json.dumps(list(symbols)), since.isoformat())
            ):
                found[row["symbol"]].append(row_to_headline(row))
        return found

    def count(self) -> int:
        with closing(self.connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0])


class NewsFeed:
    """Keeps the store current for the symbols being followed."""

    def __init__(self, client: NewsClient, store: NewsStore) -> None:
        self.client = client
        self.store = store

    def poll(self, symbols: Sequence[str], now: datetime) -> int:
        """New headlines for `symbols`: a name seen for the first time gets 60 days back."""
        fresh = self.store.untracked(symbols)
        latest = self.store.latest()
        since = now - BACKFILL if latest is None else latest - OVERLAP
        new = set(fresh)
        added = self.fetch(fresh, now - BACKFILL, now)
        added += self.fetch([symbol for symbol in symbols if symbol not in new], since, now)
        self.store.track(fresh, now)
        return added

    def fetch(self, symbols: Sequence[str], start: datetime, end: datetime) -> int:
        added = 0
        for group in batched(symbols, CHUNK, strict=False):
            request = NewsRequest(symbols=",".join(group), start=start, end=end, sort="desc")
            found = cast(dict[str, list[dict[str, Any]]], self.client.get_news(request))
            added += self.store.add(found.get("news", []))
        return added


def badge(items: list[Headline], now: datetime) -> NewsBadge:
    """What the last three days of news mean for a name, and whether it blocks a new buy."""
    newest = sorted(items, key=lambda item: item.time, reverse=True)
    recent = [item for item in newest if item.time >= now - RECENT]
    vetoes = [
        item.event
        for item in newest
        if item.time >= now - VETO_WINDOW and item.tags <= MAX_TAGS_FOR_VETO and item.event.veto
    ]
    negative = [item.event for item in recent if item.event.direction == "negative"]
    order = {"high": 0, "medium": 1, "low": 2}
    worst = min(negative, key=lambda event: order[event.severity], default=None)
    earlier = sum(item.time < now - RECENT for item in newest)
    baseline = earlier / max(1, (BACKFILL - RECENT) / RECENT)
    attention = len(recent) / baseline if earlier >= BASELINE_MIN and baseline else None
    return NewsBadge(
        count=len(recent),
        attention=attention,
        worst=worst,
        veto=vetoes[0] if vetoes else None,
    )
