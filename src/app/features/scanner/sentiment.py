import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Collection
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import NamedTuple

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.core import sidecar
from app.features.scanner import headlines
from app.features.scanner.headlines import Event
from app.features.scanner.news import Headline, NewsStore, Sentiment

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/models/finbert")
REPO = "ProsusAI/finbert"
REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
MODEL = f"{REPO}@{REVISION[:7]}"
FALLBACK_BELOW = 0.8
CHUNK = 2000
BATCH = 64
TIMEOUT = 1800
PASSES = 5
OPEN_RATE = 0.34
MIN_GATE_SAMPLE = 1000
SCHEMA = """
CREATE TABLE IF NOT EXISTS sentiment_cache (
    text_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    score REAL NOT NULL,
    risk INTEGER NOT NULL,
    confidence REAL NOT NULL,
    positive REAL NOT NULL,
    negative REAL NOT NULL,
    neutral REAL NOT NULL,
    fallback INTEGER NOT NULL,
    processed TEXT NOT NULL,
    PRIMARY KEY (text_hash, model)
);
CREATE TABLE IF NOT EXISTS article_sentiment (
    article INTEGER PRIMARY KEY REFERENCES articles(id),
    text_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sentiment_gate (
    model TEXT PRIMARY KEY,
    tau REAL NOT NULL,
    open_rate REAL NOT NULL,
    sample INTEGER NOT NULL,
    computed TEXT NOT NULL
);
"""
UNMEASURED = """
SELECT c.text_hash, MIN(a.headline) AS headline FROM sentiment_cache c
JOIN article_sentiment s ON s.text_hash = c.text_hash
JOIN articles a ON a.id = s.article
WHERE c.model = ? AND c.uncertainty IS NULL
GROUP BY c.text_hash LIMIT ?
"""
UNSCORED = """
SELECT a.id, a.headline FROM articles a
LEFT JOIN article_sentiment s ON s.article = a.id
WHERE s.article IS NULL AND TRIM(a.headline) != ''
ORDER BY a.created DESC LIMIT ?
"""
CACHED = (
    "SELECT * FROM sentiment_cache "
    "WHERE model = ? AND text_hash IN (SELECT value FROM json_each(?))"
)
Scores = list[tuple[float, float, float]]
RunJob = Callable[..., sidecar.JobResult]


class SentimentError(RuntimeError):
    """Raised when FinBERT cannot score a batch of headlines."""


class Batch(NamedTuple):
    """The probabilities of positive, negative and neutral for each text, the model, and how
    much the score moved under dropout when that was measured."""

    model: str
    scores: Scores
    uncertainty: list[float] | None = None


Classify = Callable[[list[str]], Batch]


class SentimentStatus(BaseModel):
    enabled: bool = False
    model: str = MODEL
    scored: int = 0
    backlog: int = 0
    below: float | None = None
    last: datetime | None = None
    error: str | None = None


def digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def risk_of(event: Event) -> int:
    """From 1 to 5 how serious a setback the headline reports, read from its rule typing."""
    if event.veto:
        return 5
    if event.direction == "negative":
        return 4 if event.severity == "medium" else 3
    return 2 if event.direction == "neutral" and event.severity != "low" else 1


def record(
    text: str,
    scores: tuple[float, float, float],
    model: str,
    now: datetime,
    uncertainty: float | None = None,
) -> Sentiment:
    """The structured record of one text: score, risk, confidence, model, time and text hash."""
    positive, negative, neutral = scores
    confidence = max(scores)
    return Sentiment(
        text_hash=digest(text),
        model=model,
        score=positive - negative,
        risk=risk_of(headlines.classify(text.strip())),
        confidence=confidence,
        positive=positive,
        negative=negative,
        neutral=neutral,
        fallback=confidence < FALLBACK_BELOW,
        processed=now,
        uncertainty=uncertainty,
    )


def finbert(texts: list[str], run: RunJob = sidecar.run_job, folder: Path = MODEL_DIR) -> Batch:
    """Score `texts` with FinBERT in the sidecar, with the score's spread over five dropout
    passes; weights other than the pinned ones are refused."""
    params = {"model_dir": str(folder.resolve()), "batch": BATCH, "passes": PASSES}
    outcome = run("sentiment", params, {"data": pd.DataFrame({"text": texts})}, timeout=TIMEOUT)
    if outcome.status != "ok" or outcome.data is None:
        raise SentimentError(outcome.error or "The sidecar returned nothing")
    found = outcome.data
    if found["revision"] != REVISION:
        raise SentimentError(f"FinBERT {found['revision'][:7]} ran; {REVISION[:7]} is pinned")
    order = [found["labels"].index(name) for name in ("positive", "negative", "neutral")]
    return Batch(
        MODEL,
        [(row[order[0]], row[order[1]], row[order[2]]) for row in found["probabilities"]],
        found.get("uncertainty"),
    )


def ready() -> bool:
    """Whether the sidecar is set up and holds the FinBERT weights."""
    return sidecar.available() and (MODEL_DIR / "model.safetensors").exists()


class SentimentStore:
    """The records, in the news database: one for each text, and which article has which."""

    def __init__(self, news: NewsStore) -> None:
        self.news = news

    def connect(self) -> sqlite3.Connection:
        connection = self.news.connect()
        connection.executescript(SCHEMA)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sentiment_cache)")}
        if "uncertainty" not in columns:
            connection.execute("ALTER TABLE sentiment_cache ADD COLUMN uncertainty REAL")
        return connection

    def unscored(self, limit: int) -> list[tuple[int, str]]:
        """The newest headlines that no record is linked to yet."""
        with closing(self.connect()) as connection:
            return [(row["id"], row["headline"]) for row in connection.execute(UNSCORED, (limit,))]

    def cached(self, hashes: Collection[str], model: str) -> dict[str, Sentiment]:
        """The records of `model` for the texts with these hashes."""
        if not hashes:
            return {}
        with closing(self.connect()) as connection:
            rows = connection.execute(CACHED, (model, json.dumps(sorted(hashes)))).fetchall()
        return {row["text_hash"]: row_to_sentiment(row) for row in rows}

    def add(self, records: list[Sentiment]) -> None:
        with closing(self.connect()) as connection, connection:
            connection.executemany(
                "INSERT OR IGNORE INTO sentiment_cache (text_hash, model, score, risk, "
                "confidence, positive, negative, neutral, fallback, processed, uncertainty) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.text_hash,
                        item.model,
                        item.score,
                        item.risk,
                        item.confidence,
                        item.positive,
                        item.negative,
                        item.neutral,
                        int(item.fallback),
                        item.processed.isoformat(),
                        item.uncertainty,
                    )
                    for item in records
                ],
            )

    def unmeasured(self, limit: int, model: str = MODEL) -> list[tuple[str, str]]:
        """Texts scored before their uncertainty was measured: a headline for each."""
        with closing(self.connect()) as connection:
            rows = connection.execute(UNMEASURED, (model, limit)).fetchall()
        return [(row["text_hash"], row["headline"]) for row in rows]

    def measured(self, spreads: dict[str, float], model: str = MODEL) -> None:
        with closing(self.connect()) as connection, connection:
            connection.executemany(
                "UPDATE sentiment_cache SET uncertainty = ? WHERE text_hash = ? AND model = ?",
                [(value, key, model) for key, value in spreads.items()],
            )

    def calibrate(self, now: datetime, model: str = MODEL) -> float | None:
        """The gate's threshold: the spread below which OPEN_RATE of all measured headlines
        fall, so the gate opens for about a third of them (Ngo et al., UG-CPPO)."""
        with closing(self.connect()) as connection, connection:
            values = [
                row[0]
                for row in connection.execute(
                    "SELECT uncertainty FROM sentiment_cache "
                    "WHERE model = ? AND uncertainty IS NOT NULL",
                    (model,),
                )
            ]
            if len(values) < MIN_GATE_SAMPLE:
                return None
            tau = float(np.quantile(values, OPEN_RATE))
            connection.execute(
                "INSERT OR REPLACE INTO sentiment_gate VALUES (?, ?, ?, ?, ?)",
                (model, tau, OPEN_RATE, len(values), now.isoformat()),
            )
        return tau

    def gate(self, model: str = MODEL) -> float | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT tau FROM sentiment_gate WHERE model = ?", (model,)
            ).fetchone()
        return None if row is None else float(row[0])

    def link(self, links: dict[int, str]) -> None:
        with closing(self.connect()) as connection, connection:
            connection.executemany(
                "INSERT OR REPLACE INTO article_sentiment VALUES (?, ?)", list(links.items())
            )

    def counts(self, model: str) -> tuple[int, int, float | None, datetime | None]:
        """The records of `model`, the headlines without one, and the share below the threshold."""
        with closing(self.connect()) as connection:
            scored, below, last = connection.execute(
                "SELECT COUNT(*), AVG(fallback), MAX(processed) "
                "FROM sentiment_cache WHERE model = ?",
                (model,),
            ).fetchone()
            backlog = connection.execute(
                "SELECT COUNT(*) FROM articles a LEFT JOIN article_sentiment s ON s.article = a.id "
                "WHERE s.article IS NULL AND TRIM(a.headline) != ''"
            ).fetchone()[0]
        return scored, backlog, below, None if last is None else datetime.fromisoformat(last)


def row_to_sentiment(row: sqlite3.Row) -> Sentiment:
    return Sentiment(
        text_hash=row["text_hash"],
        model=row["model"],
        score=row["score"],
        risk=row["risk"],
        confidence=row["confidence"],
        positive=row["positive"],
        negative=row["negative"],
        neutral=row["neutral"],
        fallback=bool(row["fallback"]),
        processed=datetime.fromisoformat(row["processed"]),
        uncertainty=row["uncertainty"],
    )


class Scorer:
    """Gives every headline a sentiment record, scoring each distinct text once."""

    def __init__(
        self,
        news: NewsStore,
        classify: Classify = finbert,
        usable: Callable[[], bool] = ready,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.store = SentimentStore(news)
        self.classify = classify
        self.usable = usable
        self.clock = clock
        self.lock = Lock()
        self.error: str | None = None

    def chunk(self, pending: list[tuple[int, str]]) -> None:
        """Score the headlines whose text has no record, and link every one of them to its text."""
        hashes = {article: digest(text) for article, text in pending}
        known = self.store.cached(set(hashes.values()), MODEL)
        fresh = {
            hashes[article]: text.strip()
            for article, text in pending
            if hashes[article] not in known
        }
        if fresh:
            batch = self.classify(list(fresh.values()))
            now = self.clock()
            spreads: list[float | None] = (
                list(batch.uncertainty) if batch.uncertainty else [None] * len(batch.scores)
            )
            self.store.add(
                [
                    record(text, scores, batch.model, now, spread)
                    for text, scores, spread in zip(
                        fresh.values(), batch.scores, spreads, strict=True
                    )
                ]
            )
        self.store.link(hashes)

    def measure(self, limit: int = CHUNK) -> int:
        """Measure the uncertainty of up to `limit` texts scored before it was kept, then set
        the gate's threshold again; how many were measured."""
        pending = self.store.unmeasured(limit)
        if pending:
            batch = self.classify([text.strip() for _, text in pending])
            if batch.uncertainty is None:
                raise SentimentError("The sidecar measured no uncertainty")
            self.store.measured(
                dict(zip([key for key, _ in pending], batch.uncertainty, strict=True))
            )
        self.store.calibrate(self.clock())
        return len(pending)

    def score_new(self) -> int:
        """Score the headlines that have no record yet, a chunk at a time; how many were linked."""
        if not self.usable() or not self.lock.acquire(blocking=False):
            return 0
        done = 0
        try:
            while pending := self.store.unscored(CHUNK):
                self.chunk(pending)
                done += len(pending)
            if done:
                self.store.calibrate(self.clock())
            self.error = None
        except Exception as exc:
            logger.exception("Sentiment scoring failed")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.lock.release()
        return done

    def attach(self, items: list[Headline]) -> list[Headline]:
        """The headlines with their sentiment record, where there is one."""
        found = self.store.cached({digest(item.headline) for item in items}, MODEL)
        return [
            item.model_copy(update={"sentiment": found.get(digest(item.headline))})
            for item in items
        ]

    def status(self) -> SentimentStatus:
        scored, backlog, below, last = self.store.counts(MODEL)
        return SentimentStatus(
            enabled=self.usable(),
            scored=scored,
            backlog=backlog,
            below=below,
            last=last,
            error=self.error,
        )
