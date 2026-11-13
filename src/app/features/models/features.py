import asyncio
import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core import feature_store
from app.core.database import transaction
from app.core.feature_store import FeatureStore

logger = logging.getLogger(__name__)

JOB_ID = "features"
EVERY = timedelta(minutes=5)
NEAR_BPS = 10


class FeaturePipeline:
    """Computes the registered features and writes them to both stores: the batch ones from
    finished daily closes and headlines, the streaming one from the live coin books."""

    def __init__(
        self,
        store: FeatureStore,
        symbols: Callable[[], Iterable[str]],
        closes: Callable[[list[str]], pd.DataFrame],
        sentiment: Callable[[str, datetime], float | None],
        books: Callable[[], dict[str, Any]],
    ) -> None:
        self.store = store
        self.symbols = symbols
        self.closes = closes
        self.sentiment = sentiment
        self.books = books
        self.scheduler = AsyncIOScheduler(timezone=UTC)

    def run(self, now: datetime | None = None) -> int:
        """Compute every feature in one transaction; only once it has committed do the values
        go online, so the two stores never disagree."""
        moment = now or datetime.now(UTC)
        names = sorted(set(self.symbols()))
        try:
            written = self.compute(moment, names)
        except Exception:
            self.store.discard()
            raise
        self.store.flush()
        return written

    def resync(self) -> int:
        """Rebuild the online store from the offline one, as after Redis lost its data."""
        with transaction() as connection:
            return self.store.resync(connection)

    def compute(self, moment: datetime, names: list[str]) -> int:
        written = 0
        with transaction() as connection:
            self.store.sync(connection)
            for symbol in names:
                frame = self.closes([symbol])
                if symbol not in frame:
                    continue
                series = frame[symbol].dropna()
                series = series.loc[pd.DatetimeIndex(series.index).date < moment.date()]
                if len(series) < 127:
                    continue
                stamp = series.index[-1].to_pydatetime()
                for name, value in (
                    ("momentum_126", feature_store.momentum(series)),
                    (
                        "volatility_63",
                        feature_store.volatility(series, periods=365 if "/" in symbol else 252),
                    ),
                ):
                    self.store.write(connection, name, str(symbol), value, stamp)
                    written += 1
            for symbol in names:
                score = self.sentiment(symbol, moment)
                if score is not None:
                    self.store.write(connection, "sentiment_24h", symbol, score, moment)
                    written += 1
            for symbol, book in self.books().items():
                lean = book_imbalance(book)
                if lean is not None:
                    self.store.write(connection, "obi_10bps", symbol, lean, moment)
                    written += 1
        return written

    async def _run(self) -> None:
        try:
            await asyncio.to_thread(self.run)
        except Exception:
            logger.exception("The feature pipeline failed")

    def start(self) -> None:
        try:
            self.resync()
        except Exception:
            logger.exception("Could not rebuild the online features from the offline store")
        self.scheduler.add_job(
            self._run,
            IntervalTrigger(seconds=EVERY.total_seconds()),
            id=JOB_ID,
            coalesce=True,
            max_instances=1,
            next_run_time=datetime.now(UTC),
            misfire_grace_time=None,
        )
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)


def book_imbalance(book: Any) -> float | None:
    """Imbalance of a live book within 10 bps of its mid, the same measure the book desk uses."""
    from app.features.book import metrics

    best = book.best()
    if best is None:
        return None
    (bid, _), (ask, _) = best
    mid = (bid + ask) / 2
    bids, asks = book.sides()
    near_bids, _ = metrics.within(bids, mid, NEAR_BPS, "bid")
    near_asks, _ = metrics.within(asks, mid, NEAR_BPS, "ask")
    return metrics.imbalance(near_bids, near_asks)


def headline_sentiment(store: Any) -> Callable[[str, datetime], float | None]:
    """The gated FinBERT score of a symbol's headlines over the day before `now`: the unsure
    ones left out, the rest weighted by how sure FinBERT was."""
    from app.features.scanner import mood

    def score(symbol: str, now: datetime) -> float | None:
        if store is None:
            return None
        found = mood.scored(store, [symbol], now - timedelta(days=1), now)
        trusted = found["weight"].sum()
        if trusted <= 0:
            return None
        return float((found["score"] * found["weight"]).sum() / trusted)

    return score
