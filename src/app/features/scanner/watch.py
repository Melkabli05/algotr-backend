import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock

from alpaca.trading.client import TradingClient
from pydantic import BaseModel

from app.core import audit
from app.features.scanner import filings, news, review
from app.features.scanner.filings import Edgar, FilingsBadge
from app.features.scanner.headlines import Event
from app.features.scanner.news import BACKFILL, SHOWN, NewsBadge, NewsFeed, NewsReport
from app.features.scanner.review import Review
from app.features.scanner.schemas import NewsStatus, Reason
from app.features.scanner.sentiment import Scorer, SentimentStatus

logger = logging.getLogger(__name__)
FILING_HISTORY = 400


class WatchState(BaseModel):
    polled: datetime | None = None
    following: int = 0
    error: str | None = None
    filings_at: datetime | None = None
    filings_error: str | None = None
    reviewed: date | None = None
    review: Review | None = None


class Watch:
    """What the scanner learns apart from prices: headlines, SEC filings and the pre-open review."""

    def __init__(
        self,
        trading: TradingClient,
        feed: NewsFeed | None,
        edgar: Edgar | None,
        path: Path,
        log: audit.AuditLog,
        sentiment: Scorer | None = None,
    ) -> None:
        self.trading = trading
        self.feed = feed
        self.sentiment = sentiment
        self.edgar = edgar
        self.path = path
        self.log = log
        self.lock = Lock()
        self.state = self._load()

    def _load(self) -> WatchState:
        try:
            return WatchState.model_validate_json(self.path.read_text())
        except (FileNotFoundError, ValueError):
            return WatchState()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.state.model_dump_json(indent=2))

    def retype(self) -> None:
        if self.feed is not None:
            self.feed.store.retype()

    def status(self) -> NewsStatus:
        return NewsStatus(
            enabled=self.feed is not None,
            polled=self.state.polled,
            articles=0 if self.feed is None else self.feed.store.count(),
            following=self.state.following,
            error=self.state.error,
            sec=self.edgar is not None,
            filings_at=self.state.filings_at,
            filings_error=self.state.filings_error,
            sentiment=SentimentStatus() if self.sentiment is None else self.sentiment.status(),
        )

    def poll(self, symbols: Sequence[str], now: datetime) -> None:
        """Fetch new headlines for the followed names; a failure is kept and tried again later."""
        if self.feed is None or not self.lock.acquire(blocking=False):
            return
        try:
            self.feed.poll(symbols, now)
            self.state.polled, self.state.following, self.state.error = now, len(symbols), None
        except Exception as exc:
            logger.exception("News poll failed")
            self.state.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._save()
            self.lock.release()
        if self.sentiment is not None:
            self.sentiment.score_new()

    def refresh_filings(self, symbols: Sequence[str], now: datetime) -> None:
        if self.edgar is None:
            return
        try:
            self.edgar.refresh(symbols, now)
            self.state.filings_at, self.state.filings_error = now, None
        except Exception as exc:
            logger.exception("SEC filings refresh failed")
            self.state.filings_error = f"{type(exc).__name__}: {exc}"
        self._save()

    def badges(
        self, symbols: Sequence[str], now: datetime
    ) -> dict[str, tuple[NewsBadge | None, FilingsBadge | None]]:
        headlines = {} if self.feed is None else self.feed.store.of(symbols, now - BACKFILL)
        reports = (
            {}
            if self.edgar is None
            else self.edgar.store.of(symbols, now.date() - timedelta(days=FILING_HISTORY))
        )
        return {
            symbol: (
                None if self.feed is None else news.badge(headlines.get(symbol, []), now),
                None if self.edgar is None else filings.badge(reports.get(symbol, []), now.date()),
            )
            for symbol in symbols
        }

    def veto(
        self, news_badge: NewsBadge | None, filings_badge: FilingsBadge | None
    ) -> Event | None:
        if news_badge is not None and news_badge.veto is not None:
            return news_badge.veto
        if filings_badge is not None:
            return next((event for event in filings_badge.events if event.veto), None)
        return None

    def vetoed(self, symbols: Sequence[str], now: datetime) -> dict[str, str]:
        """The followed names with a serious setback in the last five days, and what it was."""
        found = {}
        for symbol, (headline, filed) in self.badges(symbols, now).items():
            event = self.veto(headline, filed)
            if event is not None:
                found[symbol] = event.label
        return found

    def report(self, symbol: str, now: datetime) -> NewsReport:
        """The newest headlines of one name, and what the last days add up to."""
        headline, filed = self.badges([symbol], now)[symbol]
        found = (
            []
            if self.feed is None
            else self.feed.store.of([symbol], now - BACKFILL).get(symbol, [])
        )
        shown = found[:SHOWN]
        return NewsReport(
            symbol=symbol,
            badge=headline or NewsBadge(count=0),
            headlines=shown if self.sentiment is None else self.sentiment.attach(shown),
            filings=filed,
        )

    def reasons(self, symbol: str, now: datetime) -> list[Reason]:
        """The news tests a stock must pass, in the shape of the other checklist lines."""
        headline, filed = self.badges([symbol], now)[symbol]
        found: list[Reason] = []
        if headline is not None:
            found.append(
                Reason(
                    check="No serious setback in the news in 5 days",
                    passed=headline.veto is None,
                    detail=headline.veto.label if headline.veto else f"{headline.count} in 3 days",
                )
            )
        if filed is not None:
            serious = next((event for event in filed.events if event.veto), None)
            found.append(
                Reason(
                    check="No serious SEC filing in 5 days",
                    passed=serious is None,
                    detail=serious.label if serious else "none",
                )
            )
            if filed.next_results is not None:
                found.append(
                    Reason(
                        check="Results not due within a week",
                        passed=not filed.soon,
                        detail=f"next around {filed.next_results:%b} {filed.next_results.day} "
                        "(estimate from earlier reports)",
                    )
                )
        return found

    def review_if_due(self, now: datetime) -> Review | None:
        """Run the pre-open review once the market is within 45 minutes of opening."""
        if self.feed is None or not review.due(self.trading, now, self.state.reviewed):
            return None
        result = review.run(self.trading, self.feed, self.feed.store, now)
        self.state.reviewed = now.astimezone(review.EASTERN).date()
        self.state.review = result
        self._save()
        if result.checked:
            self.log.record("review", {"review": result.model_dump(mode="json")})
        return result
