import asyncio
import logging
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from pathlib import Path
from threading import Lock

import httpx
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel

from app.core import audit
from app.core.compliance import SHARIAH_FUNDS, Screening
from app.core.sleeve import Sleeve
from app.features.scanner import (
    evaluate,
    features,
    lists,
    mood,
    offer,
    rank,
    scan,
    screening,
    shadow,
    universe,
    vetoes,
)
from app.features.scanner.crypto import CryptoDesk, CryptoScan, CryptoStatus
from app.features.scanner.features import wide
from app.features.scanner.filings import Edgar
from app.features.scanner.market import Market, MarketView
from app.features.scanner.news import NewsFeed, NewsReport
from app.features.scanner.refresh import Refresher
from app.features.scanner.schemas import (
    Evaluation,
    Explanation,
    Progress,
    Refresh,
    Scan,
    ScannerStatus,
    ShadowReport,
)
from app.features.scanner.sentiment import Scorer
from app.features.scanner.store import BarStore
from app.features.scanner.watch import Watch
from app.quant.history import NotEnoughHistoryError

ROOT = Path("data/scanner")
JOB_ID = "scanner-refresh"
NEWS_JOB_ID = "scanner-news"
REVIEW_JOB_ID = "scanner-review"
CRYPTO_JOB_ID = "scanner-crypto"
CATCH_UP_JOB_ID = "scanner-catch-up"
CATCH_UP_EVERY = timedelta(minutes=10)
MAX_FAILURES = 6
NEWS_EVERY = timedelta(minutes=15)
REVIEW_EVERY = timedelta(minutes=5)
CRYPTO_EVERY = timedelta(hours=1)
RUN_AT = CronTrigger(hour=0, minute=1, timezone=UTC)
RUN_AT_TIME = clock_time(0, 1)
CATCH_UP_DELAY = timedelta(seconds=5)

logger = logging.getLogger(__name__)


class AlreadyRunningError(RuntimeError):
    """Raised when a refresh is requested while another one is still running."""


class State(BaseModel):
    last: Refresh | None = None


def parameters(names: list[str]) -> dict[str, object]:
    """Everything that decides a result besides the data, for the audit manifest."""
    return {
        "variants": [variant.label for variant in rank.VARIANTS],
        "fee_bps": evaluate.FEE_BPS,
        "min_price": universe.MIN_PRICE,
        "min_dollar_volume": universe.MIN_DOLLAR_VOLUME,
        "lists": names,
    }


class Scanner:
    """Keeps the daily-bar panel and the Shariah lists current, once a night and on demand."""

    def __init__(
        self,
        trading: TradingClient,
        stocks: StockHistoricalDataClient,
        root: Path = ROOT,
        sources: Sequence[lists.Source] = lists.SOURCES,
        http: httpx.Client | None = None,
        log: audit.AuditLog | None = None,
        news: NewsFeed | None = None,
        edgar: Edgar | None = None,
        crypto: CryptoDesk | None = None,
        sentiment: Scorer | None = None,
    ) -> None:
        self.trading = trading
        self.stocks = stocks
        self.root = root
        self.sources = sources
        self.audit = log or audit.AuditLog(audit.AUDIT_PATH)
        self.store = BarStore(root / "bars.parquet")
        self.refresher = Refresher(trading, stocks, self.store, root, sources, http)
        self.view = MarketView(self.store, root, sources)
        self.watch = Watch(trading, news, edgar, root / "watch.json", self.audit, sentiment)
        self.crypto = crypto
        self.scheduler = AsyncIOScheduler(timezone=UTC)
        self.lock = Lock()
        self.progress: Progress | None = None
        self.failures: Counter[date] = Counter()
        self.state = self._load()

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    def _load(self) -> State:
        try:
            return State.model_validate_json(self.state_path.read_text())
        except (FileNotFoundError, ValueError):
            return State()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(self.state.model_dump_json(indent=2))

    def _report(self, progress: Progress) -> None:
        self.progress = progress

    def status(self) -> ScannerStatus:
        job = self.scheduler.get_job(JOB_ID)
        return ScannerStatus(
            running=self.progress if self.lock.locked() else None,
            last=self.state.last,
            next_run=job.next_run_time if job else None,
            news=self.watch.status(),
            review=self.watch.state.review,
            crypto=CryptoStatus() if self.crypto is None else self.crypto.status(),
        )

    def refresh(self) -> Refresh:
        """Bring lists and bars up to the last completed session; the outcome is always recorded."""
        if not self.lock.acquire(blocking=False):
            raise AlreadyRunningError("A refresh is already running")
        started, clock = datetime.now(UTC), time.monotonic()
        try:
            record = self.refresher.run(started, self._report)
        except Exception as exc:
            logger.exception("Scanner refresh failed")
            record = Refresh(
                started=started, seconds=0, status="error", detail=f"{type(exc).__name__}: {exc}"
            )
        if record.status == "error":
            self.failures[started.date()] += 1
        if record.status == "ok":
            try:
                self._log_scan(record)
            except Exception as exc:
                logger.exception("Scanner could not log the scan")
                record.detail = f"Scan not logged: {type(exc).__name__}: {exc}"
            self._follow()
        try:
            record.seconds = round(time.monotonic() - clock, 1)
            self.state.last = record
            self._save()
        finally:
            self.progress = None
            self.lock.release()
        return record

    def market(self) -> Market | None:
        return self.view.get()

    def latest(self) -> Scan | None:
        """Today's ranking across the whole market and what each variant would hold."""
        found = self.market()
        if found is None:
            return None
        result = scan.scan(found.analysis, found.names, found.lists, found.stale)
        now = datetime.now(UTC)
        badges = self.watch.badges([item.symbol for item in result.candidates], now)
        for item in result.candidates:
            headline, filed = badges[item.symbol]
            event = self.watch.veto(headline, filed)
            item.news, item.results = headline, filed
            item.blocked = None if event is None else event.label
        return result

    def explain(self, symbol: str) -> Explanation | None:
        found = self.market()
        if found is None:
            return None
        result = scan.explain(found.analysis, symbol, found.names, found.lists)
        result.reasons.extend(self.watch.reasons(symbol, datetime.now(UTC)))
        return result

    def news(self, symbol: str) -> NewsReport:
        return self.watch.report(symbol, datetime.now(UTC))

    def sleeve(self, variant: str) -> Sleeve | None:
        """The holdings of `variant` for the autopilot, its promotion record and any problem."""
        found = self.market()
        if found is None:
            return None
        now = datetime.now(UTC)
        held = found.analysis.scheduled.get(variant, {})
        gate = offer.criteria(
            evaluate.latest(self.audit),
            variant,
            len(shadow.logged(self.audit)),
            self.audit.verify().valid,
            now,
        )
        return offer.build(
            variant,
            found,
            self.watch.vetoed(list(held), now),
            gate,
            offer.problems(found, self.state.last, now),
        )

    def mood(self, sessions: Sequence[date]) -> pd.DataFrame:
        """The daily FinBERT score and rule risk of the headlines on the Shariah-listed stocks."""
        members = {
            symbol
            for snapshot in lists.current(self.sources, self.root / "lists")
            for symbol in snapshot.symbols
        }
        symbols = sorted(members | set(SHARIAH_FUNDS))
        feed = self.watch.feed
        if not sessions or feed is None:
            return mood.daily(pd.DataFrame(), sessions)
        first = datetime.combine(min(sessions), datetime.min.time(), tzinfo=UTC) - timedelta(days=7)
        last = datetime.combine(max(sessions), datetime.min.time(), tzinfo=UTC) + timedelta(days=1)
        return mood.daily(mood.scored(feed.store, symbols, first, last), sessions)

    def listed(self) -> list[str]:
        today = datetime.now(UTC).date()
        snapshots = lists.current(self.sources, self.root / "lists")
        fresh = [s for s in snapshots if (today - s.as_of) <= lists.STALE_AFTER]
        return sorted({symbol for snapshot in fresh for symbol in snapshot.symbols})

    def screen(self, symbol: str) -> Screening:
        """The Shariah verdict for `symbol` from the lists on disk now (`screening.screen`)."""
        snapshots = lists.current(self.sources, self.root / "lists")
        coins = self.crypto.coins if self.crypto else []
        return screening.screen(symbol, snapshots, coins, datetime.now(UTC).date())

    def crypto_scan(self) -> CryptoScan | None:
        return None if self.crypto is None else self.crypto.scan()

    def following(self) -> list[str]:
        """The Shariah-listed names and every stock shown as a candidate."""
        found = self.market()
        if found is None:
            return []
        shown = scan.scan(found.analysis, found.names, found.lists, found.stale).candidates
        return sorted({*found.analysis.members, *(item.symbol for item in shown)})

    def _follow(self) -> None:
        symbols, now = self.following(), datetime.now(UTC)
        self.watch.poll(symbols, now)
        self.watch.refresh_filings(symbols, now)

    def poll_news(self) -> None:
        self.watch.poll(self.following(), datetime.now(UTC))

    def review(self) -> None:
        self.watch.review_if_due(datetime.now(UTC))

    def evaluate(self) -> Evaluation:
        """Judge the eight variants on the stored panel and keep the result in the audit log."""
        snapshots = lists.current(self.sources, self.root / "lists")
        members = {symbol for snapshot in snapshots for symbol in snapshot.symbols}
        versions = lists.versions(snapshots)
        panel = self.store.read()
        result = evaluate.evaluate(panel, members, versions)
        data = {"close": wide(panel[panel["symbol"].isin(members)], "close")}
        self.audit.record(
            "scan-evaluation",
            {
                "result": result.model_dump(mode="json"),
                "manifest": audit.manifest(parameters(versions), data),
            },
        )
        return result

    def veto_study(self) -> vetoes.VetoStudy:
        """Test the news veto on the stored headlines and prices and keep the result in the log."""
        feed = self.watch.feed
        if feed is None:
            raise NotEnoughHistoryError("News is not set up, so there are no headlines to test")
        snapshots = lists.current(self.sources, self.root / "lists")
        members = {symbol for snapshot in snapshots for symbol in snapshot.symbols}
        panel = self.store.read()
        found = features.compute(panel[panel["symbol"].isin(members)])
        result = vetoes.study(found.close, found.open, rank.eligible(found), feed.store, members)
        self.audit.record(
            vetoes.KIND,
            {
                "result": result.model_dump(mode="json"),
                "manifest": audit.manifest(
                    parameters(lists.versions(snapshots)), {"close": found.close}
                ),
            },
        )
        return result

    def shadow(self) -> ShadowReport:
        """How the logged picks fared afterwards, from stored prices only."""
        return shadow.shadow(self.audit, self.store.read(shadow.needed(self.audit)))

    def _log_scan(self, record: Refresh) -> None:
        found = self.market()
        if found is None or not found.analysis.members:
            record.detail = "No current Shariah list, so no scan was logged"
            return
        result = scan.scan(found.analysis, found.names, found.lists, found.stale)
        close = found.analysis.found.close[found.analysis.members]
        manifest = audit.manifest(parameters(found.versions), {"close": close})
        shadow.record(self.audit, result, manifest)
        record.scanned = result.session

    def request_run(self) -> None:
        """Start a refresh in the background; progress shows in the status."""
        if self.lock.locked():
            raise AlreadyRunningError("A refresh is already running")
        self.scheduler.add_job(self._scheduled, "date")

    async def _scheduled(self) -> None:
        try:
            await asyncio.to_thread(self.refresh)
        except AlreadyRunningError:
            logger.info("Skipped a scanner refresh: one is already running")

    async def _polled(self) -> None:
        await asyncio.to_thread(self.poll_news)

    async def _reviewed(self) -> None:
        await asyncio.to_thread(self.review)

    async def _crypto_checked(self) -> None:
        if self.crypto is not None:
            await asyncio.to_thread(self.crypto.refresh_if_stale, datetime.now(UTC))

    async def _caught_up(self) -> None:
        if self._missed_today():
            await self._scheduled()

    def _missed_today(self) -> bool:
        now = datetime.now(UTC)
        last = self.state.last
        done_today = last is not None and last.status == "ok" and last.started.date() == now.date()
        gave_up = self.failures[now.date()] >= MAX_FAILURES
        return not done_today and not gave_up and now.time() >= RUN_AT_TIME

    def start(self) -> None:
        """Schedule the nightly refresh and keep catching up until today's has succeeded."""
        self.scheduler.add_job(
            self._scheduled,
            RUN_AT,
            id=JOB_ID,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=3600,
        )
        self.scheduler.add_job(
            self._caught_up,
            IntervalTrigger(seconds=CATCH_UP_EVERY.total_seconds()),
            id=CATCH_UP_JOB_ID,
            coalesce=True,
            max_instances=1,
        )
        first = datetime.now(UTC) + CATCH_UP_DELAY
        self.scheduler.add_job(
            self._polled,
            IntervalTrigger(seconds=NEWS_EVERY.total_seconds()),
            id=NEWS_JOB_ID,
            coalesce=True,
            max_instances=1,
            next_run_time=first,
            misfire_grace_time=None,
        )
        self.scheduler.add_job(
            self._reviewed,
            IntervalTrigger(seconds=REVIEW_EVERY.total_seconds()),
            id=REVIEW_JOB_ID,
            coalesce=True,
            max_instances=1,
            next_run_time=first,
            misfire_grace_time=None,
        )
        self.scheduler.add_job(
            self._crypto_checked,
            IntervalTrigger(seconds=CRYPTO_EVERY.total_seconds()),
            id=CRYPTO_JOB_ID,
            coalesce=True,
            max_instances=1,
            next_run_time=first,
            misfire_grace_time=None,
        )
        self.watch.retype()
        self.scheduler.start()
        if self._missed_today():
            self.scheduler.add_job(
                self._scheduled, "date", run_date=datetime.now(UTC) + CATCH_UP_DELAY
            )

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
