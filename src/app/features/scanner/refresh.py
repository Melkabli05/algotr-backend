from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from itertools import batched
from pathlib import Path

import httpx
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from app.features.scanner import bars, lists, universe
from app.features.scanner.schemas import Funnel, ListHealth, Progress, Refresh
from app.features.scanner.store import BarStore, combine, diverged

RECENT_DAYS = 45
PANEL_YEARS = 8
RECENT_BATCH = 1000
HISTORY_BATCH = 100
HISTORY_CHUNK = 500

Report = Callable[[Progress], None]


def reporter(
    report: Report, stage: str, offset: int = 0, total: int | None = None
) -> Callable[[int, int], None]:
    def progress(done: int, size: int) -> None:
        report(Progress(stage=stage, done=offset + done, total=total or size))

    return progress


class Refresher:
    """One refresh: the lists, a census of the market and the missing history, saved as it goes."""

    def __init__(
        self,
        trading: TradingClient,
        stocks: StockHistoricalDataClient,
        store: BarStore,
        root: Path,
        sources: Sequence[lists.Source],
        http: httpx.Client | None,
    ) -> None:
        self.trading = trading
        self.stocks = stocks
        self.store = store
        self.root = root
        self.sources = sources
        self.http = http

    def run(self, now: datetime, report: Report) -> Refresh:
        today = now.date()
        report(Progress(stage="Lists", done=0, total=1))
        found = universe.assets(self.trading)
        tradable = universe.tradable(found)
        universe.save_names(self.root, found, tradable)
        snapshots, health = self._lists(tradable, today)
        listed = lists.listed(snapshots, tradable, today)

        recent, rejected = bars.fetch(
            self.stocks,
            sorted(tradable),
            now - timedelta(days=RECENT_DAYS),
            now,
            RECENT_BATCH,
            reporter(report, "Recent bars"),
        )
        liquid = universe.liquid(recent)

        stored = self.store.read()
        stale = diverged(stored, recent)
        panel = combine(stored, recent, bars.empty(), stale)
        if stale or len(panel) != len(stored):
            self.store.write(panel)

        need = sorted(((liquid | listed) | stale) - set(panel["symbol"].unique()))
        since = now - timedelta(days=round(365 * PANEL_YEARS))
        refused: list[str] = []
        done = 0
        for chunk in batched(need, HISTORY_CHUNK, strict=False):
            progress = reporter(report, "History", done, len(need))
            history, declined = bars.fetch(self.stocks, chunk, since, now, HISTORY_BATCH, progress)
            refused.extend(declined)
            panel = combine(panel, recent, history)
            self.store.write(panel)
            done += len(chunk)

        return Refresh(
            started=now,
            seconds=0,
            status="ok",
            session=bars.last_session(panel),
            funnel=Funnel(
                assets=len(found),
                tradable=len(tradable),
                liquid=len(liquid),
                listed=len(listed),
                listed_liquid=len(listed & liquid),
                panel=int(panel["symbol"].nunique()),
            ),
            lists=health,
            rows=len(panel),
            rebuilt=sorted(stale),
            rejected=sorted({*rejected, *refused}),
        )

    def _lists(
        self, tradable: set[str], today: date
    ) -> tuple[list[lists.Snapshot], list[ListHealth]]:
        http = self.http or lists.client()
        snapshots: list[lists.Snapshot] = []
        health: list[ListHealth] = []
        try:
            for source in self.sources:
                snapshot, problems = lists.collect(source, self.root / "lists", http, today)
                health.append(lists.health(source.name, snapshot, problems, tradable, today))
                if snapshot:
                    snapshots.append(snapshot)
        finally:
            if self.http is None:
                http.close()
        return snapshots, health
