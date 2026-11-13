import json
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from app.features.scanner.news import NewsFeed

logger = logging.getLogger(__name__)

STEP = timedelta(days=30)
PATH = Path("data/scanner/news-backfill.json")


def window_start(path: Path) -> date | None:
    """The oldest day already fetched, or None before the first window."""
    try:
        return date.fromisoformat(json.loads(path.read_text())["done_from"])
    except (FileNotFoundError, KeyError, ValueError):
        return None


def backfill(
    feed: NewsFeed,
    symbols: Sequence[str],
    since: date,
    until: date,
    path: Path = PATH,
    report: Callable[[str], None] = logger.info,
) -> int:
    """Fetch the headlines of `symbols` from `until` back to `since`, a month at a time.

    Each finished month is written to `path`, so an interrupted run continues where it stopped;
    articles already stored are kept once. Returns how many new articles were stored.
    """
    added = 0
    end = min(window_start(path) or until, until)
    while end > since:
        start = max(end - STEP, since)
        moment = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
        found = feed.fetch(symbols, moment, datetime.combine(end, datetime.min.time(), tzinfo=UTC))
        added += found
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"done_from": start.isoformat(), "since": since.isoformat()}))
        report(f"{start} to {end}: {found} new headlines")
        end = start
    return added
