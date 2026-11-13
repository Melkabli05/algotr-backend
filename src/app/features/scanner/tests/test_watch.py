import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import httpx

from app.core.tests.support import api_error
from app.features.scanner import sentiment
from app.features.scanner import service as module
from app.features.scanner.filings import Edgar, FilingStore
from app.features.scanner.news import NewsFeed, NewsStore
from app.features.scanner.sentiment import Batch, Scorer
from app.features.scanner.service import Scanner
from app.features.scanner.tests.support import (
    NAMES,
    at,
    client_with,
    listed_scanner,
    noisy_market,
)
from app.features.scanner.tests.test_filings import CONTACT, payload, website

CUT = "Outlook Lowered By Management"


def recent(number: int, headline: str, symbol: str, hours: float) -> dict[str, Any]:
    return at(number, headline, [symbol], datetime.now(UTC) - timedelta(hours=hours))


def trading_of(scanner: Scanner) -> MagicMock:
    return cast(MagicMock, scanner.trading)


def scanner_with(tmp_path: Path, articles: list[dict[str, Any]], **extra: Any) -> Scanner:
    store = NewsStore(tmp_path / "scanner" / "news.sqlite")
    return listed_scanner(
        tmp_path, noisy_market(), news=NewsFeed(client_with(articles), store), **extra
    )


def test_a_refresh_follows_every_listed_name_and_stores_their_news(tmp_path: Path) -> None:
    articles = [
        recent(1, "Acme Lowers Full-Year Outlook", "AAA", 1),
        recent(2, "Morgan Stanley Maintains Buy on BBB", "BBB", 2),
        at(3, "Stocks Rally", NAMES, datetime.now(UTC) - timedelta(hours=3)),
    ]
    scanner = scanner_with(tmp_path, articles)
    scanner.refresh()
    news = scanner.status().news
    assert news.enabled and news.error is None and news.polled is not None
    assert (news.articles, news.following, news.sec) == (3, 6, False)
    assert set(scanner.following()) == set(NAMES)


def test_without_a_news_feed_the_scanner_says_so_and_carries_on(tmp_path: Path) -> None:
    scanner = listed_scanner(tmp_path, noisy_market())
    record = scanner.refresh()
    result = scanner.latest()
    assert record.status == "ok" and not scanner.status().news.enabled
    assert result is not None and all(item.news is None for item in result.candidates)
    assert scanner.news("AAA").headlines == []


def test_a_candidate_carries_its_news_and_a_serious_setback_blocks_it(tmp_path: Path) -> None:
    scanner = scanner_with(tmp_path, [])
    scanner.refresh()
    top = scanner.latest()
    assert top is not None
    first, last = top.candidates[0].symbol, top.candidates[-1].symbol
    scanner = scanner_with(
        tmp_path,
        [
            recent(1, "Acme Lowers Full-Year Outlook", first, 1),
            recent(2, "Morgan Stanley Maintains Buy on Acme", last, 2),
        ],
    )
    scanner.poll_news()
    latest = scanner.latest()
    assert latest is not None
    blocked = {item.symbol: item for item in latest.candidates}
    assert blocked[first].blocked == "Guidance cut" and blocked[first].news is not None
    badge = blocked[last].news
    assert blocked[last].blocked is None and badge is not None and badge.count == 1
    assert scanner.watch.vetoed(NAMES, datetime.now(UTC)) == {first: "Guidance cut"}


def test_the_checklist_of_a_stock_includes_its_news(tmp_path: Path) -> None:
    scanner = scanner_with(tmp_path, [])
    scanner.refresh()
    top = scanner.latest()
    assert top is not None
    symbol = top.candidates[0].symbol
    scanner = scanner_with(tmp_path, [recent(1, "Acme Lowers Full-Year Outlook", symbol, 1)])
    scanner.poll_news()
    explanation = scanner.explain(symbol)
    assert explanation is not None
    line = next(r for r in explanation.reasons if r.check.startswith("No serious setback"))
    assert line.passed is False and line.detail == "Guidance cut"


def test_the_headlines_of_a_stock_come_newest_first_with_their_events(tmp_path: Path) -> None:
    articles = [
        recent(1, "Acme Declares Quarterly Dividend", "AAA", 30),
        recent(2, "Morgan Stanley Downgrades Acme", "AAA", 3),
        recent(3, "Citigroup Maintains Buy on Acme", "AAA", 2),
    ]
    scanner = scanner_with(tmp_path, articles)
    scanner.refresh()
    report = scanner.news("AAA")
    assert [item.id for item in report.headlines] == [3, 2, 1]
    assert report.headlines[1].event.label == "Downgrade"
    assert report.badge.count == 3 and report.badge.worst is not None


def test_a_news_outage_is_recorded_and_does_not_fail_the_refresh(tmp_path: Path) -> None:
    client = client_with([])
    client.get_news.side_effect = api_error("forbidden")
    store = NewsStore(tmp_path / "scanner" / "news.sqlite")
    scanner = listed_scanner(tmp_path, noisy_market(), news=NewsFeed(client, store))
    record = scanner.refresh()
    assert record.status == "ok"
    news = scanner.status().news
    assert news.polled is None and news.error is not None and "APIError" in news.error


def edgar_for(tmp_path: Path, rows: dict[str, list[tuple[str, str, str, date]]]) -> Edgar:
    ciks = {name: 1000 + index for index, name in enumerate(NAMES)}
    submissions = {ciks[name]: payload(found) for name, found in rows.items()}
    http, _ = website(ciks, submissions)
    return Edgar(CONTACT, FilingStore(tmp_path / "scanner" / "filings.sqlite"), http, pause=0)


def test_with_a_sec_contact_the_filings_of_the_followed_names_are_read_too(
    tmp_path: Path,
) -> None:
    today = datetime.now(UTC).date()
    probe = scanner_with(tmp_path, [])
    probe.refresh()
    top = probe.latest()
    assert top is not None
    symbol = top.candidates[0].symbol
    rows = {
        symbol: [
            ("a", "8-K", "4.02", today - timedelta(days=1)),
            ("b", "8-K", "2.02", today - timedelta(days=20)),
        ]
    }
    scanner = scanner_with(tmp_path, [], edgar=edgar_for(tmp_path, rows))
    scanner.refresh()
    status = scanner.status().news
    assert status.sec and status.filings_at is not None and status.filings_error is None
    latest = scanner.latest()
    assert latest is not None
    found = {item.symbol: item for item in latest.candidates}
    assert found[symbol].blocked == "Restated financials"
    results = found[symbol].results
    assert results is not None and results.days_since_results == 20
    explanation = scanner.explain(symbol)
    assert explanation is not None
    reasons = explanation.reasons
    assert [r.passed for r in reasons if r.check.startswith("No serious SEC")] == [False]
    assert any(r.check == "Results not due within a week" for r in reasons)


def test_a_sec_failure_is_recorded_beside_the_news_and_the_refresh_still_succeeds(
    tmp_path: Path,
) -> None:
    http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403)))
    edgar = Edgar(CONTACT, FilingStore(tmp_path / "scanner" / "filings.sqlite"), http, pause=0)
    scanner = scanner_with(tmp_path, [], edgar=edgar)
    assert scanner.refresh().status == "ok"
    status = scanner.status().news
    assert status.filings_at is None and status.filings_error is not None


def test_the_review_cancels_a_queued_buy_once_a_day_and_the_audit_log_remembers(
    tmp_path: Path,
) -> None:
    opens = datetime(2026, 11, 3, 14, 30, tzinfo=UTC)
    placed = datetime(2026, 11, 3, 0, 5, tzinfo=UTC)
    articles = [at(1, "Acme Lowers Full-Year Outlook", ["AAA"], placed + timedelta(hours=3))]
    scanner = scanner_with(tmp_path, articles)
    trading = trading_of(scanner)
    trading.get_calendar.return_value = [{"date": "2026-11-03", "open": "09:30"}]
    trading.get_orders.return_value = [
        {"id": "o1", "symbol": "AAA", "side": "buy", "created_at": placed.isoformat()}
    ]
    result = scanner.watch.review_if_due(opens - timedelta(minutes=30))
    assert result is not None and [c.symbol for c in result.cancelled] == ["AAA"]
    trading.cancel_order_by_id.assert_called_once_with("o1")
    assert scanner.watch.review_if_due(opens - timedelta(minutes=20)) is None
    assert scanner.status().review == result
    assert [e.kind for e in scanner.audit.entries()] == ["review"]


def test_outside_the_window_the_review_does_nothing(tmp_path: Path) -> None:
    scanner = scanner_with(tmp_path, [])
    trading = trading_of(scanner)
    trading.get_calendar.return_value = []
    scanner.review()
    assert scanner.status().review is None
    trading.get_orders.assert_not_called()


def test_the_state_of_the_watch_survives_a_restart(tmp_path: Path) -> None:
    scanner = scanner_with(tmp_path, [recent(1, "Acme Wins Contract", "AAA", 1)])
    scanner.refresh()
    again = scanner_with(tmp_path, [])
    assert again.status().news.polled == scanner.status().news.polled


def test_polling_and_the_review_are_scheduled_and_first_run_right_after_a_start(
    tmp_path: Path,
) -> None:
    async def scenario() -> list[datetime | None]:
        scanner = scanner_with(tmp_path, [])
        scanner.start()
        jobs = [
            scanner.scheduler.get_job(module.NEWS_JOB_ID),
            scanner.scheduler.get_job(module.REVIEW_JOB_ID),
        ]
        scanner.stop()
        return [job.next_run_time if job else None for job in jobs]

    first = asyncio.run(scenario())
    assert all(
        when is not None and when - datetime.now(UTC) <= module.CATCH_UP_DELAY for when in first
    )


class Positive:
    """A classifier that finds every headline positive and keeps what it was asked."""

    def __init__(self) -> None:
        self.asked: list[list[str]] = []

    def __call__(self, texts: list[str]) -> Batch:
        self.asked.append(list(texts))
        return Batch(sentiment.MODEL, [(0.9, 0.05, 0.05)] * len(texts))


def with_sentiment(tmp_path: Path, articles: list[dict[str, Any]], usable: bool = True) -> Scanner:
    store = NewsStore(tmp_path / "scanner" / "news.sqlite")
    return listed_scanner(
        tmp_path,
        noisy_market(),
        news=NewsFeed(client_with(articles), store),
        sentiment=Scorer(store, Positive(), lambda: usable),
    )


def test_headlines_are_scored_right_after_each_poll_and_shown_with_their_record(
    tmp_path: Path,
) -> None:
    scanner = with_sentiment(tmp_path, [recent(1, "Acme Wins A Contract", "AAA", 1)])
    scanner.refresh()
    [shown] = scanner.news("AAA").headlines
    assert shown.sentiment is not None and shown.sentiment.score == 0.85
    status = scanner.status().news.sentiment
    assert (status.enabled, status.scored, status.backlog, status.error) == (True, 1, 0, None)


def test_a_poll_that_finds_nothing_new_asks_the_model_for_nothing(tmp_path: Path) -> None:
    scanner = with_sentiment(tmp_path, [recent(1, "Acme Wins A Contract", "AAA", 1)])
    scanner.refresh()
    scorer = scanner.watch.sentiment
    assert scorer is not None and isinstance(scorer.classify, Positive)
    scanner.poll_news()
    assert len(scorer.classify.asked) == 1


def test_without_the_model_the_news_still_flows_and_says_the_sentiment_is_off(
    tmp_path: Path,
) -> None:
    scanner = with_sentiment(tmp_path, [recent(1, "Acme Wins A Contract", "AAA", 1)], usable=False)
    scanner.refresh()
    [shown] = scanner.news("AAA").headlines
    assert shown.sentiment is None and scanner.status().news.articles == 1
    assert not scanner.status().news.sentiment.enabled
    assert scanner.status().news.sentiment.backlog == 1


def test_a_news_outage_does_not_stop_the_scoring_of_what_was_stored(tmp_path: Path) -> None:
    scanner = with_sentiment(tmp_path, [recent(1, "Acme Wins A Contract", "AAA", 1)])
    scanner.refresh()
    feed = scanner.watch.feed
    assert feed is not None
    feed.client = MagicMock()
    feed.client.get_news.side_effect = api_error("down", 500)
    scanner.poll_news()
    assert scanner.status().news.error is not None
    assert scanner.status().news.sentiment.scored == 1
