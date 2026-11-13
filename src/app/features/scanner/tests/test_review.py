from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest
from alpaca.trading.client import TradingClient

from app.features.scanner import review
from app.features.scanner.news import NewsFeed, NewsStore
from app.features.scanner.tests.support import at, client_with

DAY = date(2026, 11, 3)
OPEN_UTC = datetime(2026, 11, 3, 14, 30, tzinfo=UTC)
CALENDAR = [{"date": "2026-11-03", "open": "09:30", "close": "16:00"}]


def queued(number: str, symbol: str, placed: datetime) -> dict[str, Any]:
    return {"id": number, "symbol": symbol, "side": "buy", "created_at": placed.isoformat()}


def broker(orders: list[dict[str, Any]], calendar: list[dict[str, Any]] = CALENDAR) -> MagicMock:
    mock: MagicMock = create_autospec(TradingClient, instance=True)
    mock.get_calendar.return_value = calendar
    mock.get_orders.return_value = orders
    return mock


@pytest.fixture
def store(tmp_path: Path) -> NewsStore:
    return NewsStore(tmp_path / "news.sqlite")


def test_the_market_opens_when_alpaca_s_calendar_says_in_new_york_time() -> None:
    assert review.opening(broker([]), DAY) == OPEN_UTC
    assert review.opening(broker([], calendar=[]), DAY) is None
    other = [{"date": "2026-11-04", "open": "09:30", "close": "16:00"}]
    assert review.opening(broker([], calendar=other), DAY) is None


def test_the_review_is_due_from_forty_five_minutes_before_the_open_until_the_open() -> None:
    trading = broker([])
    before = OPEN_UTC - timedelta(minutes=46)
    assert review.due(trading, before, None) is False
    assert review.due(trading, OPEN_UTC - timedelta(minutes=44), None) is True
    assert review.due(trading, OPEN_UTC, None) is False


def test_it_runs_once_a_day() -> None:
    assert review.due(broker([]), OPEN_UTC - timedelta(minutes=10), DAY) is False


def test_the_clock_change_moves_the_review_with_the_open() -> None:
    summer = [{"date": "2026-07-01", "open": "09:30", "close": "16:00"}]
    trading = broker([], calendar=summer)
    opens = datetime(2026, 7, 1, 13, 30, tzinfo=UTC)
    assert review.due(trading, opens - timedelta(minutes=30), None) is True
    assert review.due(trading, opens - timedelta(minutes=60), None) is False


def test_a_queued_buy_is_cancelled_when_its_company_had_a_setback_after_it_was_placed(
    store: NewsStore,
) -> None:
    placed = datetime(2026, 11, 3, 0, 5, tzinfo=UTC)
    articles = [
        at(1, "Acme Lowers Full-Year Outlook", ["ACME"], placed + timedelta(hours=3)),
        at(2, "Beta Declares Quarterly Dividend", ["BETA"], placed + timedelta(hours=4)),
    ]
    trading = broker(
        [
            queued("o1", "ACME", placed),
            queued("o2", "BETA", placed),
            queued("o3", "BTC/USD", placed),
        ]
    )
    feed = NewsFeed(client_with(articles), store)
    result = review.run(trading, feed, store, OPEN_UTC - timedelta(minutes=40))
    assert result.checked == 2
    assert [(c.symbol, c.label) for c in result.cancelled] == [("ACME", "Guidance cut")]
    trading.cancel_order_by_id.assert_called_once_with("o1")


def test_a_setback_reported_before_the_order_was_placed_does_not_cancel_it(
    store: NewsStore,
) -> None:
    placed = datetime(2026, 11, 3, 0, 5, tzinfo=UTC)
    articles = [at(1, "Acme Lowers Full-Year Outlook", ["ACME"], placed - timedelta(hours=20))]
    trading = broker([queued("o1", "ACME", placed)])
    result = review.run(trading, NewsFeed(client_with(articles), store), store, OPEN_UTC)
    assert result.cancelled == []
    trading.cancel_order_by_id.assert_not_called()


def test_with_nothing_queued_the_news_is_not_even_asked_for(store: NewsStore) -> None:
    client = client_with([])
    result = review.run(broker([]), NewsFeed(client, store), store, OPEN_UTC)
    assert (result.checked, result.cancelled) == (0, [])
    client.get_news.assert_not_called()
