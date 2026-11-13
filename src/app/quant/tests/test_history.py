import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import create_autospec

import pytest
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

from app.quant import history


def bar(day: datetime, close: float, hour: int = 0, open_: float | None = None) -> dict[str, Any]:
    stamp = day.replace(hour=hour, minute=0, second=0, microsecond=0).isoformat()
    return {"t": stamp, "o": close if open_ is None else open_, "c": close}


@pytest.fixture
def clients() -> tuple[Any, Any]:
    return (
        create_autospec(StockHistoricalDataClient, instance=True),
        create_autospec(CryptoHistoricalDataClient, instance=True),
    )


def test_completed_days_only_with_stock_bars_dated_by_their_session(clients: Any) -> None:
    stocks, crypto = clients
    today = datetime.now(UTC)
    days = [today - timedelta(days=offset) for offset in (3, 2, 1, 0)]
    stocks.get_stock_bars.return_value = {
        "SPUS": [bar(d, 100 + i, hour=4) for i, d in enumerate(days)]
    }
    closes = history.daily_closes(stocks, crypto, ["SPUS"], 1)
    assert list(closes["SPUS"]) == [100, 101, 102]
    assert closes.index[-1] == (today - timedelta(days=1)).replace(tzinfo=None).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def test_stock_requests_are_adjusted_sip_and_end_before_the_free_plan_delay(clients: Any) -> None:
    stocks, crypto = clients
    stocks.get_stock_bars.return_value = {"SPUS": [bar(datetime.now(UTC) - timedelta(days=2), 1)]}
    history.daily_closes(stocks, crypto, ["SPUS"], 2)
    request = stocks.get_stock_bars.call_args.args[0]
    assert request.feed == DataFeed.SIP
    assert request.adjustment == Adjustment.ALL
    assert datetime.now(UTC).replace(tzinfo=None) - request.end >= history.SIP_DELAY - timedelta(
        seconds=5
    )
    assert 700 < (request.end - request.start).days < 735


def test_opens_come_with_the_closes_on_the_same_days(clients: Any) -> None:
    stocks, crypto = clients
    today = datetime.now(UTC)
    days = [today - timedelta(days=offset) for offset in (3, 2, 1, 0)]
    stocks.get_stock_bars.return_value = {
        "SPUS": [bar(d, 100 + i, hour=4, open_=99 + i) for i, d in enumerate(days)]
    }
    prices = history.daily_prices(stocks, crypto, ["SPUS"], 1)
    assert list(prices.opens["SPUS"]) == [99, 100, 101]
    assert prices.opens.index.equals(prices.closes.index)


def test_crypto_symbols_use_the_crypto_client(clients: Any) -> None:
    stocks, crypto = clients
    crypto.get_crypto_bars.return_value = {
        "BTC/USD": [bar(datetime.now(UTC) - timedelta(days=2), 50_000)]
    }
    closes = history.daily_closes(stocks, crypto, ["BTC/USD"], 1)
    assert list(closes["BTC/USD"]) == [50_000]
    stocks.get_stock_bars.assert_not_called()


def test_symbols_are_aligned_on_their_shared_days(clients: Any) -> None:
    stocks, crypto = clients
    start = datetime.now(UTC) - timedelta(days=5)
    stocks.get_stock_bars.side_effect = [
        {"A": [bar(start + timedelta(days=i), 10 + i, hour=4) for i in range(4)]},
        {"B": [bar(start + timedelta(days=i), 20 + i, hour=4) for i in range(1, 4)]},
    ]
    closes = history.daily_closes(stocks, crypto, ["A", "B"], 1)
    assert list(closes["A"]) == [11, 12, 13]
    assert list(closes["B"]) == [21, 22, 23]


def test_a_symbol_without_bars_is_explained(clients: Any) -> None:
    stocks, crypto = clients
    stocks.get_stock_bars.return_value = {}
    with pytest.raises(history.NotEnoughHistoryError, match="no daily bars for SPUS"):
        history.daily_closes(stocks, crypto, ["SPUS"], 1)


def test_cached_closes_fetch_once_per_universe_and_serve_nine_years(clients: Any) -> None:
    stocks, crypto = clients
    stocks.get_stock_bars.return_value = {"SPUS": [bar(datetime.now(UTC) - timedelta(days=2), 1)]}
    first = history.cached_closes(stocks, crypto, ["SPUS"])
    again = history.cached_closes(stocks, crypto, ["SPUS"])
    assert first is again
    stocks.get_stock_bars.assert_called_once()
    request = stocks.get_stock_bars.call_args.args[0]
    assert 3_200 < (request.end - request.start).days < 3_300


def test_requests_arriving_together_for_one_universe_share_a_single_download(clients: Any) -> None:
    stocks, crypto = clients

    def slow(_request: Any) -> dict[str, list[dict[str, Any]]]:
        time.sleep(0.2)
        return {"SPUS": [bar(datetime.now(UTC) - timedelta(days=2), 1)]}

    stocks.get_stock_bars.side_effect = slow
    with ThreadPoolExecutor(5) as pool:
        frames = list(pool.map(lambda _: history.cached_closes(stocks, crypto, ["SPUS"]), range(5)))

    assert all(frame is frames[0] for frame in frames)
    stocks.get_stock_bars.assert_called_once()
