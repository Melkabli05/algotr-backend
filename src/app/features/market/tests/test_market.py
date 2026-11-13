from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from alpaca.common.enums import Sort
from alpaca.data.enums import DataFeed
from fastapi.testclient import TestClient


def bar(minute: int, close: float) -> dict[str, Any]:
    return {
        "t": f"2026-10-28T11:{minute:02d}:00Z",
        "o": close - 1,
        "h": close + 2,
        "l": close - 3,
        "c": close,
        "v": 10,
        "n": 3,
        "vw": close,
    }


def test_crypto_symbols_use_the_crypto_client_and_return_oldest_first(
    client: TestClient, stock_data: MagicMock, crypto_data: MagicMock
) -> None:
    crypto_data.get_crypto_bars.return_value = {"BTC/USD": [bar(2, 102), bar(1, 101)]}
    bars = client.get("/market/bars", params={"symbol": "btc/usd", "timeframe": "5Min"}).json()
    assert [b["close"] for b in bars] == [101, 102]
    assert bars[0] == {
        "time": "2026-10-28T11:01:00Z",
        "open": 100,
        "high": 103,
        "low": 98,
        "close": 101,
        "volume": 10,
    }
    stock_data.get_stock_bars.assert_not_called()
    request = crypto_data.get_crypto_bars.call_args.args[0]
    assert (request.symbol_or_symbols, request.limit, request.sort) == ("BTC/USD", 300, Sort.DESC)
    assert (request.timeframe.amount, request.timeframe.unit.value) == (5, "Min")


def test_stock_symbols_use_the_free_iex_feed(
    client: TestClient, stock_data: MagicMock, crypto_data: MagicMock
) -> None:
    stock_data.get_stock_bars.return_value = {"AAPL": [bar(1, 101)]}
    response = client.get("/market/bars", params={"symbol": "AAPL", "limit": 50})
    assert response.status_code == 200
    request = stock_data.get_stock_bars.call_args.args[0]
    assert (request.symbol_or_symbols, request.limit, request.feed) == ("AAPL", 50, DataFeed.IEX)
    crypto_data.get_crypto_bars.assert_not_called()


def test_the_chart_reads_the_chosen_feed(
    client: TestClient,
    stock_data: MagicMock,
    crypto_data: MagicMock,
    choose_feed: Callable[[DataFeed], None],
) -> None:
    choose_feed(DataFeed.SIP)
    stock_data.get_stock_bars.return_value = {"AAPL": [bar(1, 101)]}
    client.get("/market/bars", params={"symbol": "AAPL"})
    assert stock_data.get_stock_bars.call_args.args[0].feed == DataFeed.SIP


@pytest.mark.parametrize("feed", [DataFeed.IEX, DataFeed.SIP])
def test_the_feed_in_use_is_served(
    client: TestClient, choose_feed: Callable[[DataFeed], None], feed: DataFeed
) -> None:
    choose_feed(feed)
    assert client.get("/market/feed").json() == {"feed": feed.value}


def test_the_lookback_window_starts_in_the_past(
    client: TestClient, stock_data: MagicMock, crypto_data: MagicMock
) -> None:
    stock_data.get_stock_bars.return_value = {}
    client.get("/market/bars", params={"symbol": "AAPL", "timeframe": "1Day"})
    request = stock_data.get_stock_bars.call_args.args[0]
    assert (request.end or request.start).year <= 2026
    assert request.start.year in (2024, 2025)


def test_symbol_without_bars_returns_an_empty_list(
    client: TestClient, stock_data: MagicMock, crypto_data: MagicMock
) -> None:
    stock_data.get_stock_bars.return_value = {}
    assert client.get("/market/bars", params={"symbol": "ZZZZ"}).json() == []


@pytest.mark.parametrize(
    "params",
    [
        {"symbol": "../orders"},
        {"symbol": ""},
        {"symbol": "AAPL", "timeframe": "2Min"},
        {"symbol": "AAPL", "limit": 0},
        {"symbol": "AAPL", "limit": 1001},
        {},
    ],
)
def test_invalid_requests_are_rejected_before_calling_the_broker(
    client: TestClient, stock_data: MagicMock, crypto_data: MagicMock, params: dict[str, Any]
) -> None:
    assert client.get("/market/bars", params=params).status_code == 422
    stock_data.get_stock_bars.assert_not_called()
    crypto_data.get_crypto_bars.assert_not_called()


def test_clock_reports_whether_the_stock_market_is_open(
    client: TestClient, trading: MagicMock
) -> None:
    trading.get_clock.return_value = {
        "timestamp": "2026-10-28T07:56:00-04:00",
        "is_open": False,
        "next_open": "2026-10-28T09:30:00-04:00",
        "next_close": "2026-10-28T16:00:00-04:00",
    }
    assert client.get("/market/clock").json() == {
        "is_open": False,
        "next_open": "2026-10-28T09:30:00-04:00",
        "next_close": "2026-10-28T16:00:00-04:00",
    }
