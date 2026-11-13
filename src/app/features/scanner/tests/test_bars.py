import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from alpaca.common.exceptions import APIError

from app.core.tests.support import api_error
from app.features.scanner import bars
from app.features.scanner.tests.support import Market, bar


def window(days: int = 30) -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    return now - timedelta(days=days), now


def market_of(*symbols: str) -> Market:
    market = Market()
    for symbol in symbols:
        market.add(symbol, days=60)
    return market


def test_symbols_go_in_batches_and_progress_is_reported() -> None:
    market = market_of("AAA", "BBB", "CCC", "DDD", "EEE")
    seen: list[tuple[int, int]] = []
    start, end = window()
    found = bars.fetch(
        market.client(), list(market.series), start, end, 2, lambda *a: seen.append(a)
    )
    assert sorted(len(request.symbol_or_symbols) for request in market.requests) == [1, 2, 2]
    assert seen == [(0, 5), (2, 5), (4, 5), (5, 5)]
    assert set(found.bars["symbol"]) == set(market.series)
    assert found.rejected == []


def test_the_bars_are_in_the_panel_layout() -> None:
    market = market_of("AAA")
    start, end = window()
    frame = bars.fetch(market.client(), ["AAA"], start, end, 10).bars
    assert list(frame.columns) == bars.COLUMNS
    assert frame["date"].dtype == "datetime64[ns]"
    assert {str(frame[name].dtype) for name in bars.COLUMNS[2:]} == {"float64"}
    assert bars.last_session(frame) == market.through


def test_a_session_that_is_still_open_is_never_kept() -> None:
    market = market_of("AAA")
    today = datetime.now(UTC).date()
    market.series["AAA"].append(bar(today, 99.0, 1.0))
    market.through = today
    start, end = window()
    frame = bars.fetch(market.client(), ["AAA"], start, end, 10).bars
    assert frame["date"].max() < pd.Timestamp(today)


def test_a_symbol_alpaca_refuses_is_set_aside_and_the_rest_still_arrive() -> None:
    market = market_of("AAA", "BBB")
    market.invalid = {"BAD1", "BAD2"}
    start, end = window()
    found = bars.fetch(market.client(), ["AAA", "BAD1", "BBB", "BAD2"], start, end, 10)
    assert sorted(found.rejected) == ["BAD1", "BAD2"]
    assert set(found.bars["symbol"]) == {"AAA", "BBB"}


def test_a_valid_symbol_without_bars_is_simply_absent() -> None:
    market = market_of("AAA")
    start, end = window()
    found = bars.fetch(market.client(), ["AAA", "QUIET"], start, end, 10)
    assert set(found.bars["symbol"]) == {"AAA"}
    assert found.rejected == []


def test_any_other_error_stops_the_fetch() -> None:
    client = market_of("AAA").client()
    client.get_stock_bars.side_effect = api_error("forbidden")
    start, end = window()
    with pytest.raises(APIError):
        bars.fetch(client, ["AAA"], start, end, 10)


def test_no_symbols_gives_an_empty_panel() -> None:
    start, end = window()
    found = bars.fetch(market_of().client(), [], start, end, 10)
    assert found.bars.empty and list(found.bars.columns) == bars.COLUMNS
    assert bars.last_session(found.bars) is None


def test_a_failure_stops_the_batches_that_have_not_started() -> None:
    market = market_of(*[f"S{index:02d}" for index in range(60)])
    client = market.client()

    def slow_failure(request: object) -> None:
        time.sleep(0.05)
        raise api_error("forbidden")

    client.get_stock_bars.side_effect = slow_failure
    start, end = window()
    with pytest.raises(APIError):
        bars.fetch(client, list(market.series), start, end, 1)
    assert client.get_stock_bars.call_count < 30


def test_batches_are_fetched_side_by_side_but_come_back_in_order() -> None:
    market = market_of(*[f"S{index:02d}" for index in range(12)])
    start, end = window()
    found = bars.fetch(market.client(), list(market.series), start, end, 3)
    assert list(dict.fromkeys(found.bars["symbol"])) == list(market.series)
