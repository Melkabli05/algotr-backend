from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

from app.quant import volume
from app.quant.history import NotEnoughHistoryError
from app.quant.volume import COIN, STOCK, Profile

NOW = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)


def at(local: datetime) -> str:
    return local.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def session_bars(
    day: date,
    volumes: dict[int, float],
    session: volume.Session = STOCK,
    close: Callable[[int], float] | None = None,
) -> list[dict[str, Any]]:
    """One bar for each bucket given, stamped at the start of that bucket in the session's zone."""
    opens = datetime.combine(day, time(), tzinfo=session.zone) + timedelta(minutes=session.opens)
    return [
        {
            "t": at(opens + timedelta(minutes=5 * index)),
            "v": size,
            **({} if close is None else {"c": close(index)}),
        }
        for index, size in volumes.items()
    ]


def normal_day() -> dict[int, float]:
    """100 units: 20 in the first bucket, 4 in the last and 1 in each of the 76 between."""
    return {0: 20.0, **dict.fromkeys(range(1, 77), 1.0), 77: 4.0}


@pytest.fixture(autouse=True)
def fresh_cache() -> None:
    volume.PROFILES.clear()
    volume.SPREADS.clear()


@pytest.fixture
def clients() -> tuple[MagicMock, MagicMock]:
    stocks: MagicMock = create_autospec(StockHistoricalDataClient, instance=True)
    crypto: MagicMock = create_autospec(CryptoHistoricalDataClient, instance=True)
    return stocks, crypto


def sessions(count: int, first: date = date(2026, 10, 1)) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in range(count):
        rows += session_bars(first + timedelta(days=offset), normal_day())
    return rows


@pytest.mark.parametrize(
    ("when", "bucket"),
    [
        (datetime(2026, 11, 3, 14, 30, tzinfo=UTC), 0),
        (datetime(2026, 11, 3, 14, 34, 59, tzinfo=UTC), 0),
        (datetime(2026, 11, 3, 14, 35, tzinfo=UTC), 1),
        (datetime(2026, 11, 3, 20, 55, tzinfo=UTC), 77),
        (datetime(2026, 7, 15, 13, 30, tzinfo=UTC), 0),
        (datetime(2026, 7, 15, 19, 55, tzinfo=UTC), 77),
    ],
)
def test_a_stock_bucket_counts_from_the_open_in_new_york_whatever_the_season(
    when: datetime, bucket: int
) -> None:
    assert STOCK.bucket_of(when) == bucket


def test_a_coin_bucket_counts_from_midnight_utc() -> None:
    assert COIN.bucket_of(datetime(2026, 11, 3, 0, 5, tzinfo=UTC)) == 1
    assert COIN.bucket_of(datetime(2026, 11, 3, 23, 55, tzinfo=UTC)) == 287
    assert COIN.buckets == 288


def test_each_full_session_becomes_shares_of_its_own_day_oldest_first() -> None:
    rows = session_bars(date(2026, 10, 6), normal_day()) + session_bars(
        date(2026, 10, 5), {0: 1.0, 1: 3.0, 77: 1.0}
    )
    first, second = (day.shares() for day in volume.read_days(rows, STOCK))
    assert (first[0], first[1], first[77]) == (0.2, 0.6, 0.2)
    assert (second[0], second[1], second[77]) == (0.2, 0.01, 0.04)
    assert sum(first) == pytest.approx(1) and sum(second) == pytest.approx(1)


def test_trading_outside_the_regular_session_is_left_out() -> None:
    day = date(2026, 10, 6)
    rows = session_bars(day, {0: 5.0, 77: 5.0})
    pre_market = datetime.combine(day, time(8, 0), tzinfo=STOCK.zone)
    closing_print = datetime.combine(day, time(16, 0), tzinfo=STOCK.zone)
    rows += [{"t": at(pre_market), "v": 1000.0}, {"t": at(closing_print), "v": 1000.0}]
    [counted] = volume.read_days(rows, STOCK)
    assert (counted.shares()[0], counted.shares()[77]) == (0.5, 0.5)


def test_a_half_day_has_no_volume_at_the_end_and_is_left_out() -> None:
    half = session_bars(date(2026, 11, 27), dict.fromkeys(range(0, 42), 1.0))
    assert len(volume.read_days(half + sessions(1), STOCK)) == 1


def test_a_profile_is_the_mean_of_the_last_twenty_full_sessions(
    clients: tuple[MagicMock, MagicMock],
) -> None:
    stocks, crypto = clients
    stocks.get_stock_bars.return_value = {"F": sessions(25)}
    found = volume.profile(stocks, crypto, "F", NOW)
    assert found.sessions == 20
    assert (found.shares[0], found.shares[1], found.shares[77]) == pytest.approx((0.2, 0.01, 0.04))
    assert sum(found.shares) == pytest.approx(1)


def test_the_bars_come_from_the_sip_feed_in_five_minutes_and_stop_before_today(
    clients: tuple[MagicMock, MagicMock],
) -> None:
    stocks, crypto = clients
    stocks.get_stock_bars.return_value = {"F": sessions(6)}
    volume.profile(stocks, crypto, "F", NOW)
    request = stocks.get_stock_bars.call_args.args[0]
    assert request.feed == DataFeed.SIP and request.adjustment == Adjustment.RAW
    assert (request.timeframe.amount, request.timeframe.unit.value) == (5, "Min")
    midnight = datetime(2026, 11, 3, 5, 0)
    assert request.end == midnight and request.start == midnight - timedelta(days=45)
    crypto.get_crypto_bars.assert_not_called()


def test_too_few_sessions_is_refused_with_the_count(clients: tuple[MagicMock, MagicMock]) -> None:
    stocks, crypto = clients
    stocks.get_stock_bars.return_value = {"F": sessions(3)}
    with pytest.raises(NotEnoughHistoryError, match="3 full sessions of F; a profile needs 5"):
        volume.profile(stocks, crypto, "F", NOW)


def test_a_symbol_with_no_bars_is_refused(clients: tuple[MagicMock, MagicMock]) -> None:
    stocks, crypto = clients
    stocks.get_stock_bars.return_value = {}
    with pytest.raises(NotEnoughHistoryError, match="0 full sessions"):
        volume.profile(stocks, crypto, "ZZZZ", NOW)


def test_a_profile_is_read_once_a_day_for_each_symbol(clients: tuple[MagicMock, MagicMock]) -> None:
    stocks, crypto = clients
    stocks.get_stock_bars.return_value = {"F": sessions(6)}
    volume.profile(stocks, crypto, "F", NOW)
    volume.profile(stocks, crypto, "F", NOW + timedelta(minutes=40))
    assert stocks.get_stock_bars.call_count == 1
    volume.profile(stocks, crypto, "F", NOW + timedelta(days=1))
    assert stocks.get_stock_bars.call_count == 2


def test_a_coin_profile_uses_the_crypto_bars_over_the_utc_day(
    clients: tuple[MagicMock, MagicMock],
) -> None:
    stocks, crypto = clients
    day = dict.fromkeys(range(288), 1.0)
    rows = [
        row
        for offset in range(6)
        for row in session_bars(date(2026, 10, 1) + timedelta(days=offset), day, COIN)
    ]
    crypto.get_crypto_bars.return_value = {"BTC/USD": rows}
    found = volume.profile(stocks, crypto, "BTC/USD", NOW)
    assert (found.sessions, len(found.shares)) == (6, 288)
    assert found.shares[0] == pytest.approx(1 / 288)
    stocks.get_stock_bars.assert_not_called()
    assert crypto.get_crypto_bars.call_args.args[0].end == datetime(2026, 11, 3)


def test_the_share_at_a_time_clamps_to_the_session() -> None:
    shares = tuple(float(index) for index in range(78))
    found = Profile(shares=shares, sessions=20, session=STOCK)
    assert found.share_at(datetime(2026, 11, 3, 14, 40, tzinfo=UTC)) == 2.0
    assert found.share_at(datetime(2026, 11, 3, 13, 0, tzinfo=UTC)) == 0.0
    assert found.share_at(datetime(2026, 11, 3, 22, 0, tzinfo=UTC)) == 77.0


def test_the_price_steps_of_a_day_count_a_bucket_without_trades_as_no_change() -> None:
    day = volume.Day(volumes=[1.0] * 5, closes=[None, 10.0, None, None, 10.3])
    assert day.steps() == pytest.approx([0.0, 0.0, 0.3])
    assert volume.Day(volumes=[1.0], closes=[10.0]).steps() == []


def test_a_profile_carries_the_daily_volume_and_the_volatility_of_a_five_minute_step(
    clients: tuple[MagicMock, MagicMock],
) -> None:
    stocks, crypto = clients
    rows = [
        row
        for offset in range(6)
        for row in session_bars(
            date(2026, 10, 1) + timedelta(days=offset),
            normal_day(),
            close=lambda index: 100.0 + 0.1 * (index % 2),
        )
    ]
    stocks.get_stock_bars.return_value = {"F": rows}
    found = volume.profile(stocks, crypto, "F", NOW)
    assert found.volume == pytest.approx(100.0)
    assert found.step == pytest.approx(0.1, rel=0.02)


def quote_rows(*pairs: tuple[float, float]) -> dict[str, Any]:
    return {"F": [{"bp": bid, "ap": ask} for bid, ask in pairs]}


def test_the_spread_is_the_median_of_the_valid_quotes_of_the_latest_half_hour(
    clients: tuple[MagicMock, MagicMock],
) -> None:
    stocks, _ = clients
    stocks.get_stock_quotes.return_value = quote_rows(
        (10.00, 10.05), (10.00, 10.02), (10.01, 10.00), (0.0, 0.0), (10.00, 10.03)
    )
    assert volume.spread(stocks, "F", NOW) == pytest.approx(0.03)
    request = stocks.get_stock_quotes.call_args.args[0]
    assert request.feed == DataFeed.SIP and request.limit == 1000
    assert request.sort.value == "desc"
    assert request.end == datetime(2026, 11, 3, 14, 44) and request.start == datetime(
        2026, 11, 3, 14, 14
    )


def test_a_stock_without_usable_quotes_has_no_spread(clients: tuple[MagicMock, MagicMock]) -> None:
    stocks, _ = clients
    stocks.get_stock_quotes.return_value = {}
    assert volume.spread(stocks, "F", NOW) is None
    stocks.get_stock_quotes.return_value = quote_rows((10.0, 10.0))
    assert volume.spread(stocks, "G", NOW) is None


def test_a_spread_is_read_once_a_minute(clients: tuple[MagicMock, MagicMock]) -> None:
    stocks, _ = clients
    stocks.get_stock_quotes.return_value = quote_rows((10.00, 10.02))
    volume.spread(stocks, "F", NOW)
    volume.spread(stocks, "F", NOW + timedelta(seconds=5))
    assert stocks.get_stock_quotes.call_count == 1
