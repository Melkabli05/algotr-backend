from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from cachetools import TTLCache

from app.features.book import metrics
from app.features.book.service import (
    RENEW_GAP,
    SNAPSHOT_WAIT,
    USUAL_SECONDS,
    BookDesk,
    BookNotFoundError,
    BookNotReadyError,
)
from app.features.book.store import Sample
from app.features.book.tests.support import ASKS, BIDS, message, snapshot
from app.features.book.tests.test_book import Clock

NOON = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


class Renewals:
    def __init__(self) -> None:
        self.symbols: list[str] = []

    def __call__(self, symbol: str) -> None:
        self.symbols.append(symbol)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def renewals() -> Renewals:
    return Renewals()


@pytest.fixture
def desk(tmp_path: Path, clock: Clock, renewals: Renewals) -> BookDesk:
    found = BookDesk(["BTC/USD", "ETH/USD"], tmp_path / "book", clock)
    found.renew = renewals
    return found


def test_a_view_has_the_best_levels_with_a_running_total_and_the_spread(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot())
    view = desk.view("BTC/USD", 3)
    assert [level.total for level in view.bids] == [1.0, 3.0, 7.0]
    assert [level.total for level in view.asks] == [1.0, 3.0, 7.0]
    assert (view.mid, view.spread) == (100.25, 0.5)
    assert view.spread_bps == pytest.approx(0.5 / 100.25 * 1e4)
    assert view.time == datetime(2026, 11, 3, 16, 0, 0, 500000, tzinfo=UTC)


def test_a_coin_that_is_not_followed_is_not_found(desk: BookDesk) -> None:
    with pytest.raises(BookNotFoundError):
        desk.view("DOGE/USD")
    with pytest.raises(BookNotFoundError):
        desk.history("DOGE/USD", 5)


def test_a_book_without_a_snapshot_is_not_ready(desk: BookDesk) -> None:
    desk.keeper.handle(message(bids=[(100.0, 1.0)]))
    with pytest.raises(BookNotReadyError):
        desk.view("BTC/USD")


def test_a_book_goes_quiet_when_the_stream_does(desk: BookDesk, clock: Clock) -> None:
    desk.keeper.handle(snapshot())
    clock.now += 6
    with pytest.raises(BookNotReadyError):
        desk.view("BTC/USD")
    assert desk.status().stale


def test_the_status_lists_each_book_with_its_levels(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot("BTC/USD"))
    status = desk.status()
    ready = {book.symbol: book.ready for book in status.books}
    assert ready == {"BTC/USD": True, "ETH/USD": False}
    assert (status.books[0].bid_levels, status.books[0].ask_levels) == (4, 4)
    assert not status.stale


def test_a_sample_stores_each_ready_book_and_only_those(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot("BTC/USD"))
    desk.sample(NOON)
    [point] = desk.store.read("BTC/USD", NOON - timedelta(seconds=1), NOON + timedelta(seconds=1))
    assert (point.mid, point.spread) == (100.25, 0.5) and len(point.bids) == 4
    assert (
        desk.store.read("ETH/USD", NOON - timedelta(seconds=1), NOON + timedelta(seconds=1)) == []
    )


def test_a_crossed_book_is_not_stored_and_a_new_snapshot_is_asked_for(
    desk: BookDesk, renewals: Renewals
) -> None:
    desk.keeper.handle(snapshot())
    desk.keeper.handle(message(bids=[(100.5, 1.0)]))
    desk.sample(NOON)
    assert (
        desk.store.read("BTC/USD", NOON - timedelta(seconds=1), NOON + timedelta(seconds=1)) == []
    )
    assert renewals.symbols == ["BTC/USD"]
    assert not desk.keeper.books["BTC/USD"].ready


def test_a_silent_stream_makes_the_books_not_ready(desk: BookDesk, clock: Clock) -> None:
    desk.keeper.handle(snapshot())
    clock.now += 6
    desk.sample(NOON)
    assert not desk.keeper.books["BTC/USD"].ready


def test_a_book_still_waiting_for_its_snapshot_is_renewed_once_in_a_while(
    desk: BookDesk, clock: Clock, renewals: Renewals
) -> None:
    desk.sample(NOON)
    assert renewals.symbols == []
    clock.now += SNAPSHOT_WAIT + 1
    desk.sample(NOON)
    assert sorted(renewals.symbols) == ["BTC/USD", "ETH/USD"]
    clock.now += SNAPSHOT_WAIT + 1
    desk.sample(NOON)
    assert len(renewals.symbols) == 2
    clock.now += RENEW_GAP
    desk.sample(NOON)
    assert len(renewals.symbols) == 4


def test_a_book_that_has_its_snapshot_is_never_renewed(
    desk: BookDesk, clock: Clock, renewals: Renewals
) -> None:
    desk.keeper.handle(snapshot("BTC/USD"))
    desk.keeper.handle(snapshot("ETH/USD"))
    for _ in range(5):
        clock.now += 1
        desk.keeper.handle(message(bids=[(100.0, 1.5)]))
        desk.keeper.handle(message("ETH/USD", bids=[(100.0, 1.5)]))
        desk.sample(NOON)
    assert renewals.symbols == []


def test_the_history_is_the_mid_and_spread_of_each_second_in_the_window(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot())
    now = datetime.now(UTC)
    desk.sample(now - timedelta(minutes=2))
    desk.sample(now - timedelta(seconds=30))
    assert len(desk.history("BTC/USD", 5)) == 2
    [recent] = desk.history("BTC/USD", 1)
    assert recent.spread_bps == pytest.approx(0.5 / 100.25 * 1e4)


def test_old_days_are_pruned_once_a_day(desk: BookDesk, monkeypatch: pytest.MonkeyPatch) -> None:
    pruned: list[date] = []

    def record(today: date) -> int:
        pruned.append(today)
        return 0

    monkeypatch.setattr(desk.store, "prune", record)
    desk.sample(NOON)
    desk.sample(NOON + timedelta(seconds=1))
    desk.sample(NOON + timedelta(days=1))
    assert [day.isoformat() for day in pruned] == ["2026-11-03", "2026-11-04"]


def test_a_view_reads_the_imbalance_by_levels_and_by_distance(desk: BookDesk) -> None:
    desk.keeper.handle(
        message(
            "BTC/USD",
            [(100.0, 6.0), (99.5, 2.0), (99.0, 2.0)],
            [(100.5, 1.0), (101.0, 1.0), (102.0, 2.0)],
            reset=True,
        )
    )
    view = desk.view("BTC/USD")
    by_levels = {item.depth: item for item in view.imbalance_levels}
    assert sorted(by_levels) == [5, 10, 25]
    assert by_levels[5].value == pytest.approx((10 - 4) / 14) and by_levels[5].pressure == "buying"
    by_bands = {item.depth: item.value for item in view.imbalance_bands}
    assert sorted(by_bands) == [10, 25, 50, 100]
    assert by_bands[10] is None
    assert by_bands[25] == pytest.approx((6 - 1) / 7) and by_bands[50] == by_bands[25]
    assert by_bands[100] == pytest.approx((8 - 2) / 10)


def test_a_balanced_book_reads_as_balanced(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot())
    view = desk.view("BTC/USD")
    assert all(item.pressure == "balanced" for item in view.imbalance_levels)


def test_a_view_shows_only_the_levels_asked_for_but_reads_the_imbalance_of_the_whole_book(
    desk: BookDesk,
) -> None:
    desk.keeper.handle(snapshot())
    view = desk.view("BTC/USD", 1)
    assert len(view.bids) == 1 and len(view.asks) == 1
    assert view.imbalance_levels[2].depth == 25 and view.imbalance_levels[2].value == 0.0


def test_the_history_carries_the_imbalance_of_each_second(desk: BookDesk) -> None:
    desk.keeper.handle(message("BTC/USD", [(100.0, 3.0)], [(100.5, 1.0)], reset=True))
    desk.sample(datetime.now(UTC) - timedelta(seconds=30))
    [point] = desk.history("BTC/USD", 1)
    assert [item.depth for item in point.imbalance] == [5, 10, 25]
    assert point.imbalance[0].value == pytest.approx(0.5)


def test_the_first_window_only_starts_the_clock(desk: BookDesk, clock: Clock) -> None:
    desk.keeper.handle(snapshot())
    desk.sample(NOON)
    assert desk.flow("BTC/USD") is None


def test_a_window_closes_every_fifth_second_with_the_flow_of_those_seconds(
    desk: BookDesk, clock: Clock
) -> None:
    desk.keeper.handle(snapshot())
    desk.sample(NOON)
    desk.keeper.handle(message(bids=[(100.0, 4.0)]))
    for second in range(1, 5):
        desk.sample(NOON + timedelta(seconds=second))
    assert desk.flow("BTC/USD") is None
    desk.sample(NOON + timedelta(seconds=5))
    flow = desk.flow("BTC/USD")
    assert flow is not None and flow.ofi == 3.0 and flow.windows == 1
    [stored] = desk.store.read_flows("BTC/USD", NOON, NOON + timedelta(seconds=6))
    assert (stored.ofi, stored.span) == (3.0, 5.0)


def test_the_last_minute_is_the_sum_of_the_last_twelve_windows(
    desk: BookDesk, clock: Clock
) -> None:
    desk.keeper.handle(snapshot())
    desk.sample(NOON)
    for window in range(1, 15):
        desk.keeper.handle(message(bids=[(100.0, 1.0 + window)]))
        for second in range(5):
            desk.sample(NOON + timedelta(seconds=5 * (window - 1) + second + 1))
    flow = desk.flow("BTC/USD")
    assert flow is not None and flow.windows == 12
    assert flow.ofi == 1.0 and flow.ofi_60 == 12.0


def test_a_book_that_is_not_ready_has_no_window(desk: BookDesk) -> None:
    desk.sample(NOON)
    desk.sample(NOON + timedelta(seconds=5))
    assert desk.flow("BTC/USD") is None and desk.flow("ETH/USD") is None


def test_the_view_carries_the_flow(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot())
    assert desk.view("BTC/USD").flow is None
    desk.sample(NOON)
    desk.keeper.handle(message(asks=[(100.5, 3.0)]))
    desk.sample(NOON + timedelta(seconds=5))
    flow = desk.view("BTC/USD").flow
    assert flow is not None and flow.ofi == -2.0


def test_the_flows_of_the_last_minutes_are_listed_for_a_followed_coin(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot())
    now = datetime.now(UTC) - timedelta(seconds=30)
    desk.sample(now)
    desk.keeper.handle(message(bids=[(100.0, 4.0)]))
    desk.sample(now + timedelta(seconds=5))
    [point] = desk.flows("BTC/USD", 5)
    assert point.ofi == 3.0 and point.span == 5.0
    with pytest.raises(BookNotFoundError):
        desk.flows("DOGE/USD", 5)


def test_a_late_sample_still_closes_the_window_and_records_how_long_it_was(
    desk: BookDesk,
) -> None:
    desk.keeper.handle(snapshot())
    desk.sample(NOON)
    desk.keeper.handle(message(bids=[(100.0, 4.0)]))
    desk.sample(NOON + timedelta(seconds=4.4))
    assert desk.flow("BTC/USD") is None
    desk.sample(NOON + timedelta(seconds=7.2))
    [stored] = desk.store.read_flows("BTC/USD", NOON, NOON + timedelta(seconds=8))
    assert stored.span == pytest.approx(7.2)


def usual(desk: BookDesk, spread: float, count: int = 61) -> None:
    now = datetime.now(UTC)
    desk.store.add(
        [
            Sample(now - timedelta(seconds=second + 1), "BTC/USD", 100.25, spread, BIDS, ASKS)
            for second in range(count)
        ]
    )


@pytest.fixture
def guarded(tmp_path: Path, clock: Clock) -> BookDesk:
    found = BookDesk(
        ["BTC/USD", "ETH/USD"],
        tmp_path / "book",
        clock,
        max_spread_multiple=3.0,
        max_slippage_bps=100.0,
        slice_slippage_bps=20.0,
    )
    found.keeper.handle(snapshot())
    return found


def test_a_view_counts_the_depth_within_each_distance_of_the_mid(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot())
    found = {item.bps: item for item in desk.view("BTC/USD").depth}
    assert sorted(found) == [10, 25, 50, 100]
    assert (found[10].bid, found[10].ask, found[10].bid_value, found[10].ask_value) == (0, 0, 0, 0)
    assert (found[25].bid, found[25].ask) == (1.0, 1.0)
    assert (found[25].bid_value, found[25].ask_value) == (100.0, 100.5)
    assert (found[100].bid, found[100].ask) == (3.0, 3.0)
    assert (found[100].bid_value, found[100].ask_value) == (299.0, 302.5)


def test_the_usual_spread_is_the_median_of_the_last_hour_and_the_view_compares_with_it(
    desk: BookDesk,
) -> None:
    desk.keeper.handle(snapshot())
    usual(desk, 0.1)
    view = desk.view("BTC/USD")
    assert view.spread_usual_bps == pytest.approx(0.1 / 100.25 * 1e4)
    assert view.spread_ratio == pytest.approx(5.0)


def test_a_minute_of_history_is_the_least_that_makes_a_usual_spread(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot())
    usual(desk, 0.1, count=59)
    view = desk.view("BTC/USD")
    assert view.spread_usual_bps is None and view.spread_ratio is None
    desk.usual.clear()
    usual(desk, 0.1, count=60)
    assert desk.view("BTC/USD").spread_usual_bps is not None


def test_a_burst_of_wide_seconds_does_not_move_the_usual_spread(desk: BookDesk) -> None:
    usual(desk, 0.1, count=40)
    now = datetime.now(UTC)
    desk.store.add(
        [
            Sample(now - timedelta(seconds=100 + n), "BTC/USD", 100.25, 5.0, BIDS, ASKS)
            for n in range(20)
        ]
    )
    assert desk.usual_spread("BTC/USD") == pytest.approx(0.1 / 100.25 * 1e4)


def test_the_usual_spread_is_read_from_the_store_once_a_minute(
    desk: BookDesk, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    desk.usual = TTLCache(maxsize=16, ttl=USUAL_SECONDS, timer=clock)
    reads: list[str] = []
    real = desk.store.spreads

    def counting(symbol: str, since: datetime) -> list[float]:
        reads.append(symbol)
        return real(symbol, since)

    monkeypatch.setattr(desk.store, "spreads", counting)
    desk.usual_spread("BTC/USD")
    desk.usual_spread("BTC/USD")
    desk.usual_spread("ETH/USD")
    assert reads == ["BTC/USD", "ETH/USD"]
    clock.now += USUAL_SECONDS + 1
    desk.usual_spread("BTC/USD")
    assert reads == ["BTC/USD", "ETH/USD", "BTC/USD"]


def test_a_buy_walks_the_asks_and_a_sell_walks_the_bids(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot())
    buy = desk.impact("BTC/USD", "buy", 3.0)
    assert (buy.filled, buy.levels, buy.complete, buy.side) == (3.0, 2, True, "buy")
    assert buy.average == pytest.approx(302.5 / 3)
    assert buy.slippage_bps == pytest.approx((302.5 / 3 - 100.25) / 100.25 * 1e4)
    sell = desk.impact("BTC/USD", "sell", 3.0)
    assert sell.average == pytest.approx(299 / 3)
    assert sell.slippage_bps == pytest.approx((100.25 - 299 / 3) / 100.25 * 1e4)


def test_an_order_larger_than_the_book_fills_what_there_is(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot())
    impact = desk.impact("BTC/USD", "buy", 20.0)
    assert (impact.filled, impact.levels, impact.complete) == (15.0, 4, False)


def test_with_no_limits_set_the_advice_is_a_market_order(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot())
    usual(desk, 0.01)
    impact = desk.impact("BTC/USD", "buy", 20.0)
    assert (impact.advice, impact.reason, impact.max_qty) == ("market", None, None)
    assert desk.refusal("BTC/USD", "buy", 20.0) is None


def test_the_impact_of_a_book_that_is_not_ready_is_not_known(desk: BookDesk) -> None:
    with pytest.raises(BookNotReadyError):
        desk.impact("ETH/USD", "buy", 1.0)
    with pytest.raises(BookNotFoundError):
        desk.impact("DOGE/USD", "buy", 1.0)


def test_an_order_within_the_limits_is_a_market_order(guarded: BookDesk) -> None:
    impact = guarded.impact("BTC/USD", "buy", 1.0)
    assert (impact.advice, impact.reason) == ("market", None)
    assert guarded.refusal("BTC/USD", "buy", 1.0) is None


def test_an_order_that_would_pay_too_much_is_advised_to_be_a_limit(guarded: BookDesk) -> None:
    impact = guarded.impact("BTC/USD", "buy", 7.0)
    assert impact.advice == "limit"
    assert impact.reason == (
        "A market buy of 7 BTC/USD would pay about 124.7 bps over the mid; "
        "use a limit order or a smaller size"
    )
    assert guarded.refusal("BTC/USD", "buy", 7.0) == impact.reason


def test_an_order_the_book_cannot_fill_is_advised_to_be_a_limit(guarded: BookDesk) -> None:
    reason = guarded.refusal("BTC/USD", "sell", 20.0)
    assert reason == (
        "The book of BTC/USD holds 15 on that side; a market sell of 20 is too large, "
        "use a limit order"
    )


def test_a_spread_several_times_the_usual_is_advised_to_be_a_limit(guarded: BookDesk) -> None:
    usual(guarded, 0.1)
    impact = guarded.impact("BTC/USD", "buy", 1.0)
    assert impact.advice == "limit" and impact.spread_ratio == pytest.approx(5.0)
    assert impact.reason == (
        "The spread of BTC/USD is 49.9 bps, 5.0 times its usual 10.0; use a limit order"
    )


def test_a_spread_within_the_usual_range_passes(guarded: BookDesk) -> None:
    usual(guarded, 0.4)
    impact = guarded.impact("BTC/USD", "buy", 1.0)
    assert impact.advice == "market" and impact.spread_ratio == pytest.approx(1.25)


def test_without_a_minute_of_history_the_spread_rule_waits(guarded: BookDesk) -> None:
    assert guarded.impact("BTC/USD", "buy", 1.0).spread_ratio is None
    assert guarded.refusal("BTC/USD", "buy", 1.0) is None


def test_the_slice_budget_is_measured_past_the_best_price_and_is_used_to_the_full(
    guarded: BookDesk,
) -> None:
    buy = guarded.impact("BTC/USD", "buy", 1.0).max_qty
    sell = guarded.impact("BTC/USD", "sell", 1.0).max_qty
    assert buy is not None and sell is not None
    assert metrics.walk(ASKS, buy).average == pytest.approx(100.5 * 1.002)
    assert metrics.walk(BIDS, sell).average == pytest.approx(100.0 * 0.998)
    assert guarded.max_size("BTC/USD", "buy") == buy
    assert guarded.max_size("BTC/USD", "sell") == sell


def test_a_book_with_a_wide_spread_still_has_a_slice_size(guarded: BookDesk) -> None:
    guarded.keeper.handle(message(bids=[(90.0, 1.0)], asks=[(110.0, 1.0)], reset=True))
    size = guarded.max_size("BTC/USD", "buy")
    assert size is not None and size >= 1.0


def test_each_limit_is_off_at_zero(desk: BookDesk) -> None:
    desk.keeper.handle(snapshot())
    assert desk.max_size("BTC/USD", "buy") is None
    assert desk.impact("BTC/USD", "buy", 7.0).advice == "market"


def test_a_guard_that_cannot_see_the_book_lets_the_order_through(
    guarded: BookDesk, clock: Clock
) -> None:
    assert guarded.refusal("ETH/USD", "buy", 1000.0) is None
    assert guarded.max_size("ETH/USD", "buy") is None
    assert guarded.refusal("DOGE/USD", "buy", 1000.0) is None
    assert guarded.max_size("DOGE/USD", "buy") is None
    clock.now += 6
    assert guarded.refusal("BTC/USD", "buy", 1000.0) is None
    assert guarded.max_size("BTC/USD", "buy") is None


def leaning(desk: BookDesk, ofi_60: float, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.features.book.schemas import FlowView

    view = FlowView(time=NOON, ofi=ofi_60, ofi_60=ofi_60, windows=12)
    monkeypatch.setattr(desk, "flow", lambda symbol: view)


ASK_HEAVY = ([(100.0, 1.0)], [(100.5, 9.0)])


def test_a_market_buy_into_a_book_and_flow_that_both_sell_is_refused(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    desk = BookDesk(["BTC/USD"], tmp_path / "book", clock, max_imbalance=0.6)
    leaning(desk, -5.0, monkeypatch)
    reason = desk.against("BTC/USD", "buy", *ASK_HEAVY)
    assert reason is not None and "leans 80% to selling" in reason
    assert desk.against("BTC/USD", "sell", *ASK_HEAVY) is None


def test_the_book_alone_or_the_flow_alone_is_not_enough(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    desk = BookDesk(["BTC/USD"], tmp_path / "book", clock, max_imbalance=0.6)
    leaning(desk, 5.0, monkeypatch)
    assert desk.against("BTC/USD", "buy", *ASK_HEAVY) is None
    leaning(desk, -5.0, monkeypatch)
    assert desk.against("BTC/USD", "buy", [(100.0, 4.0)], [(100.5, 6.0)]) is None


def test_without_a_limit_or_a_flow_nothing_is_refused(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    off = BookDesk(["BTC/USD"], tmp_path / "book", clock)
    leaning(off, -5.0, monkeypatch)
    assert off.against("BTC/USD", "buy", *ASK_HEAVY) is None
    quiet = BookDesk(["BTC/USD"], tmp_path / "quiet", clock, max_imbalance=0.6)
    assert quiet.against("BTC/USD", "buy", *ASK_HEAVY) is None
