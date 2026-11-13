import pytest

from app.features.book import metrics

BIDS = [(100.0, 4.0), (99.9, 2.0), (99.0, 2.0), (95.0, 8.0)]
ASKS = [(100.1, 1.0), (100.2, 1.0), (101.0, 2.0), (110.0, 8.0)]


def test_the_imbalance_runs_from_minus_one_to_plus_one() -> None:
    assert metrics.imbalance(1.0, 0.0) == 1.0
    assert metrics.imbalance(0.0, 3.0) == -1.0
    assert metrics.imbalance(2.0, 2.0) == 0.0
    assert metrics.imbalance(3.0, 1.0) == pytest.approx(0.5)


def test_with_no_volume_there_is_no_imbalance() -> None:
    assert metrics.imbalance(0.0, 0.0) is None
    assert metrics.by_levels([], [], 5) is None


def test_the_labels_follow_the_threshold_of_a_third_either_way() -> None:
    assert metrics.pressure(0.31) == "buying"
    assert metrics.pressure(-0.31) == "selling"
    assert metrics.pressure(0.3) == "balanced" and metrics.pressure(-0.3) == "balanced"
    assert metrics.pressure(0.0) == "balanced"
    assert metrics.pressure(None) is None


def test_by_levels_counts_the_best_levels_of_each_side() -> None:
    assert metrics.by_levels(BIDS, ASKS, 2) == pytest.approx((6 - 2) / 8)
    assert metrics.by_levels(BIDS, ASKS, 3) == pytest.approx((8 - 4) / 12)
    assert metrics.by_levels(BIDS, ASKS, 25) == pytest.approx((16 - 12) / 28)


def test_by_band_counts_the_volume_within_a_distance_of_the_mid_on_both_sides() -> None:
    mid = 100.05
    assert metrics.by_band(BIDS, ASKS, mid, 10) == pytest.approx((4 - 1) / 5)
    assert metrics.by_band(BIDS, ASKS, mid, 100) == pytest.approx((6 - 4) / 10)
    assert metrics.by_band(BIDS, ASKS, mid, 105) == pytest.approx((8 - 4) / 12)
    assert metrics.by_band(BIDS, ASKS, mid, 1000) == pytest.approx((16 - 12) / 28)


def test_a_band_with_nothing_in_it_has_no_imbalance() -> None:
    assert metrics.by_band(BIDS, ASKS, 100.05, 1) is None


QUOTE = ((100.0, 2.0), (101.0, 3.0))


def test_a_bid_that_rises_adds_its_new_size() -> None:
    assert metrics.order_flow(QUOTE, ((100.5, 4.0), (101.0, 3.0))) == 4.0


def test_a_bid_that_falls_takes_off_its_old_size() -> None:
    assert metrics.order_flow(QUOTE, ((99.5, 5.0), (101.0, 3.0))) == -2.0


def test_a_bid_that_holds_adds_the_change_in_its_size() -> None:
    assert metrics.order_flow(QUOTE, ((100.0, 5.0), (101.0, 3.0))) == 3.0
    assert metrics.order_flow(QUOTE, ((100.0, 1.0), (101.0, 3.0))) == -1.0


def test_an_ask_that_falls_takes_off_its_new_size() -> None:
    assert metrics.order_flow(QUOTE, ((100.0, 2.0), (100.5, 4.0))) == -4.0


def test_an_ask_that_rises_adds_its_old_size() -> None:
    assert metrics.order_flow(QUOTE, ((100.0, 2.0), (101.5, 5.0))) == 3.0


def test_an_ask_that_holds_subtracts_the_change_in_its_size() -> None:
    assert metrics.order_flow(QUOTE, ((100.0, 2.0), (101.0, 1.0))) == 2.0
    assert metrics.order_flow(QUOTE, ((100.0, 2.0), (101.0, 7.0))) == -4.0


def test_a_buy_that_lifts_the_ask_is_positive_and_a_sell_that_hits_the_bid_negative() -> None:
    assert metrics.order_flow(QUOTE, ((100.0, 2.0), (101.5, 5.0))) > 0
    assert metrics.order_flow(QUOTE, ((99.5, 5.0), (101.0, 3.0))) < 0


def test_when_nothing_at_the_top_changes_there_is_no_flow() -> None:
    assert metrics.order_flow(QUOTE, QUOTE) == 0.0


def test_a_walk_takes_levels_best_first_and_averages_the_price() -> None:
    result = metrics.walk(ASKS, 2.5)
    assert result.filled == 2.5 and result.levels == 3
    assert result.average == pytest.approx((1 * 100.1 + 1 * 100.2 + 0.5 * 101.0) / 2.5)


def test_a_walk_that_fits_in_the_first_level_uses_one() -> None:
    result = metrics.walk(ASKS, 0.4)
    assert (result.filled, result.average, result.levels) == (0.4, 100.1, 1)


def test_a_walk_of_more_than_the_book_holds_fills_what_there_is() -> None:
    result = metrics.walk(ASKS, 100.0)
    assert result.filled == 12.0 and result.levels == 4


def test_a_walk_of_an_empty_book_fills_nothing() -> None:
    assert metrics.walk([], 1.0) == metrics.Walk(0.0, None, 0)


def test_slippage_is_positive_when_the_fill_is_worse_than_the_mid() -> None:
    assert metrics.slippage("buy", 100.1, 100.0) == pytest.approx(10.0)
    assert metrics.slippage("sell", 99.9, 100.0) == pytest.approx(10.0)
    assert metrics.slippage("buy", 99.9, 100.0) == pytest.approx(-10.0)


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_the_largest_size_stays_within_the_budget_and_one_more_does_not(side: str) -> None:
    levels = ASKS if side == "buy" else BIDS
    mid = 100.05
    for bps in (11.0, 20.0, 60.0, 200.0):
        size = metrics.max_size(levels, side, mid, bps)
        average = metrics.walk(levels, size).average
        assert average is not None
        assert metrics.slippage(side, average, mid) <= bps + 1e-6
        over = metrics.walk(levels, size + 0.01)
        if over.filled > size and over.average is not None and size < 12:
            assert metrics.slippage(side, over.average, mid) > bps - 1e-6


def test_a_budget_too_tight_for_even_the_first_level_allows_nothing_beyond_it() -> None:
    assert metrics.max_size(ASKS, "buy", 100.05, 0.0) == 0.0
    assert metrics.max_size([], "buy", 100.05, 50.0) == 0.0


def test_a_wide_budget_allows_the_whole_book() -> None:
    assert metrics.max_size(ASKS, "buy", 100.05, 5000.0) == 12.0


def test_within_counts_the_size_and_the_value_inside_a_distance_of_the_mid() -> None:
    bids = [(100.0, 6.0), (99.5, 2.0), (99.0, 2.0)]
    asks = [(100.5, 1.0), (101.0, 1.0), (102.0, 2.0)]
    assert metrics.within(bids, 100.25, 25, "bid") == (6.0, 600.0)
    assert metrics.within(asks, 100.25, 25, "ask") == (1.0, 100.5)
    assert metrics.within(bids, 100.25, 100, "bid") == (8.0, 799.0)
    assert metrics.within(asks, 100.25, 100, "ask") == (2.0, 201.5)


def test_nothing_is_within_a_distance_the_book_does_not_reach() -> None:
    assert metrics.within([(99.0, 5.0)], 100.25, 10, "bid") == (0.0, 0.0)
    assert metrics.within([(102.0, 5.0)], 100.25, 10, "ask") == (0.0, 0.0)
    assert metrics.within([], 100.25, 100, "bid") == (0.0, 0.0)
