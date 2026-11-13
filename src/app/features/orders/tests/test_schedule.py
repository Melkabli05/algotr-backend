from datetime import UTC, datetime, timedelta

import pytest

from app.features.orders import schedule
from app.quant.volume import STOCK, Profile

START = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)


def test_a_twap_is_equal_intervals_of_equal_weight() -> None:
    spans = schedule.uniform(START, 10, 5)
    assert [span.start for span in spans] == [START + timedelta(minutes=m) for m in (0, 2, 4, 6, 8)]
    assert {span.weight for span in spans} == {1.0}


def test_equal_weights_give_equal_slices() -> None:
    assert schedule.preview(10, [1.0] * 5, whole=True) == [2, 2, 2, 2, 2]


def test_weights_give_slices_in_proportion() -> None:
    assert schedule.preview(100, [1.0, 3.0, 1.0], whole=True) == [20, 60, 20]


@pytest.mark.parametrize("qty", [1, 7, 100, 1001])
@pytest.mark.parametrize(
    "weights", [[1.0] * 2, [1.0] * 13, [3.0, 1.0, 1.0, 2.0, 5.0], [0.0, 1.0, 0.0, 2.0]]
)
def test_whole_slices_always_add_up_to_the_quantity(qty: int, weights: list[float]) -> None:
    sizes = schedule.preview(qty, weights, whole=True)
    assert sum(sizes) == qty
    assert all(size >= 0 and float(size).is_integer() for size in sizes)


def test_coin_slices_add_up_to_the_quantity_to_nine_places() -> None:
    sizes = schedule.preview(1.0, [1.0, 1.0, 1.0], whole=False)
    assert sizes[:2] == [0.333333333, 0.333333333]
    assert sum(sizes) == pytest.approx(1.0, abs=1e-9)


def test_a_short_fill_is_spread_over_the_intervals_that_remain() -> None:
    assert schedule.next_slice(12, [1.0] * 5, 0, whole=True) == 2
    assert schedule.next_slice(12, [1.0] * 5, 1, whole=True) == 3


def test_a_skipped_interval_leaves_its_share_to_the_rest() -> None:
    assert schedule.next_slice(10, [1.0] * 5, 3, whole=True) == 5


def test_the_last_interval_takes_everything_left() -> None:
    assert schedule.next_slice(7, [1.0, 1.0, 1.0], 2, whole=True) == 7
    assert schedule.next_slice(7, [1.0, 1.0, 1.0], 9, whole=True) == 7


def test_with_no_weight_ahead_everything_is_sent() -> None:
    assert schedule.next_slice(5, [0.0, 0.0, 0.0], 0, whole=True) == 5


def test_leaning_grows_or_shrinks_a_slice_but_never_past_what_is_left() -> None:
    assert schedule.next_slice(10, [1.0] * 5, 0, whole=False, lean=0.5) == pytest.approx(3.0)
    assert schedule.next_slice(10, [1.0] * 5, 0, whole=False, lean=-0.5) == pytest.approx(1.0)
    assert schedule.next_slice(1, [1.0, 1.0], 0, whole=False, lean=5.0) == 1


OPEN = datetime(2026, 11, 3, 14, 30, tzinfo=UTC)
PROFILE = Profile(shares=(0.5, 0.3, 0.2, *([0.0] * 75)), sessions=20, session=STOCK)


def test_a_vwap_is_cut_at_the_five_minute_marks_and_weighted_by_the_profile() -> None:
    spans = schedule.volume_weighted(OPEN, OPEN + timedelta(minutes=15), PROFILE)
    assert [span.start for span in spans] == [OPEN + timedelta(minutes=m) for m in (0, 5, 10)]
    assert [span.weight for span in spans] == [0.5, 0.3, 0.2]


def test_a_part_bucket_at_either_end_counts_in_proportion_to_its_length() -> None:
    begin = OPEN + timedelta(minutes=3, seconds=30)
    spans = schedule.volume_weighted(begin, OPEN + timedelta(minutes=12), PROFILE)
    assert [span.start for span in spans] == [begin] + [
        OPEN + timedelta(minutes=m) for m in (5, 10)
    ]
    assert [span.weight for span in spans] == pytest.approx([0.15, 0.3, 0.08])


def test_vwap_slices_follow_the_weights() -> None:
    spans = schedule.volume_weighted(OPEN, OPEN + timedelta(minutes=15), PROFILE)
    assert schedule.preview(100, [span.weight for span in spans], whole=True) == [50, 30, 20]
