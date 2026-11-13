import math

import pytest

from app.features.orders.estimate import square_root_bps, touch_probability


def test_a_limit_at_the_price_is_touched_and_one_far_away_almost_never() -> None:
    assert touch_probability(0.0, 0.02, 1.0) == 1.0
    assert touch_probability(0.2, 0.02, 1.0) < 1e-6
    assert touch_probability(0.02, 0.02, 1.0) == pytest.approx(2 * (1 - 0.8413447), rel=1e-4)


def test_less_time_means_less_chance() -> None:
    assert touch_probability(0.01, 0.02, 0.25) < touch_probability(0.01, 0.02, 1.0)


def test_square_root_impact_grows_with_the_root_of_participation() -> None:
    low, high = square_root_bps(0.02, 0.01)
    assert (low, high) == pytest.approx((10.0, 30.0))
    quadrupled = square_root_bps(0.02, 0.04)
    assert quadrupled[0] / low == pytest.approx(math.sqrt(4))
