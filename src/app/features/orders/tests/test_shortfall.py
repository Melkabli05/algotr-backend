import math

import pytest

from app.features.orders import shortfall
from app.features.orders.shortfall import Market

PAPER = Market(sigma=0.95, volume=5e6, spread=0.125)
PAPER_PATH = [1000000.0, 428598.84574702, 182932.81426177, 76295.72161546, 27643.37739691, 0.0]


def test_the_papers_own_example_is_reproduced() -> None:
    path = shortfall.trajectory(1e6, days=5, count=5, risk=2.0, market=PAPER)
    assert path == pytest.approx(PAPER_PATH, abs=1e-6)


def test_the_trajectory_runs_from_everything_to_nothing_and_never_rises() -> None:
    path = shortfall.trajectory(1000, days=0.0769, count=6, risk=1.0, market=PAPER)
    assert path[0] == 1000 and path[-1] == pytest.approx(0, abs=1e-9)
    assert all(later <= earlier for earlier, later in zip(path, path[1:], strict=False))


def test_the_trades_add_up_to_the_quantity() -> None:
    trades = shortfall.weights(1000, days=0.0769, count=6, risk=1.0, market=PAPER)
    assert sum(trades) == pytest.approx(1000)
    assert all(size > 0 for size in trades)


def test_with_no_risk_aversion_the_trajectory_is_a_straight_line() -> None:
    trades = shortfall.weights(100, days=1, count=4, risk=0.0, market=PAPER)
    assert trades == pytest.approx([25, 25, 25, 25])


def test_more_risk_aversion_trades_sooner() -> None:
    market = Market(sigma=3.0, volume=5e7, spread=0.01)
    first_half = []
    for risk in (0.0, 0.5, 1.0, 2.0, 10.0):
        trades = shortfall.weights(1000, days=30 / 390, count=6, risk=risk, market=market)
        first_half.append(sum(trades[:3]) / 1000)
    assert first_half == sorted(first_half) and first_half[0] == pytest.approx(0.5)
    assert first_half[-1] > 0.9


def test_a_liquid_stock_is_worked_down_sooner_than_a_thin_one() -> None:
    liquid = Market(sigma=3.0, volume=5e7, spread=0.05)
    thin = Market(sigma=0.9, volume=1.5e5, spread=0.04)
    mild = shortfall.weights(1000, days=30 / 390, count=6, risk=1.0, market=liquid)
    firm = shortfall.weights(1000, days=30 / 390, count=6, risk=5.0, market=liquid)
    flat = shortfall.weights(1000, days=30 / 390, count=6, risk=5.0, market=thin)
    assert 0.52 < sum(mild[:3]) / 1000 < 0.55
    assert sum(firm[:3]) / 1000 > 0.6
    assert sum(flat[:3]) / 1000 == pytest.approx(0.5, abs=0.01)


def test_an_extreme_decay_stays_finite_and_trades_almost_everything_at_once() -> None:
    market = Market(sigma=1e3, volume=1e3, spread=0.01)
    trades = shortfall.weights(100, days=1, count=5, risk=100.0, market=market)
    assert all(math.isfinite(size) for size in trades)
    assert trades[0] > 99.9 and sum(trades) == pytest.approx(100)


def test_the_hold_ratio_is_the_hyperbolic_one() -> None:
    assert shortfall.held(0.8463, 5, 1) == pytest.approx(0.428598845, rel=1e-4)
    assert shortfall.held(0, 5, 2) == pytest.approx(0.6)
