import numpy as np
import pandas as pd
import pytest

from app.features.systematic import signals
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.schemas import StrategySpec
from app.features.systematic.tests.support import downtrend, uptrend


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)), dtype=float)


def frame(**columns: list[float]) -> pd.DataFrame:
    length = len(next(iter(columns.values())))
    index = pd.date_range("2024-01-01", periods=length)
    return pd.DataFrame(columns, index=index, dtype=float)


def momentum_spec(window: int = 5, top_n: int = 2) -> StrategySpec:
    return STRATEGIES["halal-etf-rotation"].model_copy(update={"window": window, "top_n": top_n})


def test_sma_trend_is_long_above_the_average_and_flat_below_it() -> None:
    rising = series(list(range(1, 31)))
    assert signals.sma_trend(rising, 5).iloc[-1] == 1.0
    assert signals.sma_trend(rising[::-1].reset_index(drop=True), 5).iloc[-1] == 0.0


def test_sma_trend_is_flat_until_it_has_a_full_window() -> None:
    assert (signals.sma_trend(series(list(range(1, 11))), 20) == 0.0).all()


@pytest.mark.parametrize(
    "name", ["crypto-donchian", "halal-etf-trend", "halal-etf-rotation", "halal-multi-trend"]
)
def test_a_weight_never_depends_on_later_prices(name: str) -> None:
    spec = STRATEGIES[name]
    prices = uptrend(spec.universe, 600)
    altered = prices.copy()
    altered.iloc[400:] *= 0.2
    before = signals.weights(prices, spec).iloc[:400]
    after = signals.weights(altered, spec).iloc[:400]
    pd.testing.assert_frame_equal(before, after)


def test_donchian_enters_on_a_breakout_and_exits_below_the_midpoint_stop() -> None:
    quiet = [100.0] * 10
    breakout = [105.0, 110.0, 115.0, 120.0]
    pullback = [118.0, 116.0, 100.0]
    state = signals.donchian_state(series(quiet + breakout + pullback), 5)
    assert state.iloc[:10].sum() == 0
    assert state.iloc[10] == 1
    assert state.iloc[13] == 1
    assert state.iloc[-1] == 0


def test_donchian_stays_flat_in_a_downtrend() -> None:
    assert signals.donchian_state(series(list(range(100, 40, -1))), 10).sum() == 0


def test_the_ensemble_is_the_share_of_models_that_are_long() -> None:
    prices = series([100.0] * 30 + [101.0 + step for step in range(40)])
    ensemble = signals.donchian_ensemble(prices, [5, 10, 400])
    assert ensemble.iloc[-1] == pytest.approx(2 / 3)
    assert set(np.round(ensemble.unique() * 3)) <= {0.0, 1.0, 2.0, 3.0}


def test_volatility_scaling_is_capped_at_one_and_shrinks_with_volatility() -> None:
    calm = frame(A=[100 + 0.01 * step for step in range(60)])
    wild = frame(A=[100 * (1.1 if step % 2 else 0.9) for step in range(60)])
    assert signals.volatility_scale(calm, 0.25, 365).iloc[-1] == 1.0
    assert 0 < signals.volatility_scale(wild, 0.25, 365).iloc[-1] < 0.5


def test_volatility_scaling_uses_the_whole_basket_and_is_zero_before_it_has_data() -> None:
    steady = [100 + 0.01 * step for step in range(60)]
    wild = [100 * (1.1 if step % 2 else 0.9) for step in range(60)]
    basket = signals.volatility_scale(frame(A=steady, B=wild), 0.25, 365)
    assert basket.iloc[0] == 0.0
    assert basket.iloc[-1] < 1.0


def test_momentum_holds_the_best_performers_that_gained() -> None:
    prices = frame(
        A=[100 + 4.0 * step for step in range(10)],
        B=[100 + 1.0 * step for step in range(10)],
        C=[100 - 2.0 * step for step in range(10)],
    )
    last = signals.momentum_selection(prices, 5, 2).iloc[-1]
    assert last.to_dict() == {"A": 0.5, "B": 0.5, "C": 0.0}
    assert signals.momentum_selection(prices, 5, 1).iloc[-1].to_dict() == {
        "A": 1.0,
        "B": 0.0,
        "C": 0.0,
    }


def test_momentum_goes_to_cash_when_nothing_gained() -> None:
    falling = frame(A=[100 - step for step in range(10)], B=[50 - step for step in range(10)])
    assert signals.momentum_selection(falling, 5, 2).iloc[-1].sum() == 0.0


def test_momentum_with_one_winner_leaves_the_other_slot_in_cash() -> None:
    mixed = frame(A=[100 + step for step in range(10)], B=[100 - step for step in range(10)])
    assert signals.momentum_selection(mixed, 5, 2).iloc[-1].to_dict() == {"A": 0.5, "B": 0.0}


def test_weights_are_shares_of_the_strategys_capital() -> None:
    etf = STRATEGIES["halal-etf-trend"]
    up = signals.weights(uptrend(etf.universe), etf)
    assert list(up) == etf.universe
    assert up.iloc[-1].tolist() == [0.5, 0.5]
    assert signals.weights(downtrend(etf.universe), etf).iloc[-1].sum() == 0.0


def test_the_volatility_target_only_ever_scales_exposure_down() -> None:
    spec = STRATEGIES["crypto-donchian"]
    prices = uptrend(spec.universe, 600)
    scaled = signals.weights(prices, spec)
    unscaled = signals.weights(prices, spec.model_copy(update={"target_vol": None}))
    assert (scaled.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (scaled <= unscaled + 1e-9).all().all()


def test_warmup_is_the_longest_lookback_plus_the_volatility_window() -> None:
    assert signals.warmup_days(STRATEGIES["btc-trend-200"]) == 200 + signals.VOLATILITY_WINDOW
    assert signals.warmup_days(STRATEGIES["crypto-donchian"]) == 360 + signals.VOLATILITY_WINDOW
    assert signals.warmup_days(STRATEGIES["halal-etf-rotation"]) == 126 + signals.VOLATILITY_WINDOW
    assert signals.warmup_days(STRATEGIES["halal-multi-trend"]) == 250 + 60


def test_the_trend_votes_are_the_share_of_windows_the_close_is_above() -> None:
    rising = series([100.0] * 30 + [101.0 + step for step in range(40)])
    votes = signals.trend_votes(rising, [5, 10, 400])
    assert votes.iloc[-1] == pytest.approx(2 / 3)
    assert votes.iloc[:5].tolist() == [0.0] * 5


def test_the_band_holds_a_weight_until_the_aim_moves_far_enough_away() -> None:
    aim = frame(A=[0.0, 0.03, 0.04, 0.06, 0.07, 0.30, 0.31, 0.0])
    held = signals.no_trade_band(aim, 0.05)
    assert held["A"].tolist() == [0.0, 0.0, 0.0, 0.06, 0.06, 0.30, 0.30, 0.0]


def test_the_band_moves_each_asset_on_its_own() -> None:
    aim = frame(A=[0.0, 0.4, 0.4], B=[0.0, 0.01, 0.02])
    held = signals.no_trade_band(aim, 0.05)
    assert held["A"].tolist() == [0.0, 0.4, 0.4] and held["B"].tolist() == [0.0, 0.0, 0.0]


def test_the_multi_speed_rule_is_sized_by_volatility_and_never_above_one() -> None:
    spec = STRATEGIES["halal-multi-trend"]
    prices = uptrend(spec.universe, 700)
    sized = signals.weights(prices, spec)
    plain = spec.model_copy(update={"target_vol": None, "trade_rate": 1.0})
    unscaled = signals.weights(prices, plain)
    assert (sized.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (sized.iloc[-1] <= unscaled.iloc[-1] + 1e-9).all()
    assert signals.weights(downtrend(spec.universe, 700), spec).iloc[-1].sum() < 1e-3


def test_the_band_version_trades_only_when_a_weight_is_far_enough_off() -> None:
    spec = STRATEGIES["halal-multi-trend"].model_copy(update={"trade_rate": 1.0, "band": 0.05})
    sized = signals.weights(uptrend(spec.universe, 700), spec)
    changes = sized.diff().abs().stack().replace(0.0, np.nan).dropna()
    assert (changes > spec.band - 1e-9).all()


def test_partial_adjustment_moves_a_fixed_share_of_the_way_to_the_aim_each_day() -> None:
    aim = frame(A=[0.0, 1.0, 1.0, 1.0, 0.0])
    held = signals.partial_adjustment(aim, 0.25)
    assert held["A"].tolist() == pytest.approx([0.0, 0.25, 0.4375, 0.578125, 0.43359375])


def test_a_trade_rate_of_one_goes_straight_to_the_aim_and_a_lower_one_trades_less() -> None:
    spec = STRATEGIES["halal-multi-trend"]
    prices = uptrend(spec.universe, 700)
    prices.iloc[400:500] *= np.linspace(1.0, 0.7, 100)[:, None]
    fast = signals.weights(prices, spec.model_copy(update={"trade_rate": 1.0}))
    slow = signals.weights(prices, spec)
    assert fast.diff().abs().sum().sum() > slow.diff().abs().sum().sum()


def test_a_longer_volatility_window_reacts_more_slowly() -> None:
    wild = frame(A=[100 * (1.05 if step % 2 else 0.95) for step in range(40)] + [100.0] * 40)
    fast = signals.volatility_scale(wild, 0.12, 252, 10)
    slow = signals.volatility_scale(wild, 0.12, 252, 60)
    assert fast.iloc[-1] > slow.iloc[-1]
