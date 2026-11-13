import pandas as pd
import pytest

from app.features.systematic import backtest
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.tests.support import closes, downtrend, uptrend
from app.quant.history import NotEnoughHistoryError

BTC = STRATEGIES["btc-trend-200"]
ETFS = STRATEGIES["halal-etf-trend"]
ROTATION = STRATEGIES["halal-etf-rotation"]


def test_a_steady_uptrend_is_held_almost_fully_and_matches_buy_and_hold() -> None:
    result = backtest.run(uptrend(BTC.universe, 500), BTC, fee_bps=0)
    assert result.metrics.exposure > 0.95
    assert result.metrics.total_return == pytest.approx(result.benchmark.total_return, rel=0.05)
    assert result.signals == {"BTC/USD": 1.0}


def test_a_steady_downtrend_stays_in_cash() -> None:
    result = backtest.run(downtrend(BTC.universe, 500), BTC)
    assert result.metrics.exposure == 0.0
    assert result.metrics.total_return == 0.0
    assert result.benchmark.total_return < -0.5
    assert result.signals == {"BTC/USD": 0.0}


def test_fees_reduce_returns_and_the_sensitivity_table_is_ordered_by_fee() -> None:
    frame = closes(BTC.universe, 0.0005, 700, seed=3)
    free, costly = backtest.run(frame, BTC, fee_bps=0), backtest.run(frame, BTC, fee_bps=100)
    assert costly.metrics.total_return < free.metrics.total_return
    assert [s.fee_bps for s in free.sensitivity] == [10, 25, 50]
    assert free.sensitivity[0].cagr > free.sensitivity[-1].cagr


def test_the_default_fee_comes_from_the_strategy() -> None:
    assert backtest.run(uptrend(BTC.universe, 500), BTC).fee_bps == BTC.fee_bps
    assert backtest.run(uptrend(BTC.universe, 500), BTC, fee_bps=7).fee_bps == 7


def test_the_window_starts_after_the_warmup_and_growth_starts_at_ten_thousand() -> None:
    frame = uptrend(BTC.universe, 500)
    result = backtest.run(frame, BTC)
    assert result.days == 500 - 220
    assert result.start == frame.index[220]
    assert result.growth[0].strategy == pytest.approx(10_000, rel=0.02)
    assert result.growth[-1].time == frame.index[-1]


def test_long_histories_are_sampled_down_but_keep_the_last_day() -> None:
    frame = uptrend(BTC.universe, 2_000)
    result = backtest.run(frame, BTC)
    assert len(result.growth) <= backtest.GROWTH_POINTS + 1
    assert result.growth[-1].time == frame.index[-1]
    assert result.days == 1_780


def test_sleeves_split_capital_equally_across_the_universe() -> None:
    result = backtest.run(uptrend(ETFS.universe, 500), ETFS)
    assert result.metrics.exposure == pytest.approx(1.0, abs=0.05)
    assert set(result.signals) == set(ETFS.universe)


def test_dual_momentum_rotates_into_what_is_rising() -> None:
    frame = closes(ROTATION.universe, 0.0, 400, seed=1)
    frame["SPUS"] = 100 * 1.004 ** pd.Series(range(400), index=frame.index)
    result = backtest.run(frame, ROTATION)
    assert result.signals["SPUS"] == 0.5
    assert result.metrics.exposure > 0.3


def test_in_and_out_of_sample_halves_cover_the_run() -> None:
    result = backtest.run(uptrend(BTC.universe, 600), BTC)
    assert result.in_sample.total_return > 0
    assert result.out_of_sample.total_return > 0


def test_too_little_history_is_explained() -> None:
    with pytest.raises(NotEnoughHistoryError, match="Need 280 daily bars"):
        backtest.run(uptrend(BTC.universe, 250), BTC)


def test_daily_returns_are_the_returns_the_backtest_measures() -> None:
    frame = closes(BTC.universe, 0.0008, 700, seed=5)
    result = backtest.run(frame, BTC)
    daily = backtest.daily_returns(frame, BTC)
    assert len(daily) == result.days and daily.index[0] == result.start
    assert (1 + daily).prod() - 1 == pytest.approx(result.metrics.total_return)


def test_daily_returns_take_a_fee_for_the_default_or_the_one_given() -> None:
    frame = closes(BTC.universe, 0.0008, 700, seed=5)
    free, costly = backtest.daily_returns(frame, BTC, 0), backtest.daily_returns(frame, BTC, 100)
    default = backtest.daily_returns(frame, BTC)
    assert costly.sum() < default.sum() < free.sum()


def gapped(closes: pd.DataFrame, gap: float) -> pd.DataFrame:
    return closes.shift(1).fillna(closes) * (1 + gap)


def test_a_stock_rule_fills_at_the_next_open_when_opens_are_given() -> None:
    closes = uptrend(ETFS.universe, 500)
    at_close = backtest.run(closes, ETFS)
    at_open = backtest.run(closes, ETFS, opens=gapped(closes, 0.01))
    assert ETFS.fills == "next_open"
    assert at_open.metrics.total_return != pytest.approx(at_close.metrics.total_return)


def test_opens_with_no_gap_give_the_close_fill_returns() -> None:
    closes = uptrend(ETFS.universe, 500)
    flat = backtest.daily_returns(closes, ETFS, opens=gapped(closes, 0.0))
    plain = backtest.daily_returns(closes, ETFS)
    assert flat.iloc[1:].tolist() == pytest.approx(plain.iloc[1:].tolist())


def test_a_coin_rule_fills_at_the_close_even_when_opens_are_given() -> None:
    closes = uptrend(BTC.universe, 500)
    assert BTC.fills == "close"
    with_opens = backtest.run(closes, BTC, opens=gapped(closes, 0.05))
    assert with_opens.metrics == backtest.run(closes, BTC).metrics


def test_daily_returns_fill_at_the_open_like_the_backtest() -> None:
    closes = uptrend(ETFS.universe, 500)
    opens = gapped(closes, 0.01)
    result = backtest.run(closes, ETFS, fee_bps=0, opens=opens)
    daily = backtest.daily_returns(closes, ETFS, 0, opens)
    assert (1 + daily).prod() - 1 == pytest.approx(result.metrics.total_return)


def century(drift: float = 0.0004, days: int = 6_000) -> pd.Series:
    import numpy as np

    rng = np.random.default_rng(11)
    calm = rng.normal(drift, 0.008, days)
    crashes = np.where(np.arange(days) % 1500 > 1350, -0.004, 0.0)
    index = pd.bdate_range("1990-01-01", periods=days)
    return pd.Series(calm + crashes, index=index)


def test_a_proxy_rule_is_tested_for_significance_on_the_long_history() -> None:
    closes = uptrend(ETFS.universe, 500)
    result = backtest.run(closes, ETFS, market=century())
    assert ETFS.proxy
    assert result.evidence is not None
    assert result.evidence.days > result.days
    assert result.evidence.name == "US market (Fama-French)"
    first, *_, last = result.checks
    assert first.name.endswith("on US market (Fama-French) since 1990")
    assert "on US market (Fama-French) since 1990" in last.name
    assert first.passed == (result.evidence.sharpe_p < 0.10)
    assert last.passed == (result.evidence.deflated_sharpe >= 0.95)


def test_without_the_proxy_file_the_checks_stay_on_the_rules_own_history() -> None:
    result = backtest.run(uptrend(ETFS.universe, 500), ETFS)
    assert result.evidence is None
    assert result.sharpe_p is not None
    assert "since" not in result.checks[0].name


def test_a_rule_without_a_proxy_ignores_one() -> None:
    result = backtest.run(uptrend(BTC.universe, 500), BTC, market=century())
    assert result.evidence is None


def test_the_multi_speed_rule_runs_on_the_long_history_and_keeps_its_exposure_inside_one() -> None:
    spec = STRATEGIES["halal-multi-trend"]
    result = backtest.run(uptrend(spec.universe, 600), spec, market=century())
    assert result.evidence is not None and result.evidence.days > result.days
    assert result.evidence.metrics.exposure <= 1.0
    assert result.metrics.exposure <= 1.0
    assert all(0.0 <= weight <= 0.5 for weight in result.signals.values())
