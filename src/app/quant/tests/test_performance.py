import pandas as pd
import pytest

from app.quant import performance
from app.quant.performance import Metrics, Sensitivity


def metric(**overrides: float) -> Metrics:
    base = {
        "total_return": 0.5,
        "cagr": 0.1,
        "volatility": 0.2,
        "sharpe": 1.0,
        "max_drawdown": -0.2,
        "exposure": 1.0,
        "turnover": 0.0,
    }
    return Metrics(**{**base, **overrides})


def test_each_trade_is_charged_once_on_the_day_it_happens() -> None:
    index = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"X": [100.0, 100.0, 100.0]}, index=index)
    held = pd.DataFrame({"X": [0.0, 1.0, 1.0]}, index=index)
    returns = performance.strategy_returns(prices, held, fee_bps=100)
    assert returns.tolist() == pytest.approx([0.0, -0.01, 0.0])


def test_filled_at_the_open_a_new_position_misses_the_overnight_gap() -> None:
    index = pd.date_range("2024-01-01", periods=3)
    closes = pd.DataFrame({"X": [100.0, 110.0, 121.0]}, index=index)
    opens = pd.DataFrame({"X": [100.0, 108.0, 115.0]}, index=index)
    held = pd.DataFrame({"X": [0.0, 1.0, 1.0]}, index=index)
    at_open = performance.strategy_returns(closes, held, 0, opens)
    at_close = performance.strategy_returns(closes, held, 0)
    assert at_open.tolist() == pytest.approx([0.0, 110 / 108 - 1, 0.1])
    assert at_close.tolist() == pytest.approx([0.0, 0.1, 0.1])


def test_filled_at_the_open_an_exit_still_carries_the_overnight_gap() -> None:
    index = pd.date_range("2024-01-01", periods=3)
    closes = pd.DataFrame({"X": [100.0, 100.0, 90.0]}, index=index)
    opens = pd.DataFrame({"X": [100.0, 100.0, 95.0]}, index=index)
    held = pd.DataFrame({"X": [1.0, 1.0, 0.0]}, index=index)
    returns = performance.strategy_returns(closes, held, 0, opens)
    assert returns.iloc[-1] == pytest.approx(-0.05)


def test_without_gaps_the_open_fill_matches_the_close_fill() -> None:
    index = pd.date_range("2024-01-01", periods=4)
    closes = pd.DataFrame({"X": [100.0, 103.0, 99.0, 104.0]}, index=index)
    held = pd.DataFrame({"X": [0.0, 1.0, 0.5, 1.0]}, index=index)
    flat_open = closes.shift(1).fillna(100.0)
    assert performance.strategy_returns(closes, held, 10, flat_open).tolist() == pytest.approx(
        performance.strategy_returns(closes, held, 10).tolist()
    )


def test_all_checks_pass_for_a_strategy_that_beats_buy_and_hold_everywhere() -> None:
    strategy = metric(sharpe=1.5, max_drawdown=-0.1)
    result = performance.checks(
        strategy,
        metric(),
        metric(cagr=0.1),
        [Sensitivity(fee_bps=50, cagr=0.05, sharpe=1)],
        0.97,
        4,
    )
    assert [c.passed for c in result] == [True, True, True, True, True]


def test_each_check_fails_for_its_own_reason() -> None:
    weak = metric(sharpe=0.5, max_drawdown=-0.4)
    result = performance.checks(
        weak, metric(), metric(cagr=-0.02), [Sensitivity(fee_bps=50, cagr=-0.01, sharpe=0)], 0.6, 4
    )
    assert [c.passed for c in result] == [False, False, False, False, False]
    assert len({c.name for c in result}) == 5


def test_the_number_of_trials_is_named_in_the_last_check() -> None:
    result = performance.checks(
        metric(), metric(), metric(), [Sensitivity(fee_bps=50, cagr=0.05, sharpe=1)], 0.97, 8
    )
    assert result[-1].name.endswith("(8 trials)")


def test_metrics_report_exposure_and_yearly_turnover() -> None:
    index = pd.date_range("2024-01-01", periods=4)
    held = pd.DataFrame({"X": [0.0, 1.0, 1.0, 0.0]}, index=index)
    returns = pd.Series([0.0, 0.01, 0.01, 0.0], index=index)
    result = performance.metrics(returns, held, 252)
    assert result.exposure == pytest.approx(0.5)
    assert result.turnover == pytest.approx(2 / 4 * 252)
    assert result.total_return == pytest.approx(1.01 * 1.01 - 1)


def test_with_a_p_value_the_sharpe_lead_must_be_unlikely_to_be_luck() -> None:
    args = (metric(sharpe=1.5), metric(), metric(), [Sensitivity(fee_bps=50, cagr=0.05, sharpe=1)])
    lucky = performance.checks(*args, 0.97, 4, sharpe_p=0.3)
    sure = performance.checks(*args, 0.97, 4, sharpe_p=0.02, evidence="the market since 1927")
    assert not lucky[0].passed and lucky[0].name == "Sharpe above buy and hold, p under 10%"
    assert sure[0].passed
    assert sure[0].name.endswith(", on the market since 1927")
    assert sure[-1].name == "Deflated Sharpe above 95% (4 trials, on the market since 1927)"
