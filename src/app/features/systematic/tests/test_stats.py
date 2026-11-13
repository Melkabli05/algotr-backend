import pandas as pd
import pytest

from app.features.systematic import stats


def test_each_invested_stretch_is_a_trade_with_its_result_and_length() -> None:
    index = pd.bdate_range("2026-01-01", periods=8)
    returns = pd.Series([0.0, 0.10, 0.10, 0.0, -0.05, 0.0, 0.02, 0.0], index=index)
    invested = pd.Series([False, True, True, False, True, False, True, True], index=index)
    found = stats.trade_stats(returns, invested, 252)
    assert found.trades == 3
    assert found.win_rate == pytest.approx(2 / 3)
    assert found.profit_factor == pytest.approx((0.21 + 0.02) / 0.05)
    assert found.average_days == pytest.approx((2 + 1 + 2) / 3)


def test_no_trades_and_no_losses_report_nothing_rather_than_a_number() -> None:
    index = pd.bdate_range("2026-01-01", periods=3)
    flat = stats.trade_stats(pd.Series(0.0, index=index), pd.Series(False, index=index), 252)
    assert (flat.trades, flat.win_rate, flat.profit_factor, flat.average_days) == (
        0,
        None,
        None,
        None,
    )
    winner = stats.trade_stats(
        pd.Series([0.0, 0.1, 0.0], index=index), pd.Series([False, True, False], index=index), 252
    )
    assert winner.profit_factor is None and winner.win_rate == 1.0
