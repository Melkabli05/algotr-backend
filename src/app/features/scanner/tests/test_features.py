import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from scipy import stats

from app.features.scanner import bars, features
from app.features.scanner.tests.support import panel_of, random_panel, stock


def test_the_panel_becomes_one_column_per_symbol_and_one_row_per_session() -> None:
    panel = panel_of(stock("AAA", 30), stock("BBB", 20, seed=1))
    close = features.wide(panel, "close")
    assert list(close.columns) == ["AAA", "BBB"]
    assert len(close) == 30 and close["BBB"].isna().sum() == 10


def test_the_regression_score_matches_a_least_squares_fit() -> None:
    panel = random_panel(symbols=4, days=300)
    close = features.wide(panel, "close")
    score = features.clenow(close)
    for symbol in ("S01", "S02"):
        for row in (150, 220, 299):
            window = np.log(close[symbol].iloc[row - 89 : row + 1].to_numpy())
            fit = stats.linregress(np.arange(90), window)
            expected = (np.exp(fit.slope * 252) - 1) * fit.rvalue**2
            assert score[symbol].iloc[row] == pytest.approx(expected, rel=1e-7)


def test_a_flat_price_has_no_regression_score() -> None:
    close = features.wide(stock("FLAT", 120, noise=0.0), "close")
    assert features.clenow(close).iloc[-1].isna().all()


def test_momentum_leaves_out_the_latest_month() -> None:
    close = features.wide(stock("AAA", 300, daily=0.001), "close")
    row = 280
    expected = close["AAA"].iloc[row - 21] / close["AAA"].iloc[row - 252] - 1
    assert features.momentum(close, 252)["AAA"].iloc[row] == pytest.approx(expected)
    assert features.momentum(close, 252)["AAA"].iloc[:252].isna().all()


def test_trend_is_a_close_above_the_200_day_average() -> None:
    up = features.compute(panel_of(stock("UP", 260, daily=0.004, noise=0.001)))
    down = features.compute(panel_of(stock("DOWN", 260, daily=-0.004, noise=0.001)))
    assert bool(up.trend["UP"].iloc[-1]) is True
    assert bool(down.trend["DOWN"].iloc[-1]) is False
    assert not up.trend["UP"].iloc[:199].any()


def test_dollar_volume_gap_volatility_and_history() -> None:
    quiet = stock("QUIET", 120, noise=0.0, price=50.0, volume=2_000_000.0)
    jumpy = stock("JUMPY", 120, noise=0.0, price=50.0)
    jumpy.loc[80:, ["open", "high", "low", "close"]] *= 1.25
    found = features.compute(panel_of(quiet, jumpy))
    assert found.adv20["QUIET"].iloc[-1] == pytest.approx(100_000_000.0)
    assert found.gap90["QUIET"].iloc[-1] == pytest.approx(0.0)
    assert found.gap90["JUMPY"].iloc[-1] == pytest.approx(0.25)
    assert found.vol60["JUMPY"].iloc[-1] > found.vol60["QUIET"].iloc[-1]
    assert found.history["QUIET"].iloc[-1] == 120


def test_features_never_look_ahead() -> None:
    panel = random_panel(symbols=12, days=420)
    full = features.compute(panel)
    sessions = full.close.index
    for cut in (260, 300, 350, 419):
        until = panel[panel["date"] <= sessions[cut]]
        part = features.compute(until)
        for name in features.Features.__dataclass_fields__:
            truncated = getattr(part, name).iloc[-1]
            whole = getattr(full, name).loc[sessions[cut]]
            pdt.assert_series_equal(
                truncated.astype(float),
                whole.reindex(truncated.index).astype(float),
                check_names=False,
                rtol=1e-9,
            )


def test_an_empty_panel_gives_empty_features() -> None:
    found = features.compute(bars.empty())
    assert found.close.empty and found.mom_12_1.empty


def test_a_steady_idiosyncratic_edge_raises_the_residual_score() -> None:
    close = features.wide(random_panel(symbols=30, days=700), "close")
    boosted = close.copy()
    boosted["S01"] = close["S01"] * np.exp(0.0015 * np.arange(len(close)))
    assert (
        features.residual_momentum(boosted)["S01"].iloc[-1]
        > features.residual_momentum(close)["S01"].iloc[-1]
    )


def test_a_high_beta_stock_in_a_rising_market_tops_plain_momentum_but_not_residual_momentum() -> (
    None
):
    index = pd.bdate_range("2024-01-01", periods=700)
    rng = np.random.default_rng(3)
    market = rng.normal(0.002, 0.01, len(index))
    close = pd.DataFrame(
        {
            f"S{i:02d}": 100
            * np.cumprod(1 + (2.0 if i == 0 else 1.0) * market + rng.normal(0, 0.01, len(index)))
            for i in range(30)
        },
        index=index,
    )
    plain = features.momentum(close, 252).iloc[-1].rank(ascending=False)
    residual = features.residual_momentum(close).iloc[-1].rank(ascending=False)
    assert plain["S00"] == 1 and residual["S00"] > 3


def test_residual_momentum_never_uses_a_later_day() -> None:
    panel = random_panel(symbols=20, days=700)
    close = features.wide(panel, "close")
    altered = close.copy()
    altered.iloc[500:] *= 0.4
    pdt.assert_frame_equal(
        features.residual_momentum(close).iloc[:500], features.residual_momentum(altered).iloc[:500]
    )


def test_residual_momentum_waits_for_a_beta_and_a_full_window() -> None:
    close = features.wide(random_panel(symbols=10, days=700), "close")
    score = features.residual_momentum(close)
    first = score["S01"].first_valid_index()
    assert first is not None
    assert close.index.get_loc(first) >= 126 + 231 - 1
