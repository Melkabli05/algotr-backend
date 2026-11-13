import numpy as np
import pandas as pd
import pytest

from app.features.systematic import regimes


def market(seed: int = 2) -> pd.Series:
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.001, 0.005, 300)
    storm = rng.normal(-0.002, 0.02, 200)
    flat = rng.normal(0.0, 0.008, 300)
    return pd.Series(
        np.concatenate([calm, storm, flat]), index=pd.bdate_range("2020-01-01", periods=800)
    )


def agents_on(index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "good": rng.normal(0.003, 0.004, len(index)),
            "fair": rng.normal(0.0015, 0.004, len(index)),
            "poor": rng.normal(-0.002, 0.004, len(index)),
        },
        index=index,
    )


def test_the_agents_competent_in_the_regime_are_chosen_by_their_competence() -> None:
    index = pd.bdate_range("2024-01-01", periods=120)
    choice = regimes.select(agents_on(index), pd.Series("volatile", index=index), "volatile", 0.9)
    assert choice.weights["poor"] == 0.0
    assert choice.weights["good"] > choice.weights["fair"] > 0
    assert sum(choice.weights.values()) == pytest.approx(1.0)
    assert "good Sharpe" in choice.reason


def test_an_unclear_regime_gets_equal_and_smaller_weights() -> None:
    index = pd.bdate_range("2024-01-01", periods=120)
    choice = regimes.select(agents_on(index), pd.Series("volatile", index=index), "volatile", 0.45)
    assert list(choice.weights.values()) == pytest.approx([0.15] * 3)
    assert "unclear" in choice.reason


def test_nobody_competent_in_the_regime_holds_cash_and_too_few_days_share_equally() -> None:
    index = pd.bdate_range("2024-01-01", periods=120)
    losing = agents_on(index).assign(good=-0.001, fair=-0.002)
    labels = pd.Series("ranging", index=index)
    assert set(regimes.select(losing, labels, "ranging", 0.9).weights.values()) == {0.0}
    few = pd.Series("trending", index=index).iloc[-5:]
    equal = regimes.select(agents_on(index), few, "trending", 0.9).weights
    assert list(equal.values()) == pytest.approx([1 / 3] * 3)


def test_a_month_is_chosen_from_the_months_before_it_only() -> None:
    history = market()
    agents = agents_on(history.index[-110:])
    first = regimes.replay(agents, history)
    changed = history.copy()
    changed.iloc[-30:] = changed.iloc[-30:] * -3
    second = regimes.replay(agents, changed)
    early = first.weights.index[: len(first.weights) - 2]
    pd.testing.assert_frame_equal(first.weights.loc[early], second.weights.loc[early])
    assert all(0.0 <= value <= 1.0 for value in first.latest.values())
    assert regimes.choose(agents, history, history.index[100]).reason == "Too little history"
