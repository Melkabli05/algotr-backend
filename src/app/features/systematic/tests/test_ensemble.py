import math

import numpy as np
import pandas as pd
import pytest

from app.features.systematic import ensemble


def daily(months: int, **moves: float) -> pd.DataFrame:
    """Agents whose every day returns a fixed amount, for `months` months from January 2024."""
    index = pd.bdate_range("2024-01-01", periods=months * 21)
    return pd.DataFrame(
        {name: np.full(len(index), move) for name, move in moves.items()}, index=index
    )


def test_a_reward_runs_from_zero_to_one_and_is_half_for_a_flat_round() -> None:
    assert ensemble.reward(0.0, 0.03) == 0.5
    low, flat, high = (ensemble.reward(move, 0.03) for move in (-0.05, 0.0, 0.05))
    assert 0 < low < flat < high < 1


def test_the_score_is_the_mean_plus_the_bonus_of_the_ucb1_formula() -> None:
    found = ensemble.scores([0.6, 0.5], [10, 4])
    assert found[0] == pytest.approx(0.6 + math.sqrt(2 * math.log(14) / 10))
    assert found[1] == pytest.approx(0.5 + math.sqrt(2 * math.log(14) / 4))


def test_an_arm_with_fewer_rounds_gets_the_larger_bonus() -> None:
    few, many = ensemble.scores([0.5, 0.5], [3, 30])
    assert few > many


def test_arms_share_equally_until_each_has_a_round() -> None:
    assert ensemble.allocate([[0.9], []]) == [0.5, 0.5]
    assert ensemble.allocate([[], [], []]) == pytest.approx([1 / 3] * 3)


def test_equal_records_give_equal_weights_and_the_weights_sum_to_one() -> None:
    mix = ensemble.allocate([[0.5, 0.6], [0.5, 0.6], [0.5, 0.6]])
    assert mix == pytest.approx([1 / 3] * 3) and sum(mix) == pytest.approx(1)


def test_the_better_record_gets_more_but_nobody_falls_below_the_floor() -> None:
    mix = ensemble.allocate([[0.9] * 20, [0.5] * 20, [0.1] * 20])
    assert mix[0] > mix[1] > mix[2] >= ensemble.FLOOR - 1e-12
    assert sum(mix) == pytest.approx(1)
    assert ensemble.FLOOR <= min(mix) < ensemble.FLOOR + 1e-3


def test_a_flatter_temperature_spreads_the_weight_out() -> None:
    rewards = [[0.8] * 10, [0.2] * 10]
    sharp, soft = (
        ensemble.allocate(rewards, temperature=0.05),
        ensemble.allocate(rewards, temperature=1.0),
    )
    assert sharp[0] > soft[0] > 0.5


def test_the_first_month_is_shared_equally_and_later_months_follow_the_record() -> None:
    result = ensemble.replay(daily(6, good=0.001, bad=-0.001))
    first, last = result.weights.iloc[0], result.weights.iloc[-1]
    assert first.tolist() == pytest.approx([0.5, 0.5])
    assert last["good"] > 0.5 > last["bad"] >= ensemble.FLOOR - 1e-12
    assert result.latest["good"] > result.latest["bad"]


def test_the_blend_earns_the_weighted_average_of_the_agents_each_day() -> None:
    agents = daily(4, good=0.001, bad=-0.001)
    result = ensemble.replay(agents)
    month = agents.index.to_period("M")[-1]
    weights = result.weights.loc[month]
    expected = weights["good"] * 0.001 + weights["bad"] * -0.001
    assert result.returns[agents.index[-1]] == pytest.approx(expected)
    assert len(result.returns) == len(agents)


def test_a_months_weights_never_depend_on_that_month_or_later() -> None:
    agents = daily(6, good=0.001, bad=-0.001)
    changed = agents.copy()
    months = agents.index.to_period("M")
    last = months == months[-1]
    changed.loc[last] = changed.loc[last].to_numpy()[:, ::-1]
    first, second = ensemble.replay(agents), ensemble.replay(changed)
    assert first.weights.iloc[:-1].to_numpy() == pytest.approx(second.weights.iloc[:-1].to_numpy())
    assert first.weights.iloc[-1].tolist() == pytest.approx(second.weights.iloc[-1].tolist())


def test_one_agent_is_all_the_ensemble() -> None:
    result = ensemble.replay(daily(3, only=0.0005))
    assert result.weights["only"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert result.returns.sum() == pytest.approx(0.0005 * 63)


def test_with_no_days_there_is_no_ensemble() -> None:
    result = ensemble.replay(pd.DataFrame({"a": [], "b": []}, index=pd.DatetimeIndex([])))
    assert result.returns.empty and result.latest == {"a": 0.5, "b": 0.5}
