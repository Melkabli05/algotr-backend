import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.quant import validation


def daily(mean: float, std: float, days: int = 1500, seed: int = 1) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mean, std, days))


def test_a_clearly_positive_edge_is_almost_surely_real() -> None:
    assert validation.probabilistic_sharpe(daily(0.002, 0.01)) > 0.99


def test_a_zero_edge_is_a_coin_flip() -> None:
    noise = daily(0.0, 0.01)
    assert validation.probabilistic_sharpe(noise - noise.mean()) == pytest.approx(0.5)


def test_a_negative_edge_is_almost_surely_not_real() -> None:
    assert validation.probabilistic_sharpe(daily(-0.002, 0.01)) < 0.01


def test_more_trials_raise_the_bar_luck_has_to_clear() -> None:
    bars = [validation.expected_max_sharpe(n, 1000) for n in (2, 4, 20, 200)]
    assert bars == sorted(bars)
    assert bars[0] > 0


def test_a_single_trial_needs_no_deflation() -> None:
    assert validation.expected_max_sharpe(1, 1000) == 0.0


@pytest.mark.parametrize(("trials", "observations"), [(10, 250), (100, 500)])
def test_the_luck_bar_matches_the_simulated_best_of_many_strategies_with_no_skill(
    trials: int, observations: int
) -> None:
    luckless = stats.t.rvs(
        df=observations - 1, size=(10_000, trials), random_state=np.random.default_rng(0)
    )
    best = (luckless / np.sqrt(observations)).max(axis=1).mean()
    assert validation.expected_max_sharpe(trials, observations) == pytest.approx(best, rel=0.03)


def test_deflation_lowers_confidence_and_longer_histories_restore_it() -> None:
    returns = daily(0.0006, 0.01)
    raw = validation.probabilistic_sharpe(returns)
    assert validation.deflated_sharpe(returns, 50) < validation.deflated_sharpe(returns, 4) < raw
    longer = daily(0.0006, 0.01, days=6000)
    assert validation.deflated_sharpe(longer, 50) > validation.deflated_sharpe(returns, 50)


def test_fat_left_tails_lower_the_confidence_in_the_same_average_edge() -> None:
    calm = daily(0.001, 0.01)
    crashy = calm.copy()
    crashy.iloc[::100] = -0.08
    crashy += calm.mean() - crashy.mean()
    assert validation.probabilistic_sharpe(crashy) < validation.probabilistic_sharpe(calm)


def candidates(edges: list[float], days: int = 1200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"c{i}": rng.normal(edge, 0.01, days) for i, edge in enumerate(edges)})


def test_a_strong_edge_clears_the_harvey_liu_zhu_hurdle_and_noise_does_not() -> None:
    assert validation.sharpe_t_statistic(daily(0.002, 0.01)) > validation.HURDLE_T
    noise = daily(0.0, 0.01, seed=5)
    assert abs(validation.sharpe_t_statistic(noise - noise.mean())) < 1e-9


def test_picking_the_winner_among_pure_noise_is_overfitting() -> None:
    pbo = validation.probability_of_overfitting(candidates([0.0] * 12))
    assert pbo > 0.35


def test_a_candidate_that_is_really_better_keeps_winning_out_of_sample() -> None:
    pbo = validation.probability_of_overfitting(candidates([0.003] + [0.0] * 11))
    assert pbo < 0.1


def test_overfitting_needs_something_to_compare() -> None:
    with pytest.raises(ValueError, match="at least two"):
        validation.probability_of_overfitting(candidates([0.001]))


def test_a_steady_edge_pays_in_every_slice_and_a_lucky_stretch_does_not() -> None:
    steady = pd.Series(np.full(800, 0.001))
    lucky = pd.Series(np.concatenate([np.full(100, 0.02), np.full(700, -0.0001)]))

    assert validation.consistency(steady) == 1.0
    assert validation.consistency(lucky) == pytest.approx(1 / 8)


def test_reliability_summarises_how_much_runs_of_one_recipe_disagree() -> None:
    steady = validation.reliability([0.9, 1.0, 1.1, 1.0, 0.95])
    erratic = validation.reliability([-0.5, 1.8, 0.1, 2.0, -0.3])

    assert steady["median"] == pytest.approx(1.0)
    assert steady["share_above"] == 1.0
    assert steady["spread"] < 0.2
    assert erratic["share_above"] == pytest.approx(0.6)
    assert erratic["spread"] > 1.5
    assert (erratic["minimum"], erratic["maximum"]) == (-0.5, 2.0)


def test_reliability_counts_only_runs_strictly_above_the_floor() -> None:
    assert validation.reliability([1.0, 1.0, 2.0], floor=1.0)["share_above"] == pytest.approx(1 / 3)


def test_a_series_that_never_moves_has_no_edge_instead_of_an_undefined_one() -> None:
    flat = pd.Series(np.zeros(500))
    assert validation.probabilistic_sharpe(flat) == 0.0
    assert validation.deflated_sharpe(flat, 10) == 0.0
    assert validation.sharpe_t_statistic(flat) == 0.0


def test_a_narrow_measured_spread_never_lowers_the_luck_bar() -> None:
    floor = validation.expected_max_sharpe(10, 1_000)
    assert validation.expected_max_sharpe(10, 1_000, 0.0001) == floor
    assert validation.expected_max_sharpe(10, 1_000, 0.1) > floor


def test_a_wider_spread_of_trials_deflates_more() -> None:
    returns = pd.Series(np.random.default_rng(3).normal(0.001, 0.01, 1_000))
    assert validation.deflated_sharpe(returns, 10, 0.1) < validation.deflated_sharpe(returns, 10)


def test_a_clear_sharpe_lead_is_unlikely_to_be_luck() -> None:
    rng = np.random.default_rng(7)
    benchmark = pd.Series(rng.normal(0.0002, 0.01, 3_000))
    strategy = benchmark * 0.5 + pd.Series(rng.normal(0.0008, 0.004, 3_000))
    assert validation.sharpe_difference_p(strategy, benchmark) < 0.05


def test_the_same_returns_lead_by_nothing() -> None:
    returns = daily(0.0005, 0.01)
    noisy = returns + pd.Series(np.random.default_rng(9).normal(0, 0.001, len(returns)))
    assert validation.sharpe_difference_p(noisy, returns) > 0.2


def test_a_strategy_behind_the_benchmark_never_passes() -> None:
    benchmark = daily(0.0008, 0.01, seed=2)
    assert (
        validation.sharpe_difference_p(benchmark * 0.0 + daily(0.0, 0.01, seed=4), benchmark) > 0.5
    )
