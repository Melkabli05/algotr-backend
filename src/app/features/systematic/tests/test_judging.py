import numpy as np
import pandas as pd
import pytest

from app.features.systematic import judging, signals
from app.features.systematic.catalog import STRATEGIES
from app.quant import performance

SPEC = STRATEGIES["halal-etf-trend"]
LABELS = ["seed 1", "seed 2", "seed 3", "seed 4", "seed 5"]


def market(days: int = 1000, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end="2026-10-30", periods=days)
    steps = 0.0006 + rng.normal(0, 0.009, (days, 2))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=SPEC.universe)


def trend(closes: pd.DataFrame) -> pd.Series:
    held = signals.weights(closes, SPEC).shift(1).fillna(0.0)
    return performance.strategy_returns(closes, held, SPEC.fee_bps)


def candidates(closes: pd.DataFrame, edge: float, noise: float = 0.0005) -> pd.DataFrame:
    """Candidates whose daily return is the trend rule's plus `edge`, with noise per candidate."""
    window = closes.index[550:]
    base = trend(closes).loc[window].to_numpy()
    return pd.DataFrame(
        {
            label: base + edge + np.random.default_rng(n).normal(0, noise, len(window))
            for n, label in enumerate(LABELS)
        },
        index=window,
    )


def trained(value: float = 1.2) -> dict[str, float | None]:
    return dict.fromkeys(LABELS, value)


def test_candidates_that_really_beat_the_trend_rule_in_every_seed_pass_every_gate() -> None:
    closes = market()
    verdict = judging.judge(closes, candidates(closes, 0.0015), trained(), trials=5)

    assert verdict.eligible
    assert all(gate.passed for gate in verdict.gates), [g for g in verdict.gates if not g.passed]
    assert all(row.beats_trend for row in verdict.rows)
    assert verdict.reliability["share_above"] == 1.0


def test_candidates_that_fall_short_of_the_trend_rule_fail_and_say_why() -> None:
    closes = market()
    verdict = judging.judge(closes, candidates(closes, -0.001), trained(), trials=5)

    gates = {g.name: g for g in verdict.gates}
    assert not verdict.eligible
    assert not gates["Stable across seeds"].passed
    assert gates["Stable across seeds"].detail.startswith("0 of 5 seeds beat the trend rule")
    assert not gates["Beats the trend rule after the search"].passed


def test_a_big_gap_between_training_and_unseen_data_fails_the_generalisation_gate() -> None:
    closes = market()
    verdict = judging.judge(closes, candidates(closes, 0.0), trained(25.0), trials=5)
    gate = {g.name: g for g in verdict.gates}["Holds up on data it never saw"]
    assert not gate.passed and "of in-sample" in gate.detail


def test_without_training_figures_that_gate_cannot_pass_by_default() -> None:
    closes = market()
    verdict = judging.judge(closes, candidates(closes, 0.0015), {}, trials=5)
    gate = {g.name: g for g in verdict.gates}["Holds up on data it never saw"]
    assert not gate.passed and "not reported" in gate.detail
    assert not verdict.eligible


def test_every_trial_ever_tried_makes_the_skill_test_harder() -> None:
    closes = market()
    frame = candidates(closes, 0.0004, noise=0.001)
    few = judging.judge(closes, frame, trained(), trials=5)
    many = judging.judge(closes, frame, trained(), trials=500)
    assert many.trials == 500
    assert many.deflated_sharpe <= few.deflated_sharpe


def test_the_verdict_compares_with_the_rule_and_buy_and_hold_on_the_same_dates() -> None:
    closes = market()
    frame = candidates(closes, 0.0)
    verdict = judging.judge(closes, frame, trained(), trials=5)
    assert verdict.start[:10] == frame.index[0].date().isoformat()
    assert verdict.days == len(frame)
    assert verdict.trend_sharpe == pytest.approx(judging.sharpe(trend(closes).loc[frame.index]))


def test_a_candidate_that_never_trades_scores_zero_and_cannot_poison_the_verdict() -> None:
    closes = market()
    frame = candidates(closes, 0.0005)
    frame["seed 1"] = 0.0

    verdict = judging.judge(closes, frame, trained(), trials=5)

    assert judging.sharpe(pd.Series([0.0] * 50)) == 0.0
    assert {row.label: row for row in verdict.rows}["seed 1"].sharpe == 0.0
    numbers = [*verdict.reliability.values(), verdict.deflated_sharpe, verdict.reality_check_p]
    assert all(np.isfinite(value) for value in numbers)
    assert judging.Judgement.model_validate_json(verdict.model_dump_json()) == verdict


def test_a_single_candidate_is_judged_without_an_overfitting_comparison() -> None:
    closes = market()
    verdict = judging.judge(closes, candidates(closes, 0.001)[["seed 1"]], {"seed 1": 1.0}, 1)
    assert verdict.probability_of_overfitting == 0.0
    assert len(verdict.rows) == 1
