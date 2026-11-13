import numpy as np
import pandas as pd
import pytest

from app.features.scanner import features, rank
from app.features.scanner.tests.support import panel_of, stock

DAYS = 420


def market() -> features.Features:
    thin = stock("THIN", DAYS, daily=0.002, noise=0.002, volume=10_000.0, seed=4)
    jumpy = stock("GAP", DAYS, daily=0.002, noise=0.002, seed=5)
    jumpy.loc[DAYS - 40 :, ["open", "high", "low", "close"]] *= 1.3
    return features.compute(
        panel_of(
            stock("UP1", DAYS, daily=0.0025, noise=0.002, seed=1),
            stock("UP2", DAYS, daily=0.0020, noise=0.002, seed=2),
            stock("UP3", DAYS, daily=0.0015, noise=0.002, seed=3),
            stock("DOWN", DAYS, daily=-0.002, noise=0.002, seed=6),
            thin,
            jumpy,
            stock("NEW", 200, daily=0.003, noise=0.002, seed=7),
        )
    )


def test_there_are_ten_variants_each_counted_once() -> None:
    assert len(rank.VARIANTS) == 10
    assert len({variant.label for variant in rank.VARIANTS}) == 10
    assert {variant.top_n for variant in rank.VARIANTS} == {10, 20}


def test_only_seasoned_liquid_trending_names_without_a_wild_gap_are_eligible() -> None:
    found = market()
    ok = rank.eligible(found)
    assert list(ok.iloc[-1][ok.iloc[-1]].index) == ["UP1", "UP2", "UP3"]
    assert not ok["NEW"].any()


def test_a_name_must_have_a_full_year_of_history() -> None:
    ok = rank.eligible(market())
    first = ok["UP1"].idxmax()
    assert market().history["UP1"].loc[first] >= rank.MIN_HISTORY


def test_the_rank_average_is_taken_among_eligible_names_only() -> None:
    found = market()
    ok = rank.eligible(found)
    combined = rank.scores(found, ok)
    last = combined["rank_avg"].iloc[-1]
    assert last[["DOWN", "THIN", "GAP", "NEW"]].isna().all()
    assert last["UP1"] > last["UP2"] > last["UP3"]
    assert last["UP1"] == pytest.approx(1.0)
    assert set(combined) == set(rank.SCORES)


def test_the_best_scoring_eligible_names_are_chosen_and_nothing_else() -> None:
    found = market()
    ok = rank.eligible(found)
    score = found.mom_12_1.copy()
    score["DOWN"] = 99.0
    assert list(rank.select(score, ok, 2).iloc[-1].loc[lambda row: row].index) == ["UP1", "UP2"]
    everything = rank.select(score, ok, 20).iloc[-1]
    assert list(everything[everything].index) == ["UP1", "UP2", "UP3"]


def frame(**columns: list[float]) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_weights_follow_inverse_volatility() -> None:
    chosen = frame(A=[True], B=[True])
    volatility = frame(A=[0.01], B=[0.02])
    result = rank.weights(chosen, volatility, cap=1.0)
    assert result.iloc[0].tolist() == pytest.approx([2 / 3, 1 / 3])


def test_a_capped_name_hands_its_excess_to_the_others() -> None:
    chosen = frame(A=[True], B=[True], C=[True], D=[True])
    volatility = frame(A=[0.01], B=[0.02], C=[0.02], D=[0.02])
    result = rank.weights(chosen, volatility, cap=0.3).iloc[0]
    assert result["A"] == pytest.approx(0.3)
    assert result[["B", "C", "D"]].tolist() == pytest.approx([0.7 / 3] * 3)
    assert result.sum() == pytest.approx(1.0)


def test_ten_names_under_a_ten_percent_cap_are_equally_weighted() -> None:
    rng = np.random.default_rng(0)
    names = [f"S{i}" for i in range(10)]
    chosen = pd.DataFrame([[True] * 10], columns=names)
    volatility = pd.DataFrame([rng.uniform(0.01, 0.04, 10)], columns=names)
    assert rank.weights(chosen, volatility).iloc[0].tolist() == pytest.approx([0.1] * 10)


def test_what_the_cap_cannot_place_stays_in_cash() -> None:
    chosen = frame(A=[True], B=[True], C=[True], D=[True])
    volatility = frame(A=[0.01], B=[0.02], C=[0.03], D=[0.04])
    result = rank.weights(chosen, volatility, cap=0.2).iloc[0]
    assert result.tolist() == pytest.approx([0.2] * 4)
    assert result.sum() < 1


def test_the_last_session_of_each_month_and_week_is_found() -> None:
    index = pd.bdate_range("2026-01-26", "2026-02-10")
    month = rank.last_of(index, "M")
    week = rank.last_of(index, "W")
    assert list(index[month]) == [pd.Timestamp("2026-01-30"), pd.Timestamp("2026-02-10")]
    assert list(index[week]) == [
        pd.Timestamp("2026-01-30"),
        pd.Timestamp("2026-02-06"),
        pd.Timestamp("2026-02-10"),
    ]


def test_holdings_change_at_month_end_and_a_failing_name_is_sold_at_the_next_weekly_check() -> None:
    index = pd.bdate_range("2026-01-01", "2026-04-30")
    picks = pd.DataFrame(0.5, index=index, columns=["A", "B"])
    picks.loc["2026-01-30"] = [1.0, 0.0]
    picks.loc["2026-02-27"] = [0.0, 1.0]
    ok = pd.DataFrame(True, index=index, columns=["A", "B"])
    ok.loc["2026-02-10":, "A"] = False
    held = rank.targets(picks, ok)
    assert held.loc[:"2026-01-29"].to_numpy().sum() == 0
    assert held.loc["2026-01-30", "A"] == 1.0
    assert held.loc["2026-02-02":"2026-02-12", "A"].eq(1.0).all()
    assert held.loc["2026-02-13":"2026-02-26", "A"].eq(0.0).all()
    assert held.loc["2026-02-13":"2026-02-26", "B"].eq(0.0).all()
    assert held.loc["2026-02-27", "B"] == 1.0
    assert held.loc["2026-03-02":"2026-03-30", "B"].eq(1.0).all()
