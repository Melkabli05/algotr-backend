import pandas as pd
import pytest

from app.features.scanner import features, rank, scan
from app.features.scanner.tests.support import panel_of, stock

DAYS = 520
NAMES = {"UP1": "Up One Inc", "UP2": "Up Two Inc"}
MEMBERS = {"UP1", "UP2", "DOWN", "THIN"}
LISTS = {"SPUS": {"UP1", "UP2", "DOWN"}, "HLAL": {"UP1", "THIN"}}


def panel() -> pd.DataFrame:
    gap = stock("GAP", DAYS, daily=0.002, noise=0.002, seed=5)
    gap.loc[DAYS - 40 :, ["open", "high", "low", "close"]] *= 1.3
    return panel_of(
        stock("UP1", DAYS, daily=0.0025, noise=0.002, seed=1),
        stock("UP2", DAYS, daily=0.0020, noise=0.002, seed=2),
        stock("UP3", DAYS, daily=0.0015, noise=0.002, seed=3),
        stock("DOWN", DAYS, daily=-0.002, noise=0.002, price=80.0, seed=6),
        stock("THIN", DAYS, daily=0.002, noise=0.002, volume=10_000.0, seed=4),
        gap,
        stock("NEW", 200, daily=0.003, noise=0.002, seed=7),
    )


@pytest.fixture(scope="module")
def analysis() -> scan.Analysis:
    return scan.analyse(panel(), MEMBERS)


def test_the_funnel_counts_each_stage_on_the_last_session(analysis: scan.Analysis) -> None:
    result = scan.scan(analysis, NAMES, LISTS, [])
    funnel = result.funnel
    assert funnel.liquid == 6
    assert funnel.listed == 3
    assert funnel.trending == 5
    assert funnel.eligible == 3
    assert funnel.listed_eligible == 2
    assert funnel.picked == 2
    assert result.benchmark == ["DOWN", "UP1", "UP2"]


def test_candidates_are_the_qualifying_names_of_the_whole_market_best_first(
    analysis: scan.Analysis,
) -> None:
    result = scan.scan(analysis, NAMES, LISTS, [])
    assert [c.symbol for c in result.candidates] == ["UP1", "UP2", "UP3"]
    assert [c.rank for c in result.candidates] == [1, 2, 3]
    assert [c.picked for c in result.candidates] == [True, True, False]
    assert [c.weight is not None for c in result.candidates] == [True, True, False]
    assert result.candidates[0].weight == result.picks[scan.PAGE.label]["UP1"]
    first = result.candidates[0]
    assert (first.name, first.lists) == ("Up One Inc", ["HLAL", "SPUS"])
    assert result.candidates[2].lists == [] and result.candidates[2].name is None
    assert first.price > 0 and first.dollar_volume > 20_000_000


def test_every_variant_picks_only_listed_names_within_the_weight_cap(
    analysis: scan.Analysis,
) -> None:
    result = scan.scan(analysis, NAMES, LISTS, [])
    assert set(result.picks) == {variant.label for variant in rank.VARIANTS}
    for weights in result.picks.values():
        assert set(weights) == {"UP1", "UP2"}
        assert all(0 < weight <= rank.CAP for weight in weights.values())


def test_the_stale_lists_are_passed_along_and_the_scan_is_plain_data(
    analysis: scan.Analysis,
) -> None:
    result = scan.scan(analysis, NAMES, LISTS, ["HLAL"])
    assert result.stale_lists == ["HLAL"]
    assert "NaN" not in result.model_dump_json()


def test_a_name_that_is_picked_says_why() -> None:
    explanation = scan.explain(scan.analyse(panel(), MEMBERS), "UP1", NAMES, LISTS)
    assert explanation.verdict == "picked"
    assert all(reason.passed for reason in explanation.reasons)
    assert explanation.reasons[-1].detail == "rank 1 of 2"
    assert explanation.reasons[-2].detail == "HLAL, SPUS"


@pytest.mark.parametrize(
    ("symbol", "verdict", "failed"),
    [
        ("UP3", "scan only", "On a Shariah list"),
        ("DOWN", "excluded", "Closed above its 200-day average"),
        ("THIN", "excluded", "Dollar volume at least $20M a day"),
        ("GAP", "excluded", "No daily move above 15% in 90 sessions"),
        ("NEW", "excluded", "At least 253 sessions of history"),
    ],
)
def test_a_name_that_is_left_out_names_the_test_it_failed(
    analysis: scan.Analysis, symbol: str, verdict: str, failed: str
) -> None:
    explanation = scan.explain(analysis, symbol, NAMES, LISTS)
    assert explanation.verdict == verdict
    assert [r.check for r in explanation.reasons if not r.passed][0] == failed


def test_a_listed_name_ranked_below_the_cut_is_eligible_but_not_picked() -> None:
    many = panel_of(
        *(
            stock(f"M{index:02d}", DAYS, daily=0.002 - 0.00005 * index, noise=0.002, seed=index)
            for index in range(25)
        )
    )
    members = {f"M{index:02d}" for index in range(25)}
    analysis = scan.analyse(many, members)
    weakest = str(analysis.within.index[-1])
    last = scan.explain(analysis, weakest, {}, {"SPUS": members})
    assert last.verdict == "eligible"
    assert last.reasons[-1].passed is False
    assert last.reasons[-1].detail == "rank 25 of 25"
    result = scan.scan(analysis, {}, {"SPUS": members}, [])
    assert len(result.candidates) == 25
    assert result.funnel.picked == 20


def test_an_unknown_symbol_is_excluded_for_having_no_data(analysis: scan.Analysis) -> None:
    explanation = scan.explain(analysis, "NOPE", {}, LISTS)
    assert explanation.verdict == "excluded"
    assert explanation.reasons[0].check == "Has a bar for the last session"


def test_only_the_last_year_of_bars_is_needed_for_the_same_answer() -> None:
    long = pd.concat(
        [stock(f"L{i}", 900, daily=0.001 + 0.0002 * i, noise=0.01, seed=i) for i in range(6)],
        ignore_index=True,
    )
    members = {f"L{i}" for i in range(6)}
    whole = scan.analyse(long, members)
    short = scan.analyse(scan.tail(long), members)
    assert whole.picks == short.picks
    assert whole.picks[rank.VARIANTS[0].label]


def test_the_window_the_scan_reads_covers_the_longest_feature() -> None:
    assert scan.LOOKBACK >= features.BETA_WINDOW // 2 + features.LONG
