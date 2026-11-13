from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from app.core import audit
from app.features.scanner import evaluate, features, rank
from app.features.scanner.schemas import Evaluation
from app.features.scanner.tests.support import random_panel
from app.quant.history import NotEnoughHistoryError

NAMES = {f"S{index:02d}" for index in range(40)}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return random_panel(symbols=40, days=1300)


@pytest.fixture(scope="module")
def result(panel: pd.DataFrame) -> Evaluation:
    return evaluate.evaluate(panel, NAMES, ["SPUS 2026-11-02"])


def test_every_variant_is_run_and_judged_together(result: Evaluation) -> None:
    assert [variant.label for variant in result.variants] == [v.label for v in rank.VARIANTS]
    assert result.trials == 10
    for variant in result.variants:
        assert len(variant.checks) == 5
        assert variant.checks[-1].name.endswith("(10 trials)")
        assert [point.fee_bps for point in variant.sensitivity] == [10.0, 25.0, 50.0]
        assert 0.0 <= variant.deflated_sharpe <= 1.0


def test_the_benchmark_holds_the_same_names_without_picking(result: Evaluation) -> None:
    assert result.benchmark.exposure > 0.99
    assert all(variant.metrics.exposure <= 1.0 + 1e-9 for variant in result.variants)
    assert result.symbols == 40


def test_the_luck_the_search_could_have_had_and_the_overfitting_risk_are_reported(
    result: Evaluation,
) -> None:
    assert result.luck_bar > 0
    assert 0.0 <= result.probability_of_overfitting <= 1.0


def test_the_window_runs_from_the_first_holding_to_the_last_session(
    result: Evaluation, panel: pd.DataFrame
) -> None:
    assert result.end == panel["date"].max().date()
    assert result.days > evaluate.MIN_DAYS
    assert result.start < result.end


def test_the_limits_of_the_data_are_stated(result: Evaluation) -> None:
    assert result.lists == ["SPUS 2026-11-02"]
    assert any("today's" in note for note in result.notes)
    assert any("delisted" in note.lower() for note in result.notes)


def test_only_the_listed_names_are_traded(panel: pd.DataFrame) -> None:
    result = evaluate.evaluate(panel, {"S01", "S02", "S04", "S05", "S07", "S08"}, [])
    assert result.symbols == 6


def test_strategy_returns_never_use_data_from_after_the_day(panel: pd.DataFrame) -> None:
    full = features.compute(panel)
    ok = rank.eligible(full)
    scored = rank.scores(full, ok)
    cut = full.close.index[1000]
    until = features.compute(panel[panel["date"] <= cut])
    ok_until = rank.eligible(until)
    scored_until = rank.scores(until, ok_until)
    for variant in rank.VARIANTS:
        whole = evaluate.run_variant(full, ok, scored, variant)
        part = evaluate.run_variant(until, ok_until, scored_until, variant)
        assert whole.held.loc[:cut].abs().sum().sum() > 0
        pdt.assert_series_equal(part.returns, whole.returns.loc[:cut], rtol=1e-9)
        pdt.assert_frame_equal(part.held, whole.held.loc[:cut], rtol=1e-9)


def test_too_little_history_or_no_listed_name_is_refused(panel: pd.DataFrame) -> None:
    first = panel["date"].min()
    short = panel[panel["date"] < first + pd.Timedelta(days=500)]
    with pytest.raises(NotEnoughHistoryError):
        evaluate.evaluate(short, NAMES, [])
    with pytest.raises(NotEnoughHistoryError):
        evaluate.evaluate(panel, {"NOT-HERE"}, [])


def test_the_latest_stored_evaluation_is_read_back(tmp_path: Path, result: Evaluation) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    assert evaluate.latest(log) is None
    log.record("scan-evaluation", {"result": result.model_dump(mode="json")})
    log.record("run", {"unrelated": True})
    assert evaluate.latest(log) == result


def test_an_evaluation_from_an_incompatible_version_reads_as_none(tmp_path: Path) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    log.record("scan-evaluation", {"result": {"start": "2020-01-01"}})
    assert evaluate.latest(log) is None
