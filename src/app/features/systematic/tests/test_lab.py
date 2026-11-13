from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

from app.core import audit
from app.features.systematic import lab
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.tests.support import fake_history, serve, uptrend

TREND = STRATEGIES["halal-etf-trend"]
ROTATION = STRATEGIES["halal-etf-rotation"]
BTC = STRATEGIES["btc-trend-200"]
DONCHIAN = STRATEGIES["crypto-donchian"]
MULTI = STRATEGIES["halal-multi-trend"]


def noisy_market(symbols: list[str], drift: float, days: int = 1200, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(end=pd.Timestamp("2026-10-30"), periods=days)
    return pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, days))) for s in symbols},
        index=index,
    )


def clients() -> tuple[StockHistoricalDataClient, CryptoHistoricalDataClient]:
    return (
        create_autospec(StockHistoricalDataClient, instance=True),
        create_autospec(CryptoHistoricalDataClient, instance=True),
    )


def losses(edges: list[float], days: int = 1500, seed: int = 8) -> tuple[pd.Series, pd.DataFrame]:
    """Benchmark and candidate losses (negative returns); each candidate tracks the benchmark."""
    rng = np.random.default_rng(seed)
    bench = rng.normal(0.0003, 0.01, days)
    models = {f"m{i}": bench + edge + rng.normal(0, 0.004, days) for i, edge in enumerate(edges)}
    return pd.Series(-bench), pd.DataFrame(models) * -1


def test_the_step_down_finds_every_genuinely_better_model_and_no_other() -> None:
    benchmark, models = losses([0.0008, 0.0006, 0.0, 0.0, -0.0005])
    assert sorted(lab.superior_models(benchmark, models)) == ["m0", "m1"]


def test_the_step_down_survives_every_model_being_better() -> None:
    benchmark, models = losses([0.0008, 0.0009, 0.001])
    assert sorted(lab.superior_models(benchmark, models)) == ["m0", "m1", "m2"]


def test_the_step_down_finds_nothing_when_nothing_is_better() -> None:
    benchmark, models = losses([0.0, 0.0, -0.0004])
    assert lab.superior_models(benchmark, models) == []


def test_each_trend_rule_gets_its_neighbouring_windows_as_variants() -> None:
    trend = lab.variants(TREND)
    assert [c.spec.window for c in trend] == list(lab.SMA_WINDOWS)
    assert {c.spec.universe[0] for c in trend} == {"SPUS"}
    assert len({c.config_hash for c in trend}) == len(trend)


def test_rotation_variants_cross_window_and_selection_size() -> None:
    rotation = lab.variants(ROTATION)
    assert len(rotation) == len(lab.MOMENTUM_WINDOWS) * len(lab.MOMENTUM_TOP)
    assert {(c.spec.window, c.spec.top_n) for c in rotation} >= {(126, 2), (252, 1)}


def test_the_multi_speed_rule_gets_other_volatility_targets_and_speeds_as_variants() -> None:
    found = lab.variants(MULTI)
    assert [c.label for c in found] == [
        "12% volatility",
        "10% volatility",
        "15% volatility",
        "faster speeds",
        "slower speeds",
        "partial 50% a day",
        "5% band",
    ]
    assert found[0].spec == MULTI
    assert [c.spec.target_vol for c in found[:3]] == [0.12, 0.10, 0.15]
    assert found[3].spec.lookbacks == [25, 50, 100, 150, 200]
    assert found[5].spec.trade_rate == 0.5 and found[6].spec.band == 0.05
    assert len({c.config_hash for c in found}) == len(found)


def test_donchian_is_left_out_of_the_lab() -> None:
    assert lab.variants(DONCHIAN) == []


def test_a_variants_hash_changes_with_its_parameters_only() -> None:
    first, second = lab.variants(TREND)[:2]
    assert first.config_hash != second.config_hash
    assert first.config_hash == lab.variants(TREND)[0].config_hash


def test_every_variant_is_compared_over_the_dates_all_of_them_can_trade() -> None:
    closes = noisy_market(["SPUS", "HLAL"], 0.0006)
    returns = lab.family_returns(closes, lab.variants(TREND))
    assert list(returns.columns) == [c.label for c in lab.variants(TREND)]
    assert len(returns) == len(closes) - max(lab.SMA_WINDOWS) - 20
    assert not returns.isna().any().any()


def test_a_family_with_a_real_edge_is_reported_with_every_statistic() -> None:
    result = lab.evaluate_family(
        "halal-etf-trend", noisy_market(["SPUS", "HLAL"], 0.0009), lab.variants(TREND), 4
    )

    assert result.trials == result.cumulative_trials == 4
    assert result.reality_check_p < 0.05
    assert 0 <= result.probability_of_overfitting <= 1
    assert [c.label for c in result.candidates] == [c.label for c in lab.variants(TREND)]
    assert any(c.beats_cash for c in result.candidates)
    assert all(0 <= c.deflated_sharpe <= 1 for c in result.candidates)
    assert result.candidate_hashes == [c.config_hash for c in lab.variants(TREND)]


def test_a_family_with_no_edge_has_no_survivors_and_no_reality_check_win() -> None:
    result = lab.evaluate_family(
        "halal-etf-trend", noisy_market(["SPUS", "HLAL"], 0.0), lab.variants(TREND), 4
    )

    assert result.reality_check_p > 0.05
    assert not any(c.survives for c in result.candidates)
    assert lab.checks_pass(result) == []


def test_more_variants_ever_tried_make_survival_harder() -> None:
    closes = noisy_market(["SPUS", "HLAL"], 0.0007)
    few = lab.evaluate_family("halal-etf-trend", closes, lab.variants(TREND), 4)
    many = lab.evaluate_family("halal-etf-trend", closes, lab.variants(TREND), 400)

    assert many.cumulative_trials == 400
    for easy, hard in zip(few.candidates, many.candidates, strict=True):
        assert hard.deflated_sharpe <= easy.deflated_sharpe


def test_too_little_history_is_refused() -> None:
    with pytest.raises(ValueError, match="common trading days"):
        lab.evaluate_family(
            "halal-etf-trend", noisy_market(["SPUS", "HLAL"], 0.0, days=300), lab.variants(TREND), 4
        )


def test_the_same_variants_run_twice_count_once_but_new_ones_add_up(tmp_path: Path) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    labels = [c.label for c in lab.variants(TREND)]
    assert lab.known_trials(log, "halal-etf-trend", labels) == 4

    entry = {
        "result": {
            "families": [
                {
                    "family": "halal-etf-trend",
                    "candidate_hashes": ["old-hash-1", "old-hash-2"],
                    "candidates": [{"label": label} for label in labels],
                }
            ]
        },
    }
    log.record("lab", entry)
    log.record("lab", entry)
    assert lab.known_trials(log, "halal-etf-trend", labels) == 4
    assert lab.known_trials(log, "halal-etf-trend", [*labels, "300-day average"]) == 5
    assert lab.known_trials(log, "halal-etf-rotation", ["x"]) == 1


def test_the_search_reports_the_spread_of_each_familys_sharpe_ratios(tmp_path: Path) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    assert lab.search(log).spreads == {}
    result = lab.evaluate_family(
        "halal-etf-trend", noisy_market(["SPUS", "HLAL"], 0.0009), lab.variants(TREND), 4
    )
    log.record(
        "lab",
        {"result": lab.LabResult(families=[result], errors={}, seconds=0, workers=1).model_dump()},
    )
    sharpes = [c.sharpe / 252**0.5 for c in result.candidates]
    found = lab.search(log)
    assert found.spreads["halal-etf-trend"] == pytest.approx(float(np.std(sharpes, ddof=1)))
    assert found.trials == 24


def test_running_the_lab_tests_every_family_and_leaves_an_audit_trail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_history(monkeypatch)
    log = audit.AuditLog(tmp_path / "audit.jsonl")

    result = lab.run_lab(*clients(), log, workers=1)

    assert {f.family for f in result.families} == {
        "btc-trend-200",
        "halal-etf-trend",
        "halal-multi-trend",
        "halal-etf-rotation",
    }
    assert result.errors == {}
    assert log.verify().valid
    entry = log.entries()[0]
    assert entry.kind == "lab"
    assert entry.payload["manifest"]["seed"] == lab.SEED
    assert set(entry.payload["manifest"]["data"]) == {f.family for f in result.families}
    assert lab.latest(log) == result


def test_a_family_that_fails_is_reported_and_the_rest_still_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.quant import history

    fake_history(monkeypatch)
    short = uptrend(["BTC/USD"], 300)
    fetch = history.daily_prices

    def closes(
        s: StockHistoricalDataClient,
        c: CryptoHistoricalDataClient,
        symbols: list[str],
        years: float,
    ) -> pd.DataFrame:
        return short if symbols == ["BTC/USD"] else fetch(s, c, symbols, years).closes

    serve(monkeypatch, closes)
    log = audit.AuditLog(tmp_path / "audit.jsonl")

    result = lab.run_lab(*clients(), log, workers=1)

    assert "btc-trend-200" in result.errors
    assert {f.family for f in result.families} == {
        "halal-etf-trend",
        "halal-multi-trend",
        "halal-etf-rotation",
    }


def test_no_lab_result_exists_before_the_first_run(tmp_path: Path) -> None:
    assert lab.latest(audit.AuditLog(tmp_path / "audit.jsonl")) is None


def test_a_result_stored_before_a_field_was_added_still_loads_without_it(tmp_path: Path) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    candidates = fake_candidates()
    for candidate in candidates:
        del candidate["consistency"]
    log.record("lab", {"result": fake_lab_result(candidates)})

    loaded = lab.latest(log)

    assert loaded is not None
    assert all(c.consistency is None for c in loaded.families[0].candidates)


def test_a_stored_result_that_no_longer_fits_the_schema_reads_as_not_run(tmp_path: Path) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    log.record("lab", {"result": {"families": [{"family": "halal-etf-trend"}]}})
    assert lab.latest(log) is None


def fake_candidates() -> list[dict[str, object]]:
    return [
        {
            "label": "200-day average",
            "sharpe": 1.0,
            "cagr": 0.1,
            "max_drawdown": -0.2,
            "t_statistic": 2.0,
            "deflated_sharpe": 0.9,
            "consistency": 0.75,
            "clears_hurdle": False,
            "beats_cash": True,
            "survives": False,
        }
    ]


def fake_lab_result(candidates: list[dict[str, object]]) -> dict[str, object]:
    family = {
        "family": "halal-etf-trend",
        "title": "Shariah ETF 200-day trend",
        "days": 500,
        "start": "2024-01-01T00:00:00",
        "end": "2026-01-01T00:00:00",
        "trials": 4,
        "cumulative_trials": 4,
        "reality_check_p": 0.02,
        "probability_of_overfitting": 0.5,
        "candidates": candidates,
        "candidate_hashes": ["a"],
    }
    return {"families": [family], "errors": {}, "seconds": 1.0, "workers": 1}
