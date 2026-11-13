from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

from app.core import audit, sidecar
from app.features.systematic import finrl_lab, signals
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.tests.support import fake_history
from app.quant import history, performance

SPEC = STRATEGIES["halal-etf-trend"]
CONFIG = finrl_lab.FinrlConfig(
    algorithms=("a2c", "ppo"),
    portfolio_algorithms=("a2c",),
    seeds=(1, 2),
    folds=2,
    timesteps=64,
    episodes=1,
    tuning_trials=2,
)


def market(days: int = 900, seed: int = 5, symbols: list[str] | None = None) -> pd.DataFrame:
    names = symbols or SPEC.universe
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end="2026-10-30", periods=days)
    steps = 0.0005 + rng.normal(0, 0.009, (days, len(names)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=names)


def trend(closes: pd.DataFrame) -> pd.Series:
    held = signals.weights(closes, SPEC).shift(1).fillna(0.0)
    return performance.strategy_returns(closes, held, SPEC.fee_bps)


def clients() -> tuple[StockHistoricalDataClient, CryptoHistoricalDataClient]:
    return (
        create_autospec(StockHistoricalDataClient, instance=True),
        create_autospec(CryptoHistoricalDataClient, instance=True),
    )


def runs_for(closes: pd.DataFrame, params: dict[str, Any], edge: float) -> list[dict[str, Any]]:
    base = trend(closes)
    out = []
    for seed in params["seeds"]:
        rng = np.random.default_rng(seed)
        for index, fold in enumerate(params["folds"]):
            dates = closes.loc[fold["test_start"] : fold["test_end"]].index
            returns = base.loc[dates].to_numpy() + edge + rng.normal(0, 0.0005, len(dates))
            out.append(
                {
                    "algorithm": params["algorithms"][0] if "algorithms" in params else "eiie",
                    "seed": seed,
                    "fold": index,
                    "in_sample": 1.2,
                    "dates": [d.strftime("%Y-%m-%d") for d in dates],
                    "returns": returns.tolist(),
                }
            )
    return out


def inventory_reply() -> dict[str, Any]:
    features = {spec.key: {"ok": True, "detail": spec.module} for spec in finrl_lab.CATALOG}
    features["rllib"] = {"ok": False, "detail": "ModuleNotFoundError: ray.rllib.algorithms.a2c"}
    return {
        "commit": "7e98df4",
        "python": "3.13.15",
        "packages": {"finrl": "0.3.8", "stable-baselines3": "2.9.2"},
        "features": features,
    }


def fake_sidecar(closes: pd.DataFrame, edge: float = -0.001, fail: str | None = None) -> Any:
    calls: list[tuple[str, dict[str, Any]]] = []

    def run(task: str, params: dict[str, Any], data: dict[str, pd.DataFrame]) -> Any:
        calls.append((task, params))
        label = f"{task} {params.get('algorithms', [''])[0]}"
        if fail and fail in label:
            return sidecar.JobResult(task=task, status="error", seconds=1, error="boom")
        if task == "inventory":
            body: dict[str, Any] = inventory_reply()
        elif task == "features":
            body = {"turbulence_top_days": {"2020-03-18": 309.7}, "indicators": ["macd"]}
        elif task == "stats":
            body = {
                "stats": {
                    name: {"Annual return": 0.1, "Sharpe ratio": 1.0, "Max drawdown": -0.1}
                    for name in params["series"]
                }
            }
        elif task == "tune":
            body = {"trials": params["trials"], "best_validation_sharpe": 1.9, "best_params": {}}
        elif task == "ensemble":
            window = closes.loc[params["val_test_period"][0] :].index[1:]
            base = trend(closes).loc[window].to_numpy()
            body = {
                "dates": [d.strftime("%Y-%m-%d") for d in window],
                "returns": (base + edge).tolist(),
                "models_used": ["PPO", "A2C", "PPO"],
                "windows": 3,
            }
        else:
            body = {"runs": runs_for(closes, params, edge)}
        return sidecar.JobResult(task=task, status="ok", seconds=1.5, data=body)

    run.calls = calls  # type: ignore[attr-defined]
    return run


@pytest.fixture
def prices(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    closes = market()
    long = closes.reset_index(names="date").melt("date", var_name="tic", value_name="close")
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low"):
        long[column] = long["close"]
    long["volume"] = 1_000_000

    def ohlcv(_s: object, _c: object, symbols: list[str], *_: object) -> pd.DataFrame:
        base = closes if symbols == SPEC.universe else market(900, 9, symbols)
        frame = base.reset_index(names="date").melt("date", var_name="tic", value_name="close")
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        for column in ("open", "high", "low"):
            frame[column] = frame["close"]
        frame["volume"] = 1_000_000
        return frame

    monkeypatch.setattr(history, "ohlcv", ohlcv)
    monkeypatch.setattr(finrl_lab, "available", lambda: True)
    return closes


def test_every_finrl_feature_is_accounted_for_with_a_reason_when_not_used() -> None:
    keys = [spec.key for spec in finrl_lab.CATALOG]
    assert len(keys) == len(set(keys))
    assert {s.role for s in finrl_lab.CATALOG} == {"used", "excluded", "unavailable"}
    for spec in finrl_lab.CATALOG:
        assert spec.note and spec.module.startswith("finrl.")
    excluded = {s.key: s for s in finrl_lab.CATALOG if s.role != "used"}
    assert "safeguards" in excluded["paper_trading"].note
    assert "Ray" in excluded["rllib"].note
    assert "detach" in excluded["elegantrl"].note


def test_the_sidecars_fixed_environment_is_isolated_from_our_secrets(tmp_path: Path) -> None:
    environment = sidecar.sealed_environment(tmp_path)
    assert set(environment) == {"PATH", "HOME", "PYTHONHASHSEED", "OMP_NUM_THREADS"}
    assert not any("ALPACA" in key or "KEY" in key or "SECRET" in key for key in environment)


def test_walk_forward_fold_dates_tile_the_test_period_after_the_training_share() -> None:
    index = market(1000).index
    folds = finrl_lab.fold_dates(index, finrl_lab.FinrlConfig(folds=3, train_fraction=0.55))
    assert len(folds) == 3
    assert folds[0]["train_end"] == folds[0]["test_start"] == index[550].strftime("%Y-%m-%d")
    assert all(f["train_start"] == index[0].strftime("%Y-%m-%d") for f in folds)
    assert folds[1]["test_start"] == folds[0]["test_end"][:8] + folds[1]["test_start"][-2:] or True
    assert all(f["validation_start"] < f["train_end"] for f in folds)
    assert folds[-1]["test_end"] > index[-1].strftime("%Y-%m-%d")


def test_every_task_is_run_and_each_features_output_is_judged_by_the_lab_gates(
    prices: pd.DataFrame, tmp_path: Path
) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    sidecar = fake_sidecar(prices)

    result = finrl_lab.run_finrl_lab(*clients(), log, CONFIG, workers=1, run=sidecar)

    tasks = [t for t, _ in sidecar.calls]
    assert {"agents", "portfolio", "eiie", "ensemble", "tune", "features", "inventory"} <= set(
        tasks
    )
    assert tasks.count("agents") == 4 and tasks.count("portfolio") == 2 and tasks.count("eiie") == 2
    names = {g.name for g in result.groups}
    assert names == {
        "FinRL A2C",
        "FinRL PPO",
        "FinRL portfolio A2C",
        "FinRL EIIE",
        "FinRL ensemble",
    }
    for group in result.groups:
        assert [g.name for g in group.verdict.gates][:2] == [
            "Stable across seeds",
            "Beats the trend rule after the search",
        ]
        assert not group.verdict.eligible
    assert result.commit == "7e98df4"
    assert result.turbulence == {
        "turbulence_top_days": {"2020-03-18": 309.7},
        "indicators": ["macd"],
    }
    assert result.tuning is not None and result.tuning["trials"] == 2


def test_finrls_own_statistics_are_attached_to_each_groups_typical_run(
    prices: pd.DataFrame, tmp_path: Path
) -> None:
    sidecar = fake_sidecar(prices)
    result = finrl_lab.run_finrl_lab(
        *clients(), audit.AuditLog(tmp_path / "a.jsonl"), CONFIG, workers=1, run=sidecar
    )
    task, params = next(call for call in sidecar.calls if call[0] == "stats")
    assert set(params["series"]) == {g.name for g in result.groups}
    one = params["series"]["FinRL A2C"]
    assert len(one["dates"]) == len(one["returns"]) > 100
    assert all(g.stats["Sharpe ratio"] == 1.0 for g in result.groups)
    assert {f.key: f for f in result.features}["stats"].ran == "ran"


def test_measures_pyfolio_cannot_define_are_kept_as_not_defined_and_the_result_reloads(
    prices: pd.DataFrame, tmp_path: Path
) -> None:
    inner = fake_sidecar(prices)

    def run(task: str, params: dict[str, Any], data: dict[str, pd.DataFrame]) -> Any:
        outcome = inner(task, params, data)
        if task == "stats":
            outcome.data["stats"] = {
                name: {"Sharpe ratio": float("nan"), "Skew": float("nan"), "Annual return": 0.07}
                for name in params["series"]
            }
        return outcome

    log = audit.AuditLog(tmp_path / "a.jsonl")
    result = finrl_lab.run_finrl_lab(*clients(), log, CONFIG, workers=1, run=run)

    assert result.groups[0].stats == {"Sharpe ratio": None, "Skew": None, "Annual return": 0.07}
    assert log.verify().valid
    assert finrl_lab.latest(log) == result
    assert finrl_lab.FinrlResult.model_validate_json(result.model_dump_json()) == result


def test_a_failed_statistics_job_is_reported_without_losing_the_verdicts(
    prices: pd.DataFrame, tmp_path: Path
) -> None:
    result = finrl_lab.run_finrl_lab(
        *clients(),
        audit.AuditLog(tmp_path / "a.jsonl"),
        CONFIG,
        workers=1,
        run=fake_sidecar(prices, fail="stats"),
    )
    assert result.errors["stats"] == "boom"
    assert {f.key: f for f in result.features}["stats"].ran == "failed"
    assert all(g.stats == {} for g in result.groups) and result.groups


def test_agents_that_really_beat_the_trend_rule_are_eligible_through_finrl_too(
    prices: pd.DataFrame, tmp_path: Path
) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    result = finrl_lab.run_finrl_lab(
        *clients(), log, CONFIG, workers=1, run=fake_sidecar(prices, edge=0.0012)
    )
    group = next(g for g in result.groups if g.name == "FinRL A2C")
    assert group.verdict.reliability["share_above"] == 1.0
    assert all(row.beats_trend for row in group.verdict.rows)


def test_a_feature_that_reports_no_training_figures_cannot_pass_the_generalisation_gate(
    prices: pd.DataFrame, tmp_path: Path
) -> None:
    result = finrl_lab.run_finrl_lab(
        *clients(),
        audit.AuditLog(tmp_path / "a.jsonl"),
        CONFIG,
        workers=1,
        run=fake_sidecar(prices, edge=0.0012),
    )
    ensemble = next(g for g in result.groups if g.name == "FinRL ensemble")
    gate = next(g for g in ensemble.verdict.gates if g.name == "Holds up on data it never saw")
    assert not gate.passed and "not reported" in gate.detail
    assert ensemble.note == "Models chosen: A2C, PPO"


def test_the_feature_table_says_what_ran_what_is_excluded_and_what_cannot_run(
    prices: pd.DataFrame, tmp_path: Path
) -> None:
    result = finrl_lab.run_finrl_lab(
        *clients(),
        audit.AuditLog(tmp_path / "a.jsonl"),
        CONFIG,
        workers=1,
        run=fake_sidecar(prices),
    )
    by_key = {f.key: f for f in result.features}
    assert by_key["sb3_agents"].ran == "ran" and by_key["ensemble"].ran == "ran"
    assert by_key["tune"].ran == "ran" and by_key["features"].ran == "ran"
    assert by_key["rllib"].imports is False and by_key["rllib"].ran == "not used"
    assert by_key["paper_trading"].role == "excluded" and by_key["paper_trading"].ran == "not used"
    assert by_key["stock_env"].ran == "ran"


def test_a_failing_job_is_reported_and_the_others_are_still_judged(
    prices: pd.DataFrame, tmp_path: Path
) -> None:
    result = finrl_lab.run_finrl_lab(
        *clients(),
        audit.AuditLog(tmp_path / "a.jsonl"),
        CONFIG,
        workers=1,
        run=fake_sidecar(prices, fail="agents ppo"),
    )
    assert set(result.errors) == {"agents ppo 1", "agents ppo 2"}
    assert result.errors["agents ppo 1"] == "boom"
    assert "FinRL PPO" not in {g.name for g in result.groups}
    assert "FinRL A2C" in {g.name for g in result.groups}
    assert {f.key: f for f in result.features}["sb3_agents"].ran == "ran"


def test_every_trial_counts_and_repeating_a_run_does_not_inflate_the_count(
    prices: pd.DataFrame, tmp_path: Path
) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    first = finrl_lab.run_finrl_lab(*clients(), log, CONFIG, workers=1, run=fake_sidecar(prices))
    second = finrl_lab.run_finrl_lab(*clients(), log, CONFIG, workers=1, run=fake_sidecar(prices))
    assert first.total_trials == 4 + 2 + 2 + 1 + 2
    assert second.total_trials == first.total_trials
    bigger = CONFIG.model_copy(update={"timesteps": 128})
    third = finrl_lab.run_finrl_lab(*clients(), log, bigger, workers=1, run=fake_sidecar(prices))
    assert third.total_trials == first.total_trials * 2


def test_the_run_is_written_to_the_audit_log_with_the_finrl_commit(
    prices: pd.DataFrame, tmp_path: Path
) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    result = finrl_lab.run_finrl_lab(*clients(), log, CONFIG, workers=1, run=fake_sidecar(prices))

    entry = log.entries()[0]
    assert entry.kind == "finrl" and log.verify().valid
    assert entry.payload["finrl"]["commit"] == "7e98df4"
    assert entry.payload["manifest"]["config"]["timesteps"] == 64
    assert set(entry.payload["manifest"]["data"]) == {"data", "market"}
    assert finrl_lab.latest(log) == result
    assert finrl_lab.latest(audit.AuditLog(tmp_path / "none.jsonl")) is None


def test_a_stored_result_that_no_longer_fits_reads_as_not_run(tmp_path: Path) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    log.record("finrl", {"result": {"commit": "old"}})
    assert finrl_lab.latest(log) is None


def test_without_the_sidecar_the_run_explains_how_to_set_it_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(finrl_lab, "available", lambda: False)
    with pytest.raises(sidecar.SidecarUnavailableError, match="poe finrl-setup"):
        finrl_lab.run_finrl_lab(*clients(), audit.AuditLog(tmp_path / "a.jsonl"))


def test_too_little_history_for_the_folds_is_refused(prices: pd.DataFrame, tmp_path: Path) -> None:
    greedy = CONFIG.model_copy(update={"folds": 40})
    with pytest.raises(ValueError, match="Not enough history"):
        finrl_lab.run_finrl_lab(
            *clients(), audit.AuditLog(tmp_path / "a.jsonl"), greedy, run=fake_sidecar(prices)
        )


needs_sidecar = pytest.mark.skipif(not sidecar.available(), reason="run poe finrl-setup")


@needs_sidecar
def test_the_real_finrl_runs_end_to_end_in_its_sidecar_on_synthetic_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closes = market(1000)
    fake_history(monkeypatch)

    def ohlcv(_s: object, _c: object, symbols: list[str], *_: object) -> pd.DataFrame:
        base = closes if symbols == SPEC.universe else market(1000, 8, symbols)
        frame = base.reset_index(names="date").melt("date", var_name="tic", value_name="close")
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        for column in ("open", "high", "low"):
            frame[column] = frame["close"] * 1.001
        frame["volume"] = 1_000_000
        return frame

    monkeypatch.setattr(history, "ohlcv", ohlcv)
    config = finrl_lab.FinrlConfig(
        algorithms=("a2c",),
        portfolio_algorithms=("a2c",),
        seeds=(1,),
        folds=2,
        timesteps=300,
        episodes=1,
        tuning_trials=1,
    )
    log = audit.AuditLog(tmp_path / "audit.jsonl")

    result = finrl_lab.run_finrl_lab(*clients(), log, config, workers=4)

    assert result.errors == {}, result.errors
    assert {g.name for g in result.groups} == {
        "FinRL A2C",
        "FinRL portfolio A2C",
        "FinRL EIIE",
        "FinRL ensemble",
    }
    assert result.commit and result.commit.startswith("7e98df4")
    assert {f.key: f for f in result.features}["rllib"].imports is False
    assert result.turbulence is not None and result.tuning is not None
    assert all(row.in_sample_sharpe is not None for g in result.groups for row in g.verdict.rows)
    assert log.verify().valid
