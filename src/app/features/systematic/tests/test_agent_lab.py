from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import audit, sidecar
from app.features.systematic import agent_lab, agents, finrl_lab
from app.features.systematic.agent_lab import AiConfig
from app.features.systematic.agents import AgentDesk
from app.features.systematic.router import get_agents, get_automation
from app.features.systematic.tests.test_agents import CRYPTO, STOCKS, Sidecar, prices
from app.main import app
from app.quant import history

LAST = date(2026, 11, 2)


def data() -> pd.DataFrame:
    frame = prices(LAST, 900)
    rng = np.random.default_rng(0)
    moves = np.exp(np.cumsum(rng.normal(0.0004, 0.006, 900)))
    frame["close"] = np.tile(100 * moves, 2)
    return frame


def body(seed: int, algorithm: str, days: int = 600) -> dict[str, Any]:
    index = pd.bdate_range(end=LAST, periods=days)
    rng = np.random.default_rng(seed * 10 + agents.ALGORITHMS.index(algorithm))
    return {
        "dates": index.strftime("%Y-%m-%d").tolist(),
        "returns": rng.normal(0.0003, 0.008, days).tolist(),
        "weights": [[0.3, 0.3, 0.4]] * days,
        "assets": ["HLAL", "SPUS"],
        "held": [0.3, 0.3, 0.4],
        "in_sample": 1.0 + seed / 10,
    }


def outcome(seed: int, algorithm: str) -> sidecar.JobResult:
    return sidecar.JobResult(
        task="ai_walk_forward", status="ok", seconds=1, data=body(seed, algorithm)
    )


def judged(log: audit.AuditLog, fail: set[tuple[str, int]] | None = None) -> agent_lab.AiEvaluation:
    config = AiConfig()
    jobs = agent_lab.job_list(config, [])
    outcomes = [
        sidecar.JobResult(task="ai_walk_forward", status="error", seconds=1, error="boom")
        if (algorithm, seed) in (fail or set())
        else outcome(seed, algorithm)
        for algorithm, seed in jobs
    ]
    return agent_lab.assemble(data(), config, jobs, outcomes, 43, 0.0, log)


@pytest.fixture
def log(tmp_path: Path) -> audit.AuditLog:
    return audit.AuditLog(tmp_path / "audit.jsonl")


def test_the_slow_agent_goes_first_and_every_seed_of_every_agent_is_a_job() -> None:
    jobs = agent_lab.job_list(AiConfig(), [])
    assert [algorithm for algorithm, _ in jobs[:3]] == ["sac"] * 3
    assert len(jobs) == 9 and {seed for _, seed in jobs} == {1, 2, 3}


def test_the_retrain_dates_are_the_month_starts_from_the_date_given() -> None:
    days = pd.bdate_range("2023-03-20", "2023-07-31").strftime("%Y-%m-%d").tolist()
    assert agent_lab.retrain_dates(days, "2023-05-01") == ["2023-05-01", "2023-06-01", "2023-07-03"]


def test_each_agent_and_the_ensemble_is_judged_by_the_five_gates(log: audit.AuditLog) -> None:
    result = judged(log)
    assert [group.name for group in result.groups] == [
        "PPO agent",
        "SAC agent",
        "A2C agent",
        "UCB1 ensemble",
        "Regime ensemble",
    ]
    for group in result.groups:
        assert len(group.verdict.gates) == 5
        assert [row.label for row in group.verdict.rows] == ["seed 1", "seed 2", "seed 3"]
    assert result.retrains == 43 and result.errors == {}


def test_every_seed_of_every_agent_and_ensemble_counts_as_a_trial_once(log: audit.AuditLog) -> None:
    first = judged(log)
    assert first.total_trials == 15 and len(first.trial_hashes) == 15
    assert judged(log).total_trials == 15


def test_trials_of_earlier_evaluations_count_in_the_search(log: audit.AuditLog) -> None:
    judged(log)
    assert finrl_lab.known_trials(log, ["a-new-trial"]) == 16


def test_a_failed_run_is_reported_and_the_rest_are_still_judged(log: audit.AuditLog) -> None:
    result = judged(log, {("sac", 2)})
    assert result.errors == {"sac 2": "boom"}
    sac = next(group for group in result.groups if group.name == "SAC agent")
    assert [row.label for row in sac.verdict.rows] == ["seed 1", "seed 3"]
    for name in ("UCB1 ensemble", "Regime ensemble"):
        ensemble = next(group for group in result.groups if group.name == name)
        assert [row.label for row in ensemble.verdict.rows] == ["seed 1", "seed 3"]


def test_the_ensemble_is_built_from_the_agents_of_the_same_seed(log: audit.AuditLog) -> None:
    runs = {(name, 1): {**body(1, name), "seed": 1} for name in agents.ALGORITHMS}
    built = agent_lab.blend(runs, AiConfig(), 1)
    assert built is not None and len(built["returns"]) == 600 and built["seed"] == 1
    assert built["in_sample"] == pytest.approx(1.1)
    assert agent_lab.blend({("ppo", 1): runs["ppo", 1]}, AiConfig(), 1) is None


def test_the_weights_for_next_month_are_the_regime_ensembles_with_cash_the_rest(
    log: audit.AuditLog,
) -> None:
    weights = judged(log).weights
    assert set(weights) == set(agents.ALGORITHMS)
    assert all(0.0 <= value <= 1.0 for value in weights.values())
    assert sum(weights.values()) <= 1.0 + 1e-9


def test_the_result_is_recorded_and_read_back(log: audit.AuditLog) -> None:
    assert agent_lab.latest(log) is None
    result = judged(log)
    [entry] = log.entries()
    assert entry.kind == "ai-evaluation" and entry.payload["manifest"]
    again = agent_lab.latest(log)
    assert again is not None and again.model_dump() == result.model_dump()


def test_evaluating_runs_every_agent_and_seed_in_the_sidecar(
    monkeypatch: pytest.MonkeyPatch, log: audit.AuditLog
) -> None:
    calls: list[dict[str, Any]] = []

    def run(task: str, params: dict[str, Any], files: Any, timeout: float = 0) -> sidecar.JobResult:
        calls.append({"task": task, **params})
        return outcome(params["seed"], params["algorithm"])

    monkeypatch.setattr(sidecar, "available", lambda: True)
    monkeypatch.setattr(history, "ohlcv", lambda *_a, **_k: data())
    config = AiConfig(since="2024-01-01")
    result = agent_lab.evaluate(STOCKS, CRYPTO, log, config, workers=1, run=run)
    assert len(calls) == 9 and {call["task"] for call in calls} == {"ai_walk_forward"}
    assert {(call["algorithm"], call["seed"]) for call in calls} == {
        (a, s) for a in agents.ALGORITHMS for s in (1, 2, 3)
    }
    assert calls[0]["retrain_dates"][0] >= "2024-01-01" and calls[0]["lookback"] == agents.LOOKBACK
    assert result.retrains == len(calls[0]["retrain_dates"]) and len(result.groups) == 5


def test_evaluating_needs_the_sidecar(monkeypatch: pytest.MonkeyPatch, log: audit.AuditLog) -> None:
    monkeypatch.setattr(sidecar, "available", lambda: False)
    with pytest.raises(sidecar.SidecarUnavailableError, match="Set FinRL up first"):
        agent_lab.evaluate(STOCKS, CRYPTO, log)


@pytest.fixture
def served(tmp_path: Path, log: audit.AuditLog) -> Iterator[AgentDesk]:
    desk = AgentDesk(STOCKS, CRYPTO, log, tmp_path / "ai", Sidecar())
    automation = MagicMock()
    automation.audit = log
    app.dependency_overrides[get_agents] = lambda: desk
    app.dependency_overrides[get_automation] = lambda: automation
    yield desk
    app.dependency_overrides.pop(get_agents, None)
    app.dependency_overrides.pop(get_automation, None)


def test_the_agents_route_gives_their_state_and_the_latest_judgement(
    client: TestClient, served: AgentDesk, log: audit.AuditLog
) -> None:
    empty = client.get("/systematic/agents").json()
    assert empty["evaluation"] is None and empty["status"]["enabled"] is False
    judged(log)
    body = client.get("/systematic/agents").json()
    assert [group["name"] for group in body["evaluation"]["groups"]][-1] == "Regime ensemble"
    assert [agent["algorithm"] for agent in body["status"]["agents"]] == ["ppo", "sac", "a2c"]
    assert body["available"] == sidecar.available()


def kronos_body(seed: int) -> dict[str, Any]:
    found = body(seed, "ppo", 300)
    return {key: found[key] for key in ("dates", "returns", "assets")} | {"in_sample": 1.0}


def test_kronos_is_one_more_job_for_every_seed_and_is_judged_by_the_same_gates(
    log: audit.AuditLog,
) -> None:
    config = AiConfig(kronos=True)
    jobs = agent_lab.job_list(config, [])
    assert jobs[-3:] == [("kronos", 1), ("kronos", 2), ("kronos", 3)] and len(jobs) == 12
    outcomes = [
        sidecar.JobResult(task="kronos_walk_forward", status="ok", seconds=1, data=kronos_body(s))
        if name == "kronos"
        else outcome(s, name)
        for name, s in jobs
    ]
    result = agent_lab.assemble(data(), config, jobs, outcomes, 43, 0.0, log)
    group = next(g for g in result.groups if g.name == agent_lab.KRONOS_NAME)
    assert len(group.verdict.gates) == 5 and "2025-09-10" in group.note
    assert all(row.in_sample_sharpe == 1.0 for row in group.verdict.rows)
    assert result.total_trials == 18 and len(result.groups) == 6


def test_kronos_settings_never_change_what_an_agent_trial_is() -> None:
    plain, with_kronos = AiConfig(), AiConfig(kronos=True, kronos_samples=7)
    assert plain.trial_hash("ppo", 1) == with_kronos.trial_hash("ppo", 1)
    assert plain.trial_hash("kronos", 1) != with_kronos.trial_hash("kronos", 1)
    assert with_kronos.trial_hash("kronos", 1) != with_kronos.trial_hash("kronos", 2)


def test_kronos_needs_its_files_fetched_first(
    monkeypatch: pytest.MonkeyPatch, log: audit.AuditLog, tmp_path: Path
) -> None:
    monkeypatch.setattr(sidecar, "available", lambda: True)
    monkeypatch.setattr(agent_lab, "KRONOS_DIR", tmp_path)
    with pytest.raises(sidecar.SidecarUnavailableError, match="kronos-setup"):
        agent_lab.evaluate(STOCKS, CRYPTO, log, AiConfig(kronos=True))
