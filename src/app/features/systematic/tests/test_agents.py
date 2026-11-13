import asyncio
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

from app.core import audit
from app.core.sidecar import JobResult
from app.features.systematic import agents
from app.features.systematic.agents import AgentDesk
from app.features.systematic.tests.support import serve
from app.quant import history

STOCKS = create_autospec(StockHistoricalDataClient, instance=True)
CRYPTO = create_autospec(CryptoHistoricalDataClient, instance=True)
TODAY = date(2026, 11, 3)
LATEST = date(2026, 11, 2)
ASSETS = ["HLAL", "SPUS"]
GROUPS = [
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_60",
    "vol_20",
    "macd",
    "boll",
    "rsi",
    "cci",
    "dx",
    "sma_30",
    "sma_60",
    "held",
]
COLUMNS = [f"{name}:{asset}" for name in GROUPS[:-1] for asset in ASSETS] + [
    f"held:{asset}" for asset in [*ASSETS, "cash"]
]


def prices(last: date, days: int = 1200) -> pd.DataFrame:
    index = pd.bdate_range(end=last, periods=days)
    frames = [
        pd.DataFrame({"date": index.strftime("%Y-%m-%d"), "tic": tic, "close": 100.0})
        for tic in agents.UNIVERSE
    ]
    return pd.concat(frames).reset_index(drop=True)


class Sidecar:
    """A stand-in for the FinRL sidecar that records what it was asked and answers in its shape."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail: set[tuple[str, str]] = set()
        self.last = LATEST

    def __call__(
        self, task: str, params: dict[str, Any], data: dict[str, pd.DataFrame], timeout: float = 0
    ) -> JobResult:
        self.calls.append((task, params))
        if (task, params["algorithm"]) in self.fail:
            return JobResult(task=task, status="error", seconds=1, error="boom")
        days = pd.bdate_range(end=self.last, periods=1200)
        body: dict[str, Any]
        if task == "ai_train":
            Path(params["model_path"]).write_text("model")
            Path(params["scaler_path"]).write_text("{}")
            body = {
                "first": str(days[-756].date()),
                "last": str(days[-1].date()),
                "days": 756,
                "assets": ["HLAL", "SPUS"],
                "in_sample_sharpe": 1.1,
            }
        elif task == "ai_walk_forward":
            window = days[-100:]
            body = {
                "dates": window.strftime("%Y-%m-%d").tolist(),
                "returns": np.full(len(window), 0.0004).tolist(),
                "weights": [[0.4, 0.4, 0.2]] * len(window),
                "assets": ["HLAL", "SPUS"],
                "held": [0.4, 0.4, 0.2],
                "in_sample": 1.1,
            }
        elif task == "ai_explain":
            values = [[0.1 / 13, 0.0, -0.1 / 13] for _ in GROUPS]
            decision = {
                "weights": [0.5, 0.3, 0.2],
                "base": [0.4, 0.3, 0.3],
                "values": values,
                "residual": 1e-8,
            }
            body = {
                "groups": GROUPS,
                "assets": [*ASSETS, "cash"],
                "columns": COLUMNS,
                "background": {
                    "first": params["first"],
                    "last": params["last"],
                    "days": 756,
                    "rows": 64,
                    "seed": 0,
                },
                "decisions": [decision] * len(params["observations"]),
                "overall": [[0.02, 0.01, 0.01]] * len(GROUPS) if params["sample"] else None,
                "sample": params["sample"],
                "shap": "0.54.1",
            }
        else:
            since = date.fromisoformat(params["since"]) if params["since"] else self.last
            fresh = [day.strftime("%Y-%m-%d") for day in days if day.date() > since]
            if not fresh:
                decided = [str(self.last)]
            else:
                decided = fresh if params["target"] else [params["since"], *fresh]
            body = {
                "dates": fresh,
                "returns": [0.001] * len(fresh),
                "applied": [[0.3, 0.3, 0.4]] * len(fresh),
                "held": [0.3, 0.3, 0.4],
                "target": [0.5, 0.3, 0.2],
                "assets": ["HLAL", "SPUS"],
                "last": str(self.last),
                "decisions": [
                    {"date": day, "observation": [0.1] * len(COLUMNS)} for day in decided
                ],
            }
        return JobResult(task=task, status="ok", seconds=1, data=body)

    def tasks(self) -> list[str]:
        return [task for task, _ in self.calls]


@pytest.fixture
def clock() -> list[date]:
    return [TODAY]


@pytest.fixture
def sidecar() -> Sidecar:
    return Sidecar()


@pytest.fixture
def desk(
    tmp_path: Path,
    sidecar: Sidecar,
    clock: list[date],
    monkeypatch: pytest.MonkeyPatch,
) -> AgentDesk:
    monkeypatch.setattr(history, "ohlcv", lambda *_a, **_k: prices(sidecar.last))

    def closes(_s: object, _c: object, symbols: list[str], _y: float) -> pd.DataFrame:
        frame = prices(sidecar.last).pivot(index="date", columns="tic", values="close")[symbols]
        frame.index = pd.to_datetime(frame.index)
        return frame

    serve(monkeypatch, closes)
    found = AgentDesk(
        STOCKS,
        CRYPTO,
        audit.AuditLog(tmp_path / "audit.jsonl"),
        tmp_path / "ai",
        sidecar,
        lambda: clock[0],
    )
    found.enabled = True
    return found


def test_the_months_start_on_their_first_trading_day() -> None:
    days = pd.bdate_range("2026-08-27", "2026-11-02").strftime("%Y-%m-%d").tolist()
    assert agents.month_starts(days, 2) == ["2026-10-01", "2026-11-02"]
    assert agents.month_starts(days, 10) == ["2026-08-27", "2026-09-01", "2026-10-01", "2026-11-02"]


def test_a_new_desk_has_no_model_and_every_agent_is_due(desk: AgentDesk) -> None:
    assert all(desk.due(name) for name in agents.ALGORITHMS)
    status = desk.status()
    assert [agent.algorithm for agent in status.agents] == ["ppo", "sac", "a2c"]
    assert all(agent.model is None and agent.days == 0 for agent in status.agents)
    assert status.ensemble == {} and status.ensemble_sharpe is None and status.enabled


def test_a_desk_that_is_off_does_nothing(desk: AgentDesk, sidecar: Sidecar) -> None:
    desk.enabled = False
    desk.run()
    assert sidecar.calls == []


def test_the_first_run_gives_each_agent_a_record_then_trains_and_follows_it(
    desk: AgentDesk, sidecar: Sidecar
) -> None:
    desk.run()
    assert sidecar.tasks() == ["ai_walk_forward", "ai_train", "ai_shadow", "ai_explain"] * 3
    assert [params["algorithm"] for _, params in sidecar.calls[::4]] == ["ppo", "sac", "a2c"]
    for algorithm in agents.ALGORITHMS:
        model, shadow = desk.state.models[algorithm], desk.state.shadow[algorithm]
        assert model.trained == TODAY and model.seed == agents.SEED and model.days == 756
        assert shadow.backfilled and shadow.since == LATEST and shadow.target == [0.5, 0.3, 0.2]
        assert shadow.dates[-1] == shadow.since or shadow.dates[-1] <= LATEST


@pytest.mark.usefixtures("desk")
def test_the_first_check_comes_at_startup_not_half_an_hour_later(
    tmp_path: Path, sidecar: Sidecar, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agents, "FIRST_RUN", timedelta(milliseconds=30))

    async def scenario() -> list[str]:
        started = AgentDesk(
            STOCKS, CRYPTO, audit.AuditLog(tmp_path / "log.jsonl"), tmp_path / "started", sidecar
        )
        started.enabled = True
        started.start()
        for _ in range(200):
            if len(sidecar.calls) >= 12:
                break
            await asyncio.sleep(0.05)
        started.stop()
        return sidecar.tasks()

    assert asyncio.run(scenario()) == ["ai_walk_forward", "ai_train", "ai_shadow", "ai_explain"] * 3


def test_the_backfill_asks_for_the_last_three_years_of_month_starts(
    desk: AgentDesk, sidecar: Sidecar
) -> None:
    desk.run()
    task, params = sidecar.calls[0]
    assert task == "ai_walk_forward" and len(params["retrain_dates"]) == agents.BACKFILL_MONTHS
    assert (params["lookback"], params["timesteps"], params["cost"]) == (
        agents.LOOKBACK,
        agents.TIMESTEPS,
        agents.COST,
    )


def test_a_second_run_on_the_same_day_does_nothing(desk: AgentDesk, sidecar: Sidecar) -> None:
    desk.run()
    sidecar.calls.clear()
    desk.run()
    assert sidecar.calls == []


def test_a_new_day_only_follows_the_agents_from_where_they_were(
    desk: AgentDesk, sidecar: Sidecar
) -> None:
    desk.run()
    sidecar.calls.clear()
    sidecar.last = LATEST + timedelta(days=1)
    history.cached_prices.cache_clear()
    desk.run()
    assert sidecar.tasks() == ["ai_shadow", "ai_explain"] * 3
    task, params = sidecar.calls[0]
    assert params["since"] == LATEST.isoformat() and params["target"] == [0.5, 0.3, 0.2]
    assert params["held"] == [0.3, 0.3, 0.4] and params["model_path"].endswith("ppo/v1.zip")
    assert desk.state.shadow["ppo"].since == sidecar.last


def test_a_new_month_trains_every_agent_again(
    desk: AgentDesk, sidecar: Sidecar, clock: list[date]
) -> None:
    desk.run()
    sidecar.calls.clear()
    clock[0] = date(2026, 12, 1)
    sidecar.last = date(2026, 11, 30)
    history.cached_prices.cache_clear()
    desk.run()
    assert sidecar.tasks() == ["ai_train", "ai_shadow", "ai_shadow", "ai_explain"] * 3
    assert all(model.version == 1 for model in desk.state.models.values())
    challengers = desk.state.challengers.values()
    assert [(c.model.version, c.model.trained) for c in challengers] == [(2, date(2026, 12, 1))] * 3


def test_a_retrain_request_trains_that_agent_at_the_next_run_and_is_then_cleared(
    desk: AgentDesk, sidecar: Sidecar
) -> None:
    desk.run()
    sidecar.calls.clear()
    assert desk.request_retrain("ai-sac").startswith("scheduled")
    assert desk.due("sac") and not desk.due("ppo")
    desk.run()
    assert [params["algorithm"] for task, params in sidecar.calls if task == "ai_train"] == ["sac"]
    assert desk.state.retrain == []


def test_the_ensemble_asks_for_every_agent_and_other_names_are_not_applicable(
    desk: AgentDesk,
) -> None:
    assert desk.request_retrain("ai-ensemble").startswith("scheduled")
    assert desk.state.retrain == ["a2c", "ppo", "sac"]
    assert desk.request_retrain("halal-etf-trend") == "not applicable: not an agent"


def test_the_agents_and_their_ensemble_are_offered_to_the_drift_monitor(desk: AgentDesk) -> None:
    assert desk.streams() == []
    desk.run()
    streams = desk.streams()
    assert [(stream.name, stream.kind) for stream in streams] == [
        ("ai-ppo", "agent"),
        ("ai-sac", "agent"),
        ("ai-a2c", "agent"),
        ("ai-ensemble", "ensemble"),
    ]
    assert all(len(stream.returns) == 100 for stream in streams)


def test_the_ensemble_waits_until_every_agent_has_a_record(
    desk: AgentDesk, sidecar: Sidecar
) -> None:
    sidecar.fail = {("ai_walk_forward", "a2c")}
    desk.run()
    assert [stream.name for stream in desk.streams()] == ["ai-ppo", "ai-sac"]
    assert desk.status().ensemble == {}


def test_a_failing_job_is_logged_and_the_other_agents_carry_on(
    desk: AgentDesk, sidecar: Sidecar, caplog: pytest.LogCaptureFixture
) -> None:
    sidecar.fail = {("ai_train", "sac")}
    desk.run()
    assert "Agent sac could not be brought up to date: ai_train sac: boom" in caplog.text
    assert set(desk.state.models) == {"ppo", "a2c"}
    assert not desk.current("sac", LATEST) and desk.current("ppo", LATEST)
    assert desk.busy is None


def test_training_is_written_to_the_audit_log_with_what_it_was_trained_on(
    desk: AgentDesk, tmp_path: Path
) -> None:
    desk.run()
    entries = [e for e in audit.AuditLog(tmp_path / "audit.jsonl").entries() if e.kind == "ai"]
    assert len(entries) == 3 and entries[0].payload["event"] == "trained"
    assert entries[0].payload["model"]["algorithm"] == "ppo" and entries[0].payload["manifest"]


def test_what_was_learned_survives_a_restart(
    desk: AgentDesk, tmp_path: Path, sidecar: Sidecar
) -> None:
    desk.run()
    again = AgentDesk(STOCKS, CRYPTO, None, tmp_path / "ai", sidecar, lambda: TODAY)
    assert set(again.state.models) == set(agents.ALGORITHMS)
    assert again.state.shadow["ppo"].target == [0.5, 0.3, 0.2]


def test_a_run_already_in_progress_is_left_alone(desk: AgentDesk, sidecar: Sidecar) -> None:
    desk.lock.acquire()
    desk.run()
    assert sidecar.calls == []


def test_the_status_reports_each_agent_its_holdings_and_the_ensemble_weights(
    desk: AgentDesk,
) -> None:
    desk.run()
    status = desk.status()
    ppo = status.agents[0]
    assert ppo.model is not None and ppo.target == {"HLAL": 0.5, "SPUS": 0.3, "cash": 0.2}
    assert ppo.days == 100 and ppo.last == LATEST and ppo.sharpe is not None
    assert (
        sum(status.ensemble.values()) == pytest.approx(1.0) and status.ensemble_sharpe is not None
    )


def logged(tmp_path: Path) -> list[audit.Entry]:
    return [e for e in audit.AuditLog(tmp_path / "audit.jsonl").entries() if e.kind == "xai"]


def test_each_decision_is_explained_by_its_model_and_written_to_the_audit_log(
    desk: AgentDesk, tmp_path: Path
) -> None:
    desk.run()
    (entry,) = logged(tmp_path)
    assert list(entry.payload["agents"]) == ["ppo", "sac", "a2c"]
    ppo = entry.payload["agents"]["ppo"]
    assert ppo["model"]["version"] == 1 and ppo["assets"] == ["HLAL", "SPUS", "cash"]
    assert ppo["overall"]["days"] == agents.OVERALL_SAMPLE
    (decision,) = ppo["decisions"]
    assert decision["decided"] == str(LATEST)
    assert decision["weights"] == {"HLAL": 0.5, "SPUS": 0.3, "cash": 0.2}
    assert decision["values"]["macd"] == {"HLAL": 0.007692, "SPUS": 0.0, "cash": -0.007692}
    assert decision["inputs"]["held"] == {"HLAL": 0.1, "SPUS": 0.1, "cash": 0.1}
    assert decision["inputs"]["ret_1"] == {"HLAL": 0.1, "SPUS": 0.1}
    assert entry.payload["unexplained"] == [] and entry.payload["manifest"]["data"]["data"]["rows"]
    assert audit.AuditLog(tmp_path / "audit.jsonl").verify().valid
    assert all(model.explained for model in desk.state.models.values())
    assert all(
        item.pending == [] and item.explained == LATEST for item in desk.state.shadow.values()
    )


def test_a_new_day_explains_the_new_decisions_and_not_the_model_again(
    desk: AgentDesk, sidecar: Sidecar, tmp_path: Path
) -> None:
    desk.run()
    sidecar.last = LATEST + timedelta(days=1)
    history.cached_prices.cache_clear()
    desk.run()
    first, second = logged(tmp_path)
    assert second.payload["agents"]["ppo"]["overall"] is None
    assert [item["decided"] for item in second.payload["agents"]["ppo"]["decisions"]] == [
        str(sidecar.last)
    ]
    assert first.payload["agents"]["ppo"]["overall"] is not None


def test_a_failed_explanation_waits_and_the_next_run_explains_without_following_again(
    desk: AgentDesk, sidecar: Sidecar, tmp_path: Path
) -> None:
    sidecar.fail = {("ai_explain", "ppo")}
    desk.run()
    (first,) = logged(tmp_path)
    assert list(first.payload["agents"]) == ["sac", "a2c"]
    assert desk.status().agents[0].unexplained == 1 and not desk.current("ppo", LATEST)
    sidecar.fail, sidecar.calls = set(), []
    desk.run()
    assert sidecar.tasks() == ["ai_explain"]
    _, second = logged(tmp_path)
    assert list(second.payload["agents"]) == ["ppo"]
    assert desk.status().agents[0].unexplained == 0 and desk.current("ppo", LATEST)


def test_a_retrained_agent_challenges_the_champion_instead_of_replacing_it(
    desk: AgentDesk, sidecar: Sidecar, tmp_path: Path
) -> None:
    sidecar.fail = {("ai_explain", "ppo")}
    desk.run()
    sidecar.fail = set()
    desk.request_retrain("ai-ppo")
    desk.run()
    last = logged(tmp_path)[-1]
    assert last.payload["agents"]["ppo"]["model"]["version"] == 1
    assert len(last.payload["agents"]["ppo"]["decisions"]) == 2
    assert last.payload["unexplained"] == []
    assert desk.state.challengers["ppo"].model.version == 2
    assert desk.status().agents[0].challenger is not None


def challenge(desk: AgentDesk, edge: float) -> None:
    """A challenger that followed the champion's last 30 sessions, `edge` a day better."""
    champion = desk.state.shadow["ppo"]
    days = champion.dates[-agents.CHALLENGE_DAYS :]
    varied = [0.002 * ((i % 5) - 1.5) for i in range(agents.CHALLENGE_DAYS)]
    champion.returns[-agents.CHALLENGE_DAYS :] = varied
    earned = [value + edge for value in varied]
    model = desk.state.models["ppo"].model_copy(update={"version": 2})
    desk.state.challengers["ppo"] = agents.Challenger(
        model=model,
        artifact="v2.zip",
        shadow=agents.Shadow(assets=champion.assets, dates=days, returns=earned),
    )


def test_a_challenger_at_least_as_good_as_the_champion_takes_over(
    desk: AgentDesk, sidecar: Sidecar, tmp_path: Path
) -> None:
    desk.run()
    challenge(desk, 0.001)
    assert desk.judge("ppo") is True
    assert desk.state.models["ppo"].version == 2 and "ppo" not in desk.state.challengers
    assert desk.log is not None
    event = [e for e in desk.log.entries() if e.payload.get("event") == "challenge"][-1]
    assert event.payload["promoted"] and "over 30 sessions" in event.payload["verdict"]
    waiting = agents.Queued(date=LATEST, version=1, observation=[0.0])
    desk.state.shadow["ppo"].pending = [waiting]
    desk.run()
    assert logged(tmp_path)[-1].payload["unexplained"] == [
        {
            "agent": "ppo",
            "decided": str(LATEST),
            "reason": "the model that made it was replaced before it was explained",
        }
    ]


def test_a_challenger_worse_than_the_champion_is_archived(
    desk: AgentDesk, sidecar: Sidecar, tmp_path: Path
) -> None:
    desk.run()
    challenge(desk, -0.001)
    assert desk.judge("ppo") is False
    assert desk.state.models["ppo"].version == 1 and "ppo" not in desk.state.challengers
    assert desk.log is not None
    rejected = [e for e in desk.log.entries() if e.payload.get("event") == "rejected"][-1]
    assert "Lost to the champion" in rejected.payload["reason"]


def test_a_challenger_is_judged_only_after_thirty_sessions(desk: AgentDesk) -> None:
    desk.run()
    challenge(desk, 0.001)
    desk.state.challengers["ppo"].shadow.dates.pop()
    desk.state.challengers["ppo"].shadow.returns.pop()
    assert desk.judge("ppo") is None and "ppo" in desk.state.challengers


def test_a_model_and_a_decision_from_before_the_explanations_are_explained_at_the_next_run(
    desk: AgentDesk, sidecar: Sidecar, tmp_path: Path
) -> None:
    desk.run()
    for name in agents.ALGORITHMS:
        desk.state.models[name].explained = False
        desk.state.shadow[name].explained = None
    sidecar.calls.clear()
    desk.run()
    assert sidecar.tasks() == ["ai_shadow", "ai_explain"] * 3
    assert sidecar.calls[0][1]["since"] == LATEST.isoformat()
    latest = logged(tmp_path)[-1].payload["agents"]["ppo"]
    assert latest["overall"] is not None and [item["decided"] for item in latest["decisions"]] == [
        str(LATEST)
    ]


def test_a_desk_without_an_audit_log_explains_nothing(
    tmp_path: Path, sidecar: Sidecar, desk: AgentDesk
) -> None:
    quiet = AgentDesk(STOCKS, CRYPTO, None, tmp_path / "quiet", sidecar, lambda: TODAY)
    quiet.enabled = True
    quiet.run()
    assert "ai_explain" not in sidecar.tasks() and quiet.current("ppo", LATEST)
    assert quiet.explanations_of("ppo", 5).decisions == []


def test_the_explanations_are_read_back_newest_first_with_what_the_model_relies_on(
    desk: AgentDesk, sidecar: Sidecar
) -> None:
    desk.run()
    sidecar.last = LATEST + timedelta(days=1)
    history.cached_prices.cache_clear()
    desk.run()
    found = desk.explanations_of("ppo", 10)
    assert [item.decided for item in found.decisions] == [sidecar.last, LATEST]
    assert found.version == 1 and found.assets == ["HLAL", "SPUS", "cash"]
    assert found.overall is not None and found.overall.days == agents.OVERALL_SAMPLE
    assert found.across["macd"]["HLAL"] == pytest.approx(0.1 / 13, abs=1e-5)
    assert [item.decided for item in desk.explanations_of("ppo", 1).decisions] == [sidecar.last]


def test_a_model_without_its_overall_gets_it_at_the_next_run_without_following_again(
    desk: AgentDesk, sidecar: Sidecar, tmp_path: Path
) -> None:
    desk.run()
    desk.state.models["sac"].explained = False
    sidecar.calls.clear()
    assert not desk.current("sac", LATEST) and desk.current("ppo", LATEST)
    desk.run()
    assert sidecar.tasks() == ["ai_explain"]
    assert sidecar.calls[0][1]["algorithm"] == "sac"
    assert sidecar.calls[0][1]["sample"] == agents.OVERALL_SAMPLE
    assert sidecar.calls[0][1]["observations"] == []
    latest = logged(tmp_path)[-1].payload["agents"]["sac"]
    assert latest["overall"] is not None and latest["decisions"] == []


def test_after_days_away_every_decision_is_explained_in_order_and_the_last_is_marked(
    desk: AgentDesk, sidecar: Sidecar, tmp_path: Path
) -> None:
    desk.run()
    sidecar.last = LATEST + timedelta(days=4)
    history.cached_prices.cache_clear()
    desk.run()
    days = [item["decided"] for item in logged(tmp_path)[-1].payload["agents"]["ppo"]["decisions"]]
    assert days == sorted(days) and len(days) == 4 and days[-1] == str(sidecar.last)
    assert desk.state.shadow["ppo"].explained == sidecar.last


def test_a_decision_from_before_the_explanations_is_explained_though_its_model_already_is(
    desk: AgentDesk, sidecar: Sidecar, tmp_path: Path
) -> None:
    desk.run()
    desk.state.shadow["ppo"].explained = None
    sidecar.calls.clear()
    assert not desk.current("ppo", LATEST)
    desk.run()
    assert sidecar.tasks() == ["ai_shadow", "ai_explain"]
    assert sidecar.calls[1][1]["sample"] == 0
    latest = logged(tmp_path)[-1].payload["agents"]["ppo"]
    assert latest["overall"] is None and [item["decided"] for item in latest["decisions"]] == [
        str(LATEST)
    ]


def test_the_sidecar_is_given_absolute_paths_whatever_folder_the_desk_keeps_its_files_in(
    tmp_path: Path, sidecar: Sidecar, desk: AgentDesk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    relative = AgentDesk(
        STOCKS, CRYPTO, audit.AuditLog(tmp_path / "rel.jsonl"), Path("data/ai"), sidecar
    )
    relative.enabled = True
    relative.run()
    asked = [params for task, params in sidecar.calls if "model_path" in params]
    assert {task for task, params in sidecar.calls if "model_path" in params} == {
        "ai_train",
        "ai_shadow",
        "ai_explain",
    }
    assert all(Path(params["model_path"]).is_absolute() for params in asked)
    assert all(Path(params["scaler_path"]).is_absolute() for params in asked)
    assert asked[0]["model_path"] == str(tmp_path / "data/ai/models/ppo/v1.zip")
    assert (tmp_path / "data/ai/models/ppo/v1.zip").exists()


def test_an_agent_whose_files_are_gone_is_trained_again(
    desk: AgentDesk, sidecar: Sidecar, tmp_path: Path
) -> None:
    desk.run()
    assert not desk.due("ppo")
    (tmp_path / "ai/models/ppo/v1.zip").unlink()
    assert desk.due("ppo") and not desk.due("sac")
    sidecar.calls.clear()
    desk.run()
    assert [params["algorithm"] for task, params in sidecar.calls if task == "ai_train"] == ["ppo"]
    assert desk.state.models["ppo"].version == 2 and not desk.due("ppo")


def test_the_ensemble_target_mixes_each_agents_target_by_the_ensembles_weights(
    desk: AgentDesk,
) -> None:
    assert desk.ensemble_target() is None and desk.ensemble_returns() is None
    desk.run()
    blend = desk._blend()
    assert blend is not None
    target = desk.ensemble_target()
    assert target is not None and set(target) == {"SPUS", "HLAL"}
    expected = 0.0
    for name, share in blend.latest.items():
        shadow = desk.state.shadow[name]
        assert shadow.target is not None
        expected += share * shadow.target[shadow.assets.index("SPUS")]
    assert target["SPUS"] == pytest.approx(expected)
    assert sum(target.values()) <= 1 + 1e-9
    returns = desk.ensemble_returns()
    assert returns is not None and len(returns) == 100


def test_the_ensemble_is_eligible_only_when_the_latest_evaluation_says_so(
    desk: AgentDesk, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace as NS

    from app.features.systematic import agent_lab

    assert desk.ensemble_eligible() is False

    def judged(eligible: bool) -> object:
        group = NS(name="UCB1 ensemble", verdict=NS(eligible=eligible))
        return NS(groups=[NS(name="PPO agent", verdict=NS(eligible=True)), group])

    monkeypatch.setattr(agent_lab, "latest", lambda log: judged(True))
    assert desk.ensemble_eligible() is True
    monkeypatch.setattr(agent_lab, "latest", lambda log: judged(False))
    assert desk.ensemble_eligible() is False


class Mood:
    def mood(self, sessions: Sequence[date]) -> pd.DataFrame:
        index = pd.DatetimeIndex([pd.Timestamp(day) for day in sessions])
        values = [0.5 if day.day % 2 else float("nan") for day in sessions]
        return pd.DataFrame(
            {"sentiment": values, "news_risk": [3.0] * len(sessions), "headlines": 1},
            index=index,
        )


def test_the_market_mood_is_added_to_every_fund_on_its_day() -> None:
    data = pd.DataFrame(
        {
            "date": ["2026-11-05", "2026-11-05", "2026-11-06", "2026-11-06"],
            "tic": ["HLAL", "SPUS", "HLAL", "SPUS"],
            "close": [1.0, 2.0, 1.1, 2.1],
        }
    )
    found = agents.with_mood(data, Mood())
    assert found["sentiment"].tolist()[:2] == [0.5, 0.5]
    assert found["sentiment"].iloc[2:].isna().all()
    assert set(found.columns) == {"date", "tic", "close", "sentiment", "news_risk"}
    assert agents.with_mood(data, None) is data


def test_the_versions_running_are_registered_with_their_weights_at_start(
    desk: AgentDesk, tmp_path: Path
) -> None:
    desk.run()
    published: list[tuple[str, int, str]] = []

    class Registry:
        def publish(self, model: Any, artifact: str, *rest: Any) -> None:
            published.append((model.algorithm, model.version, artifact))

        def production(self, algorithm: str) -> None:
            return None

    desk.registry = Registry()  # type: ignore[assignment]
    for name in ("ppo", "sac"):
        weights = Path(desk._paths(name)["model_path"])
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(name.encode())
    published.clear()
    desk.register_current()
    assert {name for name, _, _ in published} == {"ppo", "sac"}
    assert all(version == 1 for _, version, _ in published)
