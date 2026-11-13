from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine

from app.core.tables import strategy_health
from app.features.models import drift, health
from app.features.systematic.agents import Shadow, calibration_points

NOW = datetime(2026, 11, 9, 2, tzinfo=UTC)


def finding(
    subject: str, kind: drift.Kind, severity: drift.Severity, detail: str = "x"
) -> drift.DriftFinding:
    return drift.DriftFinding(
        subject=subject,
        kind=kind,
        metric="m",
        value=1.0,
        threshold=1.0,
        drift_severity=severity,
        detail=detail,
    )


def test_a_technical_failure_outranks_a_statistical_one_and_low_is_healthy() -> None:
    found = health.classify(
        [
            finding("rule", "statistical", "medium", "Sharpe decayed"),
            finding("rule", "technical", "high", "Replay differs"),
            finding("agent", "statistical", "medium", "Badly calibrated"),
            finding("calm", "statistical", "low"),
        ],
        NOW,
    )
    assert (found["rule"].failure_type, found["rule"].factor) == ("technical", 0.0)
    assert found["rule"].reason == "Replay differs"
    assert (found["agent"].failure_type, found["agent"].factor) == ("statistical", 0.5)
    assert (found["calm"].failure_type, found["calm"].factor) == ("none", 1.0)
    assert health.blocked(found) == {"rule": "Replay differs"}
    assert health.factors(found) == {"agent": 0.5}


def test_a_failure_keeps_its_start_while_it_lasts_and_an_old_state_stops_counting(
    database: Engine,
) -> None:
    with database.begin() as connection:
        health.write(connection, health.classify([finding("a", "statistical", "medium")], NOW))
        later = NOW + timedelta(days=1)
        health.write(connection, health.classify([finding("a", "statistical", "medium")], later))
        row = connection.execute(sa.select(strategy_health)).one()
        assert (row.since, row.updated) == (NOW, later)
        assert "a" in health.read(connection, later)
        assert health.read(connection, later + timedelta(days=3)) == {}
        health.write(connection, health.classify([finding("a", "statistical", "low")], later))
        assert connection.execute(sa.select(strategy_health.c.since)).scalar_one() == later


def test_the_sharpe_band_flags_decay_only_when_even_its_top_falls_short() -> None:
    rng = np.random.default_rng(3)
    weak = pd.Series(rng.normal(-0.001, 0.01, 400))
    found = {f.metric: f for f in drift.sharpe_band("rule", weak, 252, expected=2.5)}
    assert set(found) == {"sharpe_band_30", "sharpe_band_90", "sharpe_band_365"}
    assert found["sharpe_band_365"].drift_severity == "medium"
    assert found["sharpe_band_30"].drift_severity == "low"
    same = pd.Series(rng.normal(0.0008, 0.01, 400))
    assert all(f.drift_severity == "low" for f in drift.sharpe_band("rule", same, 252, 1.2))


def test_calibration_compares_stated_confidence_with_the_hit_rate() -> None:
    sure = drift.calibration("ppo", [0.9] * 40, [True, False] * 20)
    assert sure is not None and sure.drift_severity == "medium"
    honest = drift.calibration("ppo", [0.55] * 40, [True] * 22 + [False] * 18)
    assert honest is not None and honest.drift_severity == "low"
    assert drift.calibration("ppo", [0.9] * 10, [True] * 10) is None


def test_an_agents_move_is_scored_against_the_session_it_earns() -> None:
    days = pd.bdate_range("2026-11-02", periods=4)
    closes = pd.DataFrame({"SPUS": [100.0, 101.0, 100.0, 102.0]}, index=days)
    shadow = Shadow(
        assets=["SPUS"],
        dates=[d.date() for d in days[1:]],
        returns=[0.0, 0.0, 0.0],
        applied=[[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]],
        confidence={"2026-11-03": [0.8, 0.8], "2026-11-04": [0.7, 0.7]},
    )
    stated, correct = calibration_points(shadow, closes)
    assert stated == [0.8, 0.7]
    assert correct == [False, False]
    assert date(2026, 11, 4) in shadow.dates


def test_the_nightly_job_writes_each_subjects_health(database: Engine) -> None:
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.001, 0.01, 400))
    asked: list[str] = []
    sources = drift.DriftSources(
        deployed=lambda: ["halal-etf-trend", "halal-multi-trend"],
        replay_gap=lambda name: 0.02 if name == "halal-etf-trend" else 0.0,
        returns=lambda name: (returns, 252),
        inputs=lambda name: None,
        model_returns=dict,
        model_decisions=dict,
        adwin_changes=dict,
        retrain=asked.append,
        expected_sharpe=lambda name: 9.0,
        calibrations=lambda: {"ppo": ([0.9] * 40, [True, False] * 20)},
    )
    with database.begin() as connection:
        drift.nightly(connection, sources, -5.0, NOW)
        found = health.read(connection, NOW)
    assert found["halal-etf-trend"].failure_type == "technical"
    assert found["halal-multi-trend"].failure_type == "statistical"
    assert found["ppo"].failure_type == "statistical"
    assert sorted(asked) == ["halal-multi-trend", "ppo"]


def test_the_health_states_are_served(client: TestClient, database: Engine) -> None:
    now = datetime.now(UTC)
    with database.begin() as connection:
        health.write(
            connection, health.classify([finding("rule", "technical", "high", "Replay")], now)
        )
    [found] = client.get("/models/health").json()
    assert (found["subject"], found["failure_type"], found["factor"]) == ("rule", "technical", 0.0)
