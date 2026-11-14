from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import fakeredis
import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine

from app.core import breakers, feature_store, registry
from app.core.feature_store import ANOMALOUS, FeatureStore, read_online
from app.core.tables import FEATURE_TABLES, model_drift
from app.features.models import drift
from app.features.models.features import FeaturePipeline

NOW = datetime(2026, 11, 9, 2, tzinfo=UTC)


def put_model(connection: sa.Connection, version: int, folder: Path, passed: bool = True) -> None:
    weights = folder / f"v{version}.zip"
    weights.write_bytes(f"weights {version}".encode())
    registry.register(
        connection,
        f"ppo-v{version}",
        "ppo",
        "rl_agent",
        "2023-10-30..2026-11-03",
        {"seed": 1},
        str(weights),
        {
            "in_sample_sharpe": 1.0 + version,
            "gates": [{"name": "Learned from its training window", "passed": passed}],
        },
        strategy="ai-ensemble",
        feature_version={"agent_features": "abc"},
    )


def test_promotion_archives_the_champion_and_rollback_brings_it_back(
    database: Engine, tmp_path: Path
) -> None:
    with database.begin() as connection:
        put_model(connection, 1, tmp_path)
        assert registry.get(connection, "ppo-v1").deployment_stage == "staging"
        registry.promote(connection, "ppo-v1", "gate: first version", "No champion")
        put_model(connection, 2, tmp_path)
        promoted = registry.promote(connection, "ppo-v2", "gate: 30-session shadow", "Won")
        assert (promoted.promoted_by, promoted.promotion_reason) == (
            "gate: 30-session shadow",
            "Won",
        )
        assert registry.get(connection, "ppo-v1").deployment_stage == "archived"
        back = registry.rollback(connection, "ppo", "manual", "v2 trades badly")
        assert back.model_id == "ppo-v1" and back.deployment_stage == "production"
        assert registry.get(connection, "ppo-v2").deployment_stage == "rolled_back"
        assert registry.productions(connection) == {"ppo": "ppo-v1"}


def test_a_version_names_one_artifact_for_good(database: Engine, tmp_path: Path) -> None:
    with database.begin() as connection:
        put_model(connection, 1, tmp_path)
        first = registry.get(connection, "ppo-v1")
        assert first.artifact_hash is not None and len(first.artifact_hash) == 64
        put_model(connection, 1, tmp_path)
        assert registry.get(connection, "ppo-v1").created == first.created
        (tmp_path / "v1.zip").write_bytes(b"other weights")
        with pytest.raises(registry.ImmutableError):
            registry.register(
                connection, "ppo-v1", "ppo", "rl_agent", "x", {}, str(tmp_path / "v1.zip"), {}
            )


def test_the_hash_of_a_weights_folder_ignores_the_download_cache(tmp_path: Path) -> None:
    def folder(name: str, stamp: str) -> str:
        root = tmp_path / name
        (root / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (root / "model.safetensors").write_bytes(b"weights")
        (root / ".cache" / "huggingface" / "download" / "model.metadata").write_text(stamp)
        return str(root)

    assert registry.artifact_hash(folder("first", "1699")) == registry.artifact_hash(
        folder("second", "1700")
    )


def test_the_hash_of_a_weights_folder_changes_with_the_weights(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    before = registry.artifact_hash(str(tmp_path))
    (tmp_path / "model.safetensors").write_bytes(b"other weights")

    assert registry.artifact_hash(str(tmp_path)) != before


def test_a_candidate_that_failed_a_gate_cannot_be_promoted(
    database: Engine, tmp_path: Path
) -> None:
    with database.begin() as connection:
        put_model(connection, 1, tmp_path, passed=False)
        with pytest.raises(registry.GateError, match="Learned from its training window"):
            registry.promote(connection, "ppo-v1", "manual", "try anyway")


def test_models_are_promoted_and_rolled_back_through_the_api(
    client: TestClient, database: Engine, tmp_path: Path
) -> None:
    with database.begin() as connection:
        put_model(connection, 1, tmp_path)
        put_model(connection, 2, tmp_path)
        put_model(connection, 3, tmp_path, passed=False)
        registry.promote(connection, "ppo-v1", "gate", "first")
    body = {"reason": "v2 shadowed better"}
    assert client.post("/models/ppo-v2/promote", json=body).json()["promoted_by"] == "manual"
    refused = client.post("/models/ppo-v3/promote", json=body)
    assert refused.status_code == 422 and "Learned from" in refused.json()["detail"]
    back = client.post("/models/ppo/rollback", json={"reason": "v2 misbehaves"})
    assert back.json()["model_id"] == "ppo-v1"
    stages = {item["model_id"]: item["deployment_stage"] for item in client.get("/models").json()}
    assert stages == {"ppo-v1": "production", "ppo-v2": "rolled_back", "ppo-v3": "staging"}
    assert client.post("/models/nothing/promote", json=body).status_code == 404
    assert client.post("/models/nothing/rollback", json=body).status_code == 404


def test_a_feature_goes_to_both_stores_and_an_odd_value_is_flagged(database: Engine) -> None:
    online = fakeredis.FakeRedis()
    store = FeatureStore(online)
    with database.begin() as connection:
        store.sync(connection)
        store.write(connection, "momentum_126", "SPUS", 0.12, NOW)
        flagged = store.write(connection, "sentiment_24h", "SPUS", 3.0, NOW)
        rows = connection.execute(sa.select(FEATURE_TABLES["alpha_factors"])).all()
        assert read_online(online, "momentum_126", "SPUS") is None
    assert store.flush() == 2
    assert flagged.quality_flags == ANOMALOUS and not flagged.usable
    assert [(row.feature, row.symbol, row.value) for row in rows] == [
        ("momentum_126", "SPUS", 0.12)
    ]
    found = read_online(online, "momentum_126", "SPUS")
    assert found is not None and found.value == 0.12 and found.usable
    assert (
        online.ttl("feature:momentum_126:v1:entity:SPUS")
        == feature_store.BY_NAME["momentum_126"].ttl_seconds
    )


def test_a_dead_online_store_never_loses_the_offline_value(database: Engine) -> None:
    class Down:
        def set(self, *args: Any, **kwargs: Any) -> None:
            raise ConnectionError("no redis")

    store = FeatureStore(Down())
    with database.begin() as connection:
        store.write(connection, "obi_10bps", "BTC/USD", 0.3, NOW)
        count = connection.execute(
            sa.select(sa.func.count()).select_from(FEATURE_TABLES["obi_metrics"])
        ).scalar_one()
    store.flush()
    assert count == 1


def test_the_pipeline_computes_features_from_finished_sessions_only(database: Engine) -> None:
    days = pd.date_range(end=NOW.date(), periods=200, tz="UTC")
    closes = pd.DataFrame({"SPUS": np.linspace(50, 60, 200)}, index=days)
    online = fakeredis.FakeRedis()
    pipeline = FeaturePipeline(
        FeatureStore(online),
        symbols=lambda: ["SPUS"],
        closes=lambda names: closes,
        sentiment=lambda symbol, now: 0.4,
        books=dict,
    )

    assert pipeline.run(NOW) == 3
    momentum = read_online(online, "momentum_126", "SPUS")
    assert momentum is not None
    assert momentum.value == pytest.approx(feature_store.momentum(closes["SPUS"].iloc[:-1]))
    assert momentum.time.date() == (NOW - timedelta(days=1)).date()


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    assert drift.psi(base, rng.normal(0, 1, 5000)) < 0.02
    assert drift.psi(base, rng.normal(1.5, 1, 5000)) > drift.PSI_HIGH


def test_technical_drift_beyond_a_tenth_of_a_percent_is_high() -> None:
    assert drift.technical("a", 0.0005).drift_severity == "low"  # type: ignore[union-attr]
    assert drift.technical("a", 0.002).drift_severity == "high"  # type: ignore[union-attr]
    assert drift.technical("a", None) is None


def test_shap_drift_compares_recent_attribution_with_earlier(database: Engine) -> None:
    steady = {"values": {"ret_1": {"SPUS": 0.5}, "rsi": {"SPUS": 0.5}}}
    moved = {"values": {"ret_1": {"SPUS": 0.95}, "rsi": {"SPUS": 0.05}}}
    found = drift.shap_drift("ppo", [steady] * 15 + [moved] * 10)
    assert found is not None and found.drift_severity == "high"
    assert drift.shap_drift("ppo", [steady] * 5) is None


def test_the_nightly_job_records_and_halts_a_strategy_whose_replay_differs(
    database: Engine,
) -> None:
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.001, 0.01, 400))
    asked: list[str] = []
    sources = drift.DriftSources(
        deployed=lambda: ["halal-etf-trend"],
        replay_gap=lambda name: 0.02,
        returns=lambda name: (returns, 252),
        inputs=lambda name: returns,
        model_returns=dict,
        model_decisions=dict,
        adwin_changes=lambda: {"ppo": 2},
        retrain=asked.append,
        alert=lambda title, detail: None,
    )
    with database.begin() as connection:
        found = drift.nightly(connection, sources, -0.5, NOW)
        switches = breakers.switches(connection)
        rows = connection.execute(sa.select(model_drift.c.metric)).all()
        blocked = drift.blocked(connection, NOW)
    assert {row.metric for row in rows} == {
        "replay_divergence",
        "rolling_sharpe_30d",
        "psi",
        "ks",
        "adwin",
    }
    assert switches["strategy:halal-etf-trend"].engaged
    assert "code review" in (switches["strategy:halal-etf-trend"].reason or "")
    assert asked == ["ppo"]
    assert "halal-etf-trend" in blocked
    assert len(found) == len(rows)


def test_the_risk_actor_blocks_a_technical_failure_and_scales_a_statistical_one() -> None:
    from trading_node.actors.arbiter import screen
    from trading_node.arbiter import StrategyTarget

    targets = [
        StrategyTarget("halal-etf-trend", "live", 0.5, {"SPUS": 1.0}),
        StrategyTarget("ai-ensemble", "live", 1.0, {"HLAL": 0.2}),
        StrategyTarget("halal-etf-rotation", "live", 0.4, {"UMMA": 1.0}),
    ]
    kept, notes = screen(targets, {"sac": "PSI 0.4"}, {"halal-etf-rotation": 0.5})
    assert [(t.strategy, t.allocation) for t in kept] == [
        ("halal-etf-trend", 0.5),
        ("halal-etf-rotation", 0.2),
    ]
    assert notes == [
        "ai-ensemble blocked, technical failure: PSI 0.4",
        "halal-etf-rotation at 50% of its allocation, statistical failure",
    ]


def test_the_pipeline_takes_daily_closes_without_a_time_zone_as_alpaca_serves_them(
    database: Engine,
) -> None:
    days = pd.date_range(end=NOW.date(), periods=200)
    closes = pd.DataFrame({"SPUS": np.linspace(50, 60, 200)}, index=days)
    online = fakeredis.FakeRedis()
    pipeline = FeaturePipeline(
        FeatureStore(online),
        symbols=lambda: ["SPUS"],
        closes=lambda names: closes,
        sentiment=lambda symbol, now: None,
        books=dict,
    )

    assert pipeline.run(NOW) == 2
    found = read_online(online, "momentum_126", "SPUS")
    assert found is not None and found.time.date() == (NOW - timedelta(days=1)).date()
