from datetime import UTC, datetime, timedelta

import fakeredis
import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from app.core import feature_store
from app.core.feature_store import (
    FEATURES,
    LOCKED,
    FeatureStore,
    FeatureValue,
    as_of,
    key,
    lineage,
    read_online,
)
from app.core.tables import FEATURE_TABLES
from app.features.models.features import FeaturePipeline
from app.features.systematic.catalog import TEMPLATES

NOW = datetime(2026, 11, 9, 2, tzinfo=UTC)


def market() -> pd.DataFrame:
    days = pd.date_range(end=NOW.date(), periods=300, tz="UTC")
    rng = np.random.default_rng(5)
    walks = {
        symbol: 100 * np.cumprod(1 + rng.normal(drift, 0.01, 300))
        for symbol, drift in (("SPUS", 0.0006), ("HLAL", 0.0002))
    }
    return pd.DataFrame(walks, index=days)


def pipeline(online: object, closes: pd.DataFrame) -> FeaturePipeline:
    return FeaturePipeline(
        FeatureStore(online),
        symbols=lambda: list(closes),
        closes=lambda names: closes[names],
        sentiment=lambda symbol, now: None,
        books=dict,
    )


@pytest.mark.parametrize("feature", FEATURES, ids=lambda feature: feature.ref)
def test_a_feature_whose_code_changed_must_be_a_new_version(feature: feature_store.Feature) -> None:
    assert lineage(feature) == LOCKED[feature.ref], (
        f"{feature.ref} computes differently now: bump its version and lock the new lineage"
    )


def test_online_offline_and_backtest_agree_for_the_same_symbol_and_session(
    database: Engine,
) -> None:
    closes = market()
    online = fakeredis.FakeRedis()
    pipeline(online, closes).run(NOW)
    finished = closes.loc[closes.index.date < NOW.date()]
    gains = finished.pct_change(TEMPLATES["halal-etf-rotation"].window).iloc[-1]
    with database.connect() as connection:
        for symbol in closes:
            served = read_online(online, "momentum_126", symbol)
            stored = as_of(connection, "momentum_126", symbol, datetime.now(UTC))
            assert served is not None and stored is not None
            assert served.value == stored.value == gains[symbol]
            assert served.time == stored.time
            volatility = read_online(online, "volatility_63", symbol)
            assert volatility is not None
            expected = finished[symbol].pct_change().iloc[-63:].std() * np.sqrt(252)
            assert volatility.value == pytest.approx(expected, rel=1e-12)


def test_a_value_whose_transaction_failed_never_goes_online(database: Engine) -> None:
    online = fakeredis.FakeRedis()
    broken = pipeline(online, market())

    def fail(moment: datetime, names: list[str]) -> int:
        with database.begin() as connection:
            broken.store.write(connection, "momentum_126", "SPUS", 0.1, NOW)
            raise RuntimeError("the database went away")

    broken.compute = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        broken.run(NOW)
    assert broken.store.flush() == 0
    assert online.keys("feature:*") == []


def test_a_backtest_sees_a_value_only_once_it_was_known(database: Engine) -> None:
    store = FeatureStore(fakeredis.FakeRedis())
    with database.begin() as connection:
        store.write(connection, "momentum_126", "SPUS", 0.1, NOW - timedelta(days=1))
        known: datetime = connection.execute(
            sa.select(FEATURE_TABLES["alpha_factors"].c.knowledge_ts)
        ).scalar_one()
        assert as_of(connection, "momentum_126", "SPUS", known - timedelta(seconds=1)) is None
        found = as_of(connection, "momentum_126", "SPUS", known)
        assert found is not None and found.value == 0.1


def test_a_lost_online_store_is_rebuilt_from_the_offline_one(database: Engine) -> None:
    online = fakeredis.FakeRedis()
    store = FeatureStore(online)
    with database.begin() as connection:
        store.write(connection, "momentum_126", "SPUS", 0.1, NOW)
        store.write(connection, "obi_10bps", "BTC/USD", 0.2, NOW)
    store.flush()
    online.flushall()
    with database.begin() as connection:
        assert store.resync(connection, datetime.now(UTC)) == 2
        later = datetime.now(UTC) + timedelta(hours=1)
        assert store.resync(connection, later) == 1
    found = read_online(online, "momentum_126", "SPUS")
    assert found is not None and found.value == 0.1


def test_a_value_from_other_code_is_refused_online() -> None:
    online = fakeredis.FakeRedis()
    feature = feature_store.BY_NAME["momentum_126"]
    forged = FeatureValue(
        name="momentum_126", symbol="SPUS", value=0.5, time=NOW, version=1, lineage="old"
    )
    online.set(key("momentum_126", "SPUS"), forged.model_dump_json())
    assert read_online(online, "momentum_126", "SPUS") is None
    honest = forged.model_copy(update={"lineage": lineage(feature)})
    online.set(key("momentum_126", "SPUS"), honest.model_dump_json())
    assert read_online(online, "momentum_126", "SPUS") is not None
