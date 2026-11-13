import hashlib
import importlib
import inspect
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import distinct_on, insert
from sqlalchemy.engine import Connection

from app.core.tables import FEATURE_TABLES, features

logger = logging.getLogger(__name__)

GAP = 1
STALE = 2
ANOMALOUS = 4
MISSING = 8
FLAG_NAMES = {GAP: "gap", STALE: "stale", ANOMALOUS: "anomalous", MISSING: "missing"}
DAY = 24 * 3600


@dataclass(frozen=True)
class Feature:
    name: str
    version: int
    description: str
    computation: str
    ttl_seconds: int
    expected_min: float | None
    expected_max: float | None
    offline_table: str
    sources: tuple[str, ...]
    owner: str
    entity: str = "symbol"

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"


FEATURES = (
    Feature(
        "momentum_126",
        1,
        "Gain over the last 126 daily closes, as the dual momentum rule ranks it",
        "app.core.feature_store:momentum",
        2 * DAY,
        -0.95,
        20.0,
        "alpha_factors",
        ("alpaca daily bars, adjusted",),
        "systematic",
    ),
    Feature(
        "volatility_63",
        1,
        "Annualised volatility of the last 63 daily returns",
        "app.core.feature_store:volatility",
        2 * DAY,
        0.0,
        5.0,
        "alpha_factors",
        ("alpaca daily bars, adjusted",),
        "systematic",
    ),
    Feature(
        "sentiment_24h",
        2,
        "FinBERT score of the symbol's headlines over the last day, through the uncertainty gate",
        "app.features.scanner.mood:scored",
        DAY,
        -1.0,
        1.0,
        "sentiment_cache",
        ("alpaca news", "finbert"),
        "scanner",
    ),
    Feature(
        "obi_10bps",
        1,
        "Order book imbalance within 10 bps of the mid: +1 all bids, -1 all asks",
        "app.features.book.metrics:imbalance",
        600,
        -1.0,
        1.0,
        "obi_metrics",
        ("alpaca crypto order books",),
        "book",
    ),
)
BY_NAME = {feature.name: feature for feature in FEATURES}

LOCKED = {
    "momentum_126@v1": "97982e9f11f7cf01",
    "volatility_63@v1": "1742a1c372cbea5f",
    "sentiment_24h@v2": "feba892433a7a9fd",
    "obi_10bps@v1": "d797fbf274af7f3b",
}


def lineage(feature: Feature) -> str:
    """A fingerprint of what a version computes: its identity, sources and the code that
    computes it. Code that changes without a version bump changes it, and CI refuses that."""
    module, _, function = feature.computation.partition(":")
    code = inspect.getsource(getattr(importlib.import_module(module), function))
    identity = json.dumps([feature.ref, feature.sources, feature.entity, code])
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


class FeatureValue(BaseModel):
    name: str
    symbol: str
    value: float
    time: datetime
    quality_flags: int = 0
    version: int = 1
    lineage: str = ""

    @property
    def usable(self) -> bool:
        """A value with any quality flag is treated as missing, not as data."""
        return self.quality_flags == 0


def key(name: str, symbol: str, version: int | None = None) -> str:
    found = BY_NAME[name].version if version is None else version
    return f"feature:{name}:v{found}:entity:{symbol}"


def flags_of(feature: Feature, value: float) -> int:
    """The quality flags a value earns on its own: missing when not a number, anomalous when
    outside the feature's expected range."""
    if value is None or not np.isfinite(value):
        return MISSING
    low, high = feature.expected_min, feature.expected_max
    if (low is not None and value < low) or (high is not None and value > high):
        return ANOMALOUS
    return 0


def explain(flags: int) -> list[str]:
    return [name for bit, name in FLAG_NAMES.items() if flags & bit]


def momentum(closes: pd.Series, window: int = 126) -> float:
    """The same gain the dual momentum rule ranks: the last close over the one `window` back."""
    return float(closes.pct_change(window).iloc[-1])


def volatility(closes: pd.Series, window: int = 63, periods: int = 252) -> float:
    return float(closes.pct_change().iloc[-window:].std() * np.sqrt(periods))


class FeatureStore:
    """Writes each feature once, to Postgres for backtests and to Redis for live decisions.

    The offline row is written in the caller's transaction; the online value only once that
    transaction has committed (`flush`), so Redis never holds a value Postgres lost, and
    `resync` rebuilds Redis from Postgres when Redis lost one.
    """

    def __init__(self, online: Any) -> None:
        self.online = online
        self.pending: list[FeatureValue] = []

    def sync(self, connection: Connection) -> None:
        """Keep the registry table in step with the features defined here."""
        for feature in FEATURES:
            values = {
                "description": feature.description,
                "computation": feature.computation,
                "ttl_seconds": feature.ttl_seconds,
                "expected_min": feature.expected_min,
                "expected_max": feature.expected_max,
                "offline_table": feature.offline_table,
                "entity": feature.entity,
                "sources": list(feature.sources),
                "owner": feature.owner,
                "lineage": lineage(feature),
            }
            statement = insert(features).values(
                name=feature.name, version=feature.version, **values
            )
            connection.execute(
                statement.on_conflict_do_update(
                    index_elements=[features.c.name, features.c.version],
                    set_={**values, "updated": sa.func.now()},
                )
            )

    def write(
        self,
        connection: Connection,
        name: str,
        symbol: str,
        value: float,
        time: datetime | None = None,
        flags: int = 0,
    ) -> FeatureValue:
        feature = BY_NAME[name]
        found = FeatureValue(
            name=name,
            symbol=symbol,
            value=float(value) if value is not None and np.isfinite(value) else 0.0,
            time=time or datetime.now(UTC),
            quality_flags=flags | flags_of(feature, value),
            version=feature.version,
            lineage=lineage(feature),
        )
        table = FEATURE_TABLES[feature.offline_table]
        connection.execute(
            insert(table)
            .values(
                feature=name,
                version=feature.version,
                symbol=symbol,
                time=found.time,
                value=found.value,
                quality_flags=found.quality_flags,
            )
            .on_conflict_do_nothing()
        )
        self.pending.append(found)
        return found

    def flush(self) -> int:
        """Send the values written since the last flush to the online store, once committed."""
        sent, self.pending = self.pending, []
        for found in sent:
            self.publish(found, BY_NAME[found.name].ttl_seconds)
        return len(sent)

    def discard(self) -> None:
        self.pending = []

    def publish(self, found: FeatureValue, ttl: int) -> None:
        name = key(found.name, found.symbol, found.version)
        try:
            self.online.set(name, found.model_dump_json(), ex=max(1, ttl))
        except Exception as exc:
            logger.warning("The online feature store did not take %s: %s", name, exc)

    def resync(self, connection: Connection, now: datetime | None = None) -> int:
        """Put each feature's latest offline value back online while it is still fresh."""
        moment = now or datetime.now(UTC)
        sent = 0
        for feature in FEATURES:
            table = FEATURE_TABLES[feature.offline_table]
            rows = connection.execute(
                sa.select(table)
                .where(table.c.feature == feature.name, table.c.version == feature.version)
                .ext(distinct_on(table.c.symbol))
                .order_by(table.c.symbol, table.c.time.desc())
            )
            for row in rows:
                left = feature.ttl_seconds - (moment - row.knowledge_ts).total_seconds()
                if left <= 0:
                    continue
                found = FeatureValue(
                    name=feature.name,
                    symbol=row.symbol,
                    value=row.value,
                    time=row.time,
                    quality_flags=row.quality_flags,
                    version=feature.version,
                    lineage=lineage(feature),
                )
                self.publish(found, int(left))
                sent += 1
        return sent


def read_online(client: Any, name: str, symbol: str) -> FeatureValue | None:
    """The latest value of the registered version of a feature from Redis, or None once its
    time to live has passed or when it was computed by other code than this version's."""
    feature = BY_NAME[name]
    raw = client.get(key(name, symbol, feature.version))
    if raw is None:
        return None
    found = FeatureValue.model_validate(json.loads(raw))
    if found.version != feature.version or found.lineage != lineage(feature):
        logger.warning("Ignored %s for %s: it came from other code", feature.ref, symbol)
        return None
    return found


def as_of(
    connection: Connection, name: str, symbol: str, decision: datetime
) -> FeatureValue | None:
    """The value a decision at `decision` could have seen: known by then, not merely stamped
    by then, so a backtest reading this never sees the future."""
    feature = BY_NAME[name]
    table = FEATURE_TABLES[feature.offline_table]
    row = connection.execute(
        sa.select(table)
        .where(
            table.c.feature == name,
            table.c.version == feature.version,
            table.c.symbol == symbol,
            table.c.knowledge_ts <= decision,
            table.c.time <= decision,
        )
        .order_by(table.c.time.desc(), table.c.knowledge_ts.desc())
        .limit(1)
    ).one_or_none()
    if row is None:
        return None
    return FeatureValue(
        name=name,
        symbol=symbol,
        value=row.value,
        time=row.time,
        quality_flags=row.quality_flags,
        version=feature.version,
        lineage=lineage(feature),
    )
