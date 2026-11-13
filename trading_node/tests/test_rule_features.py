from datetime import UTC, datetime

import pandas as pd

from app.core.feature_store import ANOMALOUS, FeatureValue
from app.features.systematic import signals
from app.features.systematic.catalog import STRATEGIES
from trading_node.strategies.rule import RuleStrategy

ROTATION = STRATEGIES["halal-etf-rotation"]


def closes() -> pd.DataFrame:
    days = pd.date_range("2026-01-01", periods=200, tz="UTC")
    return pd.DataFrame(
        {
            "SPUS": [100 + i * 0.3 for i in range(200)],
            "HLAL": [100 + i * 0.1 for i in range(200)],
            "UMMA": [100 - i * 0.05 for i in range(200)],
            "SPSK": [100 + i * 0.01 for i in range(200)],
        },
        index=days,
    )


def reader(values: dict[str, float], flags: int = 0, session: datetime | None = None):  # type: ignore[no-untyped-def]
    def read(name: str, symbol: str) -> FeatureValue | None:
        if symbol not in values:
            return None
        when = session or datetime(2026, 7, 19, tzinfo=UTC)
        return FeatureValue(
            name=name, symbol=symbol, value=values[symbol], time=when, quality_flags=flags
        )

    return read


def gains(frame: pd.DataFrame) -> dict[str, float]:
    return {str(s): float(frame[s].pct_change(126).iloc[-1]) for s in frame}


def test_the_momentum_rule_decides_from_the_online_features_exactly_as_from_closes() -> None:
    frame = closes()
    rule = RuleStrategy(ROTATION, "live", "v", 0.5, reader(gains(frame)))
    found = rule.from_features(frame)
    expected = signals.weights(frame, ROTATION).iloc[-1]
    assert found is not None and rule.source == "features"
    assert found.to_dict() == expected.to_dict()


def test_a_flagged_missing_or_old_feature_falls_back_to_the_closes() -> None:
    frame = closes()
    values = gains(frame)
    assert (
        RuleStrategy(ROTATION, "live", "v", 0.5, reader(values, ANOMALOUS)).from_features(frame)
        is None
    )
    partial = {k: v for k, v in values.items() if k != "UMMA"}
    assert RuleStrategy(ROTATION, "live", "v", 0.5, reader(partial)).from_features(frame) is None
    old = reader(values, session=datetime(2026, 7, 1, tzinfo=UTC))
    assert RuleStrategy(ROTATION, "live", "v", 0.5, old).from_features(frame) is None
    trend = RuleStrategy(STRATEGIES["halal-etf-trend"], "live", "v", 0.5, reader(values))
    assert trend.from_features(frame) is None


def test_a_momentum_rule_of_another_window_never_reads_the_126_day_feature() -> None:
    frame = closes()
    faster = ROTATION.model_copy(update={"window": 90})
    rule = RuleStrategy(faster, "live", "v", 0.5, reader(gains(frame)))
    assert rule.from_features(frame) is None
