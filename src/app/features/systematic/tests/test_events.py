import numpy as np
import pandas as pd
import pytest

from app.features.systematic import events
from app.features.systematic.catalog import STRATEGIES
from app.quant.history import Prices


def market(days: int = 600) -> Prices:
    index = pd.bdate_range("2022-01-03", periods=days)
    rng = np.random.default_rng(8)
    closes = pd.DataFrame(
        {
            symbol: base * np.exp(np.cumsum(rng.normal(0.0004, 0.01, days)))
            for symbol, base in (("SPUS", 40.0), ("HLAL", 50.0))
        },
        index=index,
    )
    return Prices(closes, closes.shift(1).fillna(closes) * 1.0005)


def test_the_two_engines_agree_closely_on_the_trend_rule() -> None:
    found = events.check(STRATEGIES["halal-etf-trend"], market())
    assert found.days > 300
    assert found.orders == found.fills > 0
    assert abs(found.cagr_gap) < 0.01
    assert found.event.sharpe == pytest.approx(found.vector.sharpe, abs=0.15)


def test_only_trend_rules_are_checked() -> None:
    with pytest.raises(events.EventCheckError, match="trend rules only"):
        events.check(STRATEGIES["halal-etf-rotation"], market())


def test_a_run_that_writes_nothing_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        events, "command", lambda prices, result, spec: ["python3", "-c", "raise SystemExit(3)"]
    )
    with pytest.raises(events.EventCheckError, match="exited with code 3"):
        events.check(STRATEGIES["halal-etf-trend"], market())
