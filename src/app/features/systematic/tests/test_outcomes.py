from datetime import date

import pandas as pd
import pytest

from app.features.systematic.explanations import AgentExplanations, Decision
from app.features.systematic.outcomes import (
    Outcome,
    Record,
    accuracy,
    agent_outcomes,
    calibration,
    importance,
)


def decision(day: str, spus: float, push: float = 0.1) -> Decision:
    return Decision(
        decided=date.fromisoformat(day),
        version=1,
        trained=date(2026, 1, 1),
        weights={"SPUS": spus, "cash": 1 - spus},
        base={"SPUS": 0.5, "cash": 0.5},
        values={"ret_1": {"SPUS": push, "cash": -push}, "rsi": {"SPUS": push / 2, "cash": 0.0}},
        inputs={},
        residual=0.0,
    )


def agent(*decisions: Decision) -> AgentExplanations:
    return AgentExplanations(
        algorithm="ppo",
        version=1,
        assets=["SPUS", "cash"],
        groups=["ret_1", "rsi"],
        decisions=list(decisions),
        overall=None,
        across={},
        unexplained=0,
    )


CLOSES = pd.Series(
    [100.0, 101.0, 99.0], index=pd.to_datetime(["2026-11-02", "2026-11-03", "2026-11-04"])
)


def test_a_weight_move_is_judged_by_the_day_it_is_held() -> None:
    days = [date(2026, 11, 3), date(2026, 11, 4)]
    record = {"ppo": (["SPUS"], days, [[0.2, 0.8], [0.5, 0.5]], {"2026-11-03": [0.83, 0.9]})}
    [signal] = agent_outcomes(record, {"SPUS": CLOSES})
    assert (signal.day, signal.direction, signal.right) == (days[1], "Buy", False)
    assert signal.confidence == 0.83
    assert signal.realized == pytest.approx(99 / 101 - 1)


def test_small_moves_are_holds_and_not_counted() -> None:
    days = [date(2026, 11, 3), date(2026, 11, 4)]
    record: dict[str, Record] = {"ppo": (["SPUS"], days, [[0.5, 0.5], [0.51, 0.49]], {})}
    assert agent_outcomes(record, {"SPUS": CLOSES}) == []


def outcome(day: str, right: bool, confidence: float | None = None) -> Outcome:
    return Outcome(
        source="FinBERT",
        day=date.fromisoformat(day),
        symbol="X",
        direction="Positive",
        confidence=confidence,
        detail="",
        realized=0.01 if right else -0.01,
        right=right,
    )


def test_accuracy_counts_only_the_window() -> None:
    rows = [outcome("2026-11-01", True), outcome("2026-11-02", False), outcome("2026-01-01", True)]
    found = {item.window: item for item in accuracy(rows, date(2026, 11, 7))}
    assert (found[30].signals, found[30].hit_rate) == (2, 0.5)
    assert found[365].signals == 3


def test_calibration_bins_by_confidence() -> None:
    rows = [
        outcome("2026-11-01", True, 0.95),
        outcome("2026-11-01", False, 0.92),
        outcome("2026-11-01", True, 0.55),
    ]
    bins = {item.low: item for item in calibration(rows) if item.source == "FinBERT"}
    assert (bins[0.9].headlines, bins[0.9].hit_rate) == (2, 0.5)
    assert bins[0.5].hit_rate == 1.0
    assert bins[0.7].hit_rate is None


def test_shap_importance_ranks_the_larger_push_first() -> None:
    found = agent(decision("2026-11-05", 0.5, push=0.2))
    rows = {item.group: item for item in importance([found], date(2026, 11, 7))}
    assert rows["ret_1"].rank[30] == 1 and rows["rsi"].rank[30] == 2
    assert rows["ret_1"].windows[30] == pytest.approx(0.2)


def test_agents_are_calibrated_on_their_own_bins() -> None:
    rows = [
        Outcome(
            source="PPO",
            day=date(2026, 11, 5),
            symbol="SPUS",
            direction="Buy",
            confidence=confidence,
            detail="",
            realized=0.01,
            right=right,
        )
        for confidence, right in ((0.3, False), (0.95, True), (0.92, True))
    ]
    bins = {item.low: item for item in calibration(rows)}
    assert bins[0.0].headlines == 1 and bins[0.0].hit_rate == 0.0
    assert bins[0.9].hit_rate == 1.0
