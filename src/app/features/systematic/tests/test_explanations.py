from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app.core import audit
from app.features.systematic import explanations

ASSETS = ["HLAL", "SPUS", "cash"]
GROUPS = ["ret_1", "held"]


def found(count: int, sample: int) -> dict[str, Any]:
    decision = {
        "weights": [0.5, 0.3, 0.2],
        "base": [0.4, 0.3, 0.3],
        "values": [[0.06, 0.0, -0.06], [0.04, 0.0, -0.04]],
        "residual": 1.23e-8,
    }
    return {
        "groups": GROUPS,
        "assets": ASSETS,
        "columns": ["ret_1:HLAL", "ret_1:SPUS", "held:HLAL", "held:SPUS", "held:cash"],
        "background": {"first": "2023-01-02", "last": "2026-10-30", "days": 756, "rows": 64},
        "decisions": [decision] * count,
        "overall": [[0.03, 0.01, 0.02], [0.05, 0.0, 0.05]] if sample else None,
        "sample": sample,
        "shap": "0.54.1",
    }


def record(version: int, days: list[date], sample: int = 0, scale: float = 1.0) -> dict[str, Any]:
    queued = [(day, [0.12345678, -1.5, 0.4, 0.4, 0.2]) for day in days]
    model = {"version": version, "trained": "2026-11-01"}
    output = explanations.build(found(len(days), sample), queued, model, sample)
    for item in output["decisions"]:
        for pushes in item["values"].values():
            for asset in pushes:
                pushes[asset] *= scale
    return output


def log_of(tmp_path: Path, *entries: dict[str, Any]) -> list[audit.Entry]:
    log = audit.AuditLog(tmp_path / "log.jsonl")
    log.record("run", {"run": {}})
    for payload in entries:
        log.record(explanations.KIND, payload)
    return log.entries()


def test_the_arrays_of_the_sidecar_become_named_rounded_values() -> None:
    output = record(3, [date(2026, 11, 3)], sample=60)
    (decision,) = output["decisions"]
    assert decision["decided"] == "2026-11-03"
    assert decision["weights"] == {"HLAL": 0.5, "SPUS": 0.3, "cash": 0.2}
    assert decision["values"] == {
        "ret_1": {"HLAL": 0.06, "SPUS": 0.0, "cash": -0.06},
        "held": {"HLAL": 0.04, "SPUS": 0.0, "cash": -0.04},
    }
    assert decision["inputs"] == {
        "ret_1": {"HLAL": 0.123457, "SPUS": -1.5},
        "held": {"HLAL": 0.4, "SPUS": 0.4, "cash": 0.2},
    }
    assert decision["residual"] == 1.23e-08
    assert output["model"] == {"version": 3, "trained": "2026-11-01"}
    assert output["groups"] == GROUPS and output["shap"] == "0.54.1"
    assert output["overall"] == {
        "days": 60,
        "values": {
            "ret_1": {"HLAL": 0.03, "SPUS": 0.01, "cash": 0.02},
            "held": {"HLAL": 0.05, "SPUS": 0.0, "cash": 0.05},
        },
    }


def test_a_model_already_explained_carries_no_overall() -> None:
    assert record(1, [date(2026, 11, 3)], sample=0)["overall"] is None


def test_a_decision_and_an_observation_that_do_not_pair_up_are_refused() -> None:
    with pytest.raises(ValueError, match="zip"):
        explanations.build(found(2, 0), [(date(2026, 11, 3), [0.0] * 5)], {}, 0)


def test_the_newest_decisions_come_first_across_entries_up_to_the_limit(tmp_path: Path) -> None:
    first = record(1, [date(2026, 11, 2), date(2026, 11, 3)], sample=60)
    second = record(1, [date(2026, 11, 4)])
    entries = log_of(
        tmp_path,
        {"agents": {"ppo": first}, "unexplained": []},
        {"agents": {"ppo": second, "sac": second}, "unexplained": []},
    )
    found_ppo = explanations.read(entries, "ppo", 2)
    assert [item.decided for item in found_ppo.decisions] == [date(2026, 11, 4), date(2026, 11, 3)]
    assert len(explanations.read(entries, "ppo", 10).decisions) == 3
    assert explanations.read(entries, "ppo", 0).decisions == []
    assert found_ppo.assets == ASSETS and found_ppo.groups == GROUPS and found_ppo.version == 1
    assert {(item.version, item.trained) for item in found_ppo.decisions} == {
        (1, date(2026, 11, 1))
    }


def test_the_overall_and_the_version_are_those_of_the_newest_entry_that_has_them(
    tmp_path: Path,
) -> None:
    old = record(1, [date(2026, 10, 30)], sample=60)
    new = record(2, [date(2026, 11, 3)], sample=60)
    new["overall"]["days"] = 30
    entries = log_of(
        tmp_path,
        {"agents": {"ppo": old}, "unexplained": []},
        {"agents": {"ppo": new}, "unexplained": []},
        {"agents": {"ppo": record(2, [date(2026, 11, 4)])}, "unexplained": []},
    )
    found_ppo = explanations.read(entries, "ppo", 10)
    assert found_ppo.version == 2 and found_ppo.overall is not None
    assert found_ppo.overall.days == 30


def test_what_was_lost_is_counted_for_that_agent_only(tmp_path: Path) -> None:
    lost = {"agent": "ppo", "decided": "2026-11-02", "reason": "replaced"}
    other = {"agent": "sac", "decided": "2026-11-02", "reason": "replaced"}
    entries = log_of(tmp_path, {"agents": {}, "unexplained": [lost, other, lost]})
    assert explanations.read(entries, "ppo", 5).unexplained == 2
    assert explanations.read(entries, "a2c", 5).unexplained == 0


def test_an_agent_with_no_entries_reads_as_empty(tmp_path: Path) -> None:
    empty = explanations.read(log_of(tmp_path), "ppo", 5)
    assert empty.decisions == [] and empty.overall is None and empty.version is None
    assert empty.across == {} and empty.assets == []


def test_the_typical_push_is_the_mean_size_whatever_the_sign(tmp_path: Path) -> None:
    entries = log_of(
        tmp_path,
        {
            "agents": {
                "ppo": record(1, [date(2026, 11, 2)], scale=1.0)
                | {
                    "decisions": record(1, [date(2026, 11, 2)], scale=1.0)["decisions"]
                    + record(1, [date(2026, 11, 3)], scale=-3.0)["decisions"]
                }
            },
            "unexplained": [],
        },
    )
    across = explanations.read(entries, "ppo", 5).across
    assert across["ret_1"]["HLAL"] == pytest.approx((0.06 + 0.18) / 2)
    assert across["ret_1"]["cash"] == pytest.approx((0.06 + 0.18) / 2)
    assert across["held"]["SPUS"] == 0


def test_decisions_of_models_with_different_indicators_can_be_read_together(
    tmp_path: Path,
) -> None:
    older = record(1, [date(2026, 11, 2)])
    newer = record(2, [date(2026, 11, 3)])
    for item in newer["decisions"]:
        item["values"]["sentiment"] = {"HLAL": 0.02, "SPUS": 0.0, "cash": -0.02}
    newer["groups"] = [*GROUPS, "sentiment"]
    entries = log_of(
        tmp_path,
        {"agents": {"ppo": older}, "unexplained": []},
        {"agents": {"ppo": newer}, "unexplained": []},
    )
    found_ = explanations.read(entries, "ppo", 5)
    assert len(found_.decisions) == 2
    assert found_.across["sentiment"]["HLAL"] == pytest.approx(0.02)
    assert found_.across["ret_1"]["HLAL"] == pytest.approx(0.06)
