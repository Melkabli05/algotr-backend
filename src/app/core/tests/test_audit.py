from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.core import audit


@pytest.fixture
def log(tmp_path: Path) -> audit.AuditLog:
    return audit.AuditLog(tmp_path / "audit.jsonl")


def test_an_empty_log_is_valid_and_has_no_entries(log: audit.AuditLog) -> None:
    assert log.entries() == []
    assert log.verify() == audit.Verification(valid=True, entries=0)


def test_entries_are_numbered_and_each_carries_the_hash_of_the_one_before(
    log: audit.AuditLog,
) -> None:
    first = log.record("run", {"strategy": "halal-etf-trend"})
    second = log.record("lab", {"candidates": 12})

    assert (first.seq, second.seq) == (1, 2)
    assert first.previous == audit.GENESIS
    assert second.previous == first.hash
    assert log.verify() == audit.Verification(valid=True, entries=2)
    assert [e.kind for e in log.entries()] == ["run", "lab"]
    assert [e.kind for e in log.entries(1)] == ["lab"]
    assert log.count("lab") == 1


def test_editing_a_recorded_value_breaks_the_chain_at_that_entry(
    log: audit.AuditLog, tmp_path: Path
) -> None:
    for sharpe in (0.9, 1.1, 1.3):
        log.record("run", {"sharpe": sharpe})
    path = tmp_path / "audit.jsonl"
    path.write_text(path.read_text().replace("1.1", "9.9"))

    assert log.verify() == audit.Verification(valid=False, entries=3, broken_at=2)


def test_deleting_an_entry_from_the_middle_is_detected(log: audit.AuditLog, tmp_path: Path) -> None:
    for number in range(3):
        log.record("run", {"n": number})
    path = tmp_path / "audit.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")

    assert not log.verify().valid


def test_equal_content_always_gets_the_same_hash_whatever_the_key_order() -> None:
    assert audit.digest({"a": 1, "b": [1, 2]}) == audit.digest({"b": [1, 2], "a": 1})
    assert audit.digest({"a": 1}) != audit.digest({"a": 2})


def prices(shift: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=5)
    return pd.DataFrame({"SPUS": [1.0, 2, 3, 4, 5 + shift], "HLAL": [5.0, 4, 3, 2, 1]}, index=index)


def test_a_data_fingerprint_changes_when_any_price_changes() -> None:
    same, changed = audit.fingerprint(prices()), audit.fingerprint(prices(0.01))

    assert same["rows"] == 5
    assert (same["first"][:10], same["last"][:10]) == ("2024-01-01", "2024-01-05")
    assert same["symbols"] == ["SPUS", "HLAL"]
    assert same["sha256"] == audit.fingerprint(prices())["sha256"]
    assert same["sha256"] != changed["sha256"]


def test_a_manifest_pins_config_data_code_and_environment() -> None:
    manifest = audit.manifest({"window": 200}, {"trend": prices()}, seed=7)

    assert manifest["config_hash"] == audit.digest({"window": 200})
    assert manifest["data"]["trend"]["rows"] == 5
    assert manifest["seed"] == 7
    assert set(manifest["git"]) == {"commit", "uncommitted_changes"}
    assert manifest["environment"]["packages"]["pandas"]
    assert manifest["environment"]["python"].startswith("3.")


def test_the_long_layout_finrl_reads_is_fingerprinted_by_content_too() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "tic": ["HLAL", "SPUS", "HLAL", "SPUS"],
            "close": [1.0, 2.0, 3.0, 4.0],
        }
    )
    changed = frame.assign(close=[1.0, 2.0, 3.0, 4.01])

    first = audit.fingerprint(frame)

    assert first["symbols"] == ["HLAL", "SPUS"]
    assert (first["rows"], first["first"], first["last"]) == (4, "2024-01-02", "2024-01-03")
    assert first["sha256"] == audit.fingerprint(frame.copy())["sha256"]
    assert first["sha256"] != audit.fingerprint(changed)["sha256"]


def test_numbers_json_cannot_hold_are_stored_as_null_and_the_chain_still_verifies(
    log: audit.AuditLog,
) -> None:
    entry = log.record("run", {"sharpe": float("nan"), "nested": [1.0, float("inf")], "ok": 2.5})

    assert entry.payload == {"sharpe": None, "nested": [1.0, None], "ok": 2.5}
    log.record("run", {"after": 1})
    assert log.verify() == audit.Verification(valid=True, entries=2)
    assert [e.payload["sharpe"] for e in log.entries()[:1]] == [None]


def test_dates_tuples_and_paths_are_stored_as_the_json_a_reader_expects(
    log: audit.AuditLog,
) -> None:
    entry = log.record(
        "run",
        {"day": date(2026, 1, 2), "pair": (1, 2.5), "where": Path("data/x"), "big": 2**70},
    )

    assert entry.payload == {
        "day": "2026-01-02",
        "pair": [1, 2.5],
        "where": "data/x",
        "big": 2**70,
    }
    assert log.verify().valid
