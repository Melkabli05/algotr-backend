from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.features.systematic.navlog import NavLog, tracking


def test_one_reading_a_day_per_book(tmp_path: Path) -> None:
    log = NavLog(tmp_path / "nav.jsonl")
    log.record({"trend": SimpleNamespace(nav=1.0)}, date(2026, 11, 2))
    log.record({"trend": SimpleNamespace(nav=1.01)}, date(2026, 11, 2))
    log.record({"trend": SimpleNamespace(nav=1.03)}, date(2026, 11, 3))
    assert log.readings() == {"trend": {date(2026, 11, 2): 1.01, date(2026, 11, 3): 1.03}}


def test_paper_and_model_start_together_and_the_gap_is_their_difference() -> None:
    paper = {date(2026, 11, 2): 2.0, date(2026, 11, 3): 2.2}
    model = pd.Series([0.05, 0.05], index=pd.to_datetime(["2026-11-02", "2026-11-03"]))
    found = tracking("trend", paper, model)
    assert [point.paper for point in found.points] == pytest.approx([1.0, 1.1])
    assert [point.model for point in found.points] == pytest.approx([1.0, 1.05])
    assert found.gap == pytest.approx(0.05)


def test_no_record_says_so() -> None:
    assert tracking("trend", {}, pd.Series(dtype=float)).since is None
