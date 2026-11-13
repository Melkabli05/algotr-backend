from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.features.drift.monitor import RULES, DriftMonitor
from app.features.drift.router import get_monitor
from app.features.drift.tests.support import Source, series
from app.features.drift.tests.test_monitor import first_live_day
from app.main import app


@pytest.fixture
def desk(tmp_path: Path) -> Iterator[DriftMonitor]:
    today: date = first_live_day()
    found = DriftMonitor(RULES, None, tmp_path / "drift.json", lambda: today)
    found.sources = [Source(series((0.0005, 0.02)))]
    app.dependency_overrides[get_monitor] = lambda: found
    yield found
    app.dependency_overrides.pop(get_monitor, None)


def test_the_status_lists_the_rules_the_streams_and_what_was_recorded(
    client: TestClient, desk: DriftMonitor
) -> None:
    client.post("/drift/check")
    body = client.get("/drift/status").json()
    assert body["action"] == "act" and body["threshold"] == 0.1 and body["hold_days"] == 30
    [stream] = body["streams"]
    assert stream["name"] == "halal-etf-trend" and stream["monitored"] is True
    assert stream["events"] and body["recorded"]


def test_a_check_scans_now_and_returns_the_new_status(
    client: TestClient, desk: DriftMonitor
) -> None:
    assert desk.checked is None
    body = client.post("/drift/check").json()
    assert desk.checked is not None and body["checked"] is not None
    assert body["started"] == first_live_day().isoformat()
