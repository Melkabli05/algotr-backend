from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.system_health import latency, next_month, order_path, percentile


def test_percentiles_and_the_slowest_routes() -> None:
    rows = [(0.0, "/a", float(ms), 200) for ms in range(1, 101)] + [(0.0, "/b", 900.0, 502)]
    found = latency(rows)
    assert found.p50 == pytest.approx(51)
    assert found.errors == 1
    assert found.slowest[0] == ("/b", 900.0)
    assert percentile([], 0.5) is None


def test_a_model_retrains_the_month_after_it_was_trained() -> None:
    assert next_month(date(2026, 11, 4)) == date(2026, 12, 1)
    assert next_month(date(2026, 12, 31)) == date(2027, 1, 1)


def test_the_order_path_times_each_stage() -> None:
    start = datetime(2026, 11, 5, 15, tzinfo=UTC)
    orders = [
        SimpleNamespace(
            created_at=start,
            submitted_at=start + timedelta(milliseconds=40),
            filled_at=start + timedelta(seconds=2),
        )
    ]
    entries = [
        SimpleNamespace(
            kind="order",
            payload={"action": "submit", "outcome": "accepted", "result": {"ms": 120.0}},
        ),
        SimpleNamespace(
            kind="order", payload={"action": "submit", "outcome": "refused", "result": None}
        ),
    ]
    stages, outcomes = order_path(orders, entries)
    assert [item.median for item in stages] == pytest.approx([120.0, 40.0, 1.96])
    assert outcomes == {"accepted": 1, "refused": 1}
