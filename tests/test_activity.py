from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS
from typing import Any

from app import activity
from app.core import audit

NOW = datetime.now(UTC)


def fill(minutes: int) -> Any:
    return NS(time=NOW - timedelta(minutes=minutes), side="buy", qty=3.0, symbol="SPUS", price=55.5)


def state(log: audit.AuditLog, fills: list[Any]) -> Any:
    return NS(audit=log, ledger=NS(fills=lambda days, limit: fills))


def test_orders_fills_and_runs_come_newest_first_with_refusals_as_risk(tmp_path: Any) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    log.record(
        "order",
        {
            "action": "submit",
            "request": {"symbol": "JPM", "side": "buy", "qty": 1.0},
            "outcome": "refused",
            "reason": "JPM fails the Shariah screen",
        },
    )
    log.record("run", {"run": {"status": "halted", "detail": "drawdown"}})
    log.record("lab", {"result": {}})
    found = activity.collect(state(log, [fill(1)]), NOW - timedelta(hours=1))
    kinds = [(event.kind, event.level, event.title) for event in found]
    assert ("risk", "warn", "Submit buy 1.0 JPM") in kinds
    assert ("run", "error", "Autopilot run halted") in kinds
    assert ("research", "info", "Rule lab run") in kinds
    assert found[-1].title == "Filled buy 3 SPUS at 55.50"
    assert [event.time for event in found] == sorted((e.time for e in found), reverse=True)


def test_older_events_are_left_out(tmp_path: Any) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    found = activity.collect(state(log, [fill(120)]), NOW - timedelta(hours=1))
    assert found == []
