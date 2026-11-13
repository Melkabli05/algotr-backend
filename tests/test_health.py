from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS
from typing import Any

import pytest

from app import health
from app.core import sidecar
from app.core.broker import get_trading_client
from app.core.compliance import Screening

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


def state(**over: Any) -> Any:
    parts: dict[str, Any] = {
        "automation": NS(
            status=lambda: NS(
                halted=None,
                runs=[NS(status="ok", detail=None)],
                config=NS(mode="paper"),
                next_run=NOW + timedelta(hours=12),
            )
        ),
        "audit": NS(verify=lambda: NS(valid=True, entries=5, broken_at=None)),
        "ledger": NS(
            status=lambda: NS(error=None, activities=9, reconciliation=NS(ok=True, differences=[]))
        ),
        "book": NS(status=lambda: NS(stale=False, books=[1, 2])),
        "scanner": NS(
            status=lambda: NS(
                last=NS(
                    started=NOW - timedelta(hours=3),
                    status="ok",
                    detail=None,
                    session="2026-11-03",
                    lists=[],
                ),
                news=NS(error=None, filings_error=None, sentiment=NS(error=None)),
            ),
            screen=lambda symbol: Screening(symbol=symbol, allowed=symbol != "XOM", reason="x"),
        ),
        "drift": NS(status=lambda: NS(streams=[NS(name="ai-ppo", until=None)])),
        "agents": NS(status=lambda: NS(enabled=False, busy=None, agents=[])),
    }
    return NS(**{**parts, **over})


def levels(found: health.Status) -> dict[str, str]:
    return {check.name: check.level for check in found.checks}


def test_a_healthy_app_is_ok_throughout() -> None:
    found = health.report(state(), NOW)
    assert found.level == "ok" and set(levels(found).values()) == {"ok"}
    assert [check.name for check in found.checks] == [
        "Autopilot",
        "Audit log",
        "Ledger",
        "Order book",
        "Scanner",
        "Drift",
        "Agents",
    ]


def test_a_failed_autopilot_run_is_an_error_with_its_reason() -> None:
    failing = NS(
        status=lambda: NS(
            halted=None,
            runs=[NS(status="error", detail="NautilusRunError: boom")],
            config=NS(mode="paper"),
            next_run=None,
        )
    )
    found = health.report(state(automation=failing), NOW)
    assert found.level == "error"
    assert "NautilusRunError: boom" in found.checks[0].detail


def test_only_the_newest_run_counts_and_the_status_lists_runs_newest_first() -> None:
    runs = [NS(status="ok", detail=None), NS(status="error", detail="old failure")]
    recovered = NS(
        status=lambda: NS(halted=None, runs=runs, config=NS(mode="paper"), next_run=None)
    )
    assert health.report(state(automation=recovered), NOW).checks[0].level == "ok"
    failing = NS(
        status=lambda: NS(halted=None, runs=runs[::-1], config=NS(mode="paper"), next_run=None)
    )
    assert health.report(state(automation=failing), NOW).checks[0].level == "error"


def test_a_halted_autopilot_is_an_error_and_an_off_one_is_fine() -> None:
    halted = NS(
        status=lambda: NS(halted="drawdown 16%", runs=[], config=NS(mode="paper"), next_run=None)
    )
    off = NS(
        status=lambda: NS(
            halted=None, runs=[NS(status="error", detail="x")], config=NS(mode="off"), next_run=None
        )
    )
    assert health.report(state(automation=halted), NOW).checks[0].detail == "Halted: drawdown 16%"
    assert health.report(state(automation=off), NOW).checks[0].level == "ok"


def test_a_broken_chain_is_an_error() -> None:
    broken = NS(verify=lambda: NS(valid=False, entries=5, broken_at=3))
    found = health.report(state(audit=broken), NOW)
    assert levels(found)["Audit log"] == "error" and "#3" in found.checks[1].detail


def test_ledger_book_and_drift_problems_are_warnings() -> None:
    found = health.report(
        state(
            ledger=NS(
                status=lambda: NS(
                    error=None, activities=1, reconciliation=NS(ok=False, differences=[1, 2])
                )
            ),
            book=NS(status=lambda: NS(stale=True, books=[1])),
            drift=NS(status=lambda: NS(streams=[NS(name="ai-ppo", until="2026-11-20")])),
        ),
        NOW,
    )
    assert found.level == "warn"
    assert levels(found) | {} == {
        "Autopilot": "ok",
        "Audit log": "ok",
        "Ledger": "warn",
        "Order book": "warn",
        "Scanner": "ok",
        "Drift": "warn",
        "Agents": "ok",
    }
    assert "2 differences" in found.checks[2].detail and "ai-ppo" in found.checks[5].detail


@pytest.mark.parametrize(
    ("last", "words"),
    [
        (None, "No scan yet"),
        (
            NS(started=NOW - timedelta(hours=40), status="ok", detail=None, session="s", lists=[]),
            "40 hours old",
        ),
        (
            NS(started=NOW, status="error", detail="boom", session="s", lists=[]),
            "Refresh failed: boom",
        ),
        (
            NS(
                started=NOW,
                status="ok",
                detail=None,
                session="s",
                lists=[NS(name="HLAL", error="no file")],
            ),
            "HLAL: no file",
        ),
    ],
)
def test_the_scanner_says_why_it_is_not_well(last: Any, words: str) -> None:
    scanner = NS(
        status=lambda: NS(
            last=last, news=NS(error=None, filings_error=None, sentiment=NS(error=None))
        )
    )
    found = health.report(state(scanner=scanner), NOW)
    assert levels(found)["Scanner"] == "warn" and words in found.checks[4].detail


def test_a_news_or_sentiment_failure_is_a_scanner_warning() -> None:
    last = NS(started=NOW, status="ok", detail=None, session="s", lists=[])
    news = NS(error=None, filings_error="HTTPError", sentiment=NS(error="SidecarUnavailableError"))
    found = health.report(state(scanner=NS(status=lambda: NS(last=last, news=news))), NOW)
    assert (
        "HTTPError" in found.checks[4].detail
        and "SidecarUnavailableError" in found.checks[4].detail
    )


def test_agents_that_are_on_report_their_backlog_and_a_missing_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    on = NS(
        status=lambda: NS(enabled=True, busy=None, agents=[NS(unexplained=2), NS(unexplained=1)])
    )
    monkeypatch.setattr(sidecar, "available", lambda: True)
    assert "3 decisions waiting" in health.report(state(agents=on), NOW).checks[6].detail
    monkeypatch.setattr(sidecar, "available", lambda: False)
    assert "sidecar is not set up" in health.report(state(agents=on), NOW).checks[6].detail


def test_the_route_serves_the_report(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import health
    from app.main import app

    class Frozen(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> "Frozen":
            return cls.fromtimestamp(NOW.timestamp(), tz)

    monkeypatch.setattr(health, "datetime", Frozen)
    trading = app.dependency_overrides[get_trading_client]()
    trading.get_all_positions.return_value = []
    for name, value in state().__dict__.items():
        monkeypatch.setattr(app.state, name, value, raising=False)
    body = client.get("/status").json()
    assert body["level"] == "ok" and len(body["checks"]) == 8


def held(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "side": "long",
        "qty": "1",
        "avg_entry_price": "10",
        "current_price": "10",
        "market_value": "10",
        "unrealized_pl": "0",
    }


def test_a_holding_that_left_the_shariah_lists_is_a_warning() -> None:
    from unittest.mock import MagicMock

    trading = MagicMock()
    trading.get_all_positions.return_value = [held("SPUS"), held("XOM")]
    found = health.report(state(), NOW, trading=trading).checks[-1]
    assert (found.name, found.level, found.detail) == (
        "Shariah",
        "warn",
        "Not on the lists now: XOM",
    )
    trading.get_all_positions.return_value = [held("SPUS")]
    found = health.report(state(), NOW, trading=trading).checks[-1]
    assert (found.level, found.detail) == ("ok", "1 holdings pass the screen")
    trading.get_all_positions.side_effect = ConnectionError("offline")
    assert health.report(state(), NOW, trading=trading).checks[-1].level == "warn"
