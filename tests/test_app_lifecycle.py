from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import main
from app.core.settings import get_settings
from app.features.alerts.service import AlertCenter
from app.features.book.service import BookDesk
from app.features.drift.monitor import DriftMonitor
from app.features.ledger.service import Ledger
from app.features.market.schemas import OrderUpdate
from app.features.orders.algo import AlgoDesk
from app.features.orders.iceberg import IcebergDesk
from app.features.risk.history import RiskRecorder
from app.features.scanner.service import Scanner
from app.features.systematic import explanations
from app.features.systematic.agents import AgentDesk
from app.features.systematic.automation import Automation
from app.features.systematic.navlog import NavLog

fetched: list[list[str]] = []


class FakeStreams:
    instances: list["FakeStreams"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.events: list[str] = []
        FakeStreams.instances.append(self)

    def renew_book(self, symbol: str) -> None:
        self.events.append(f"renew {symbol}")

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")

    def pause_data(self) -> None:
        return None

    def resume_data(self) -> None:
        return None

    def data_paused(self) -> bool:
        return False

    def watch(self, symbols: list[str]) -> None:
        return None


@pytest.fixture
def lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeStreams]:
    FakeStreams.instances = []
    monkeypatch.setattr(main, "MarketStreams", FakeStreams)
    offline = {"state_path": tmp_path / "automation.json", "warm_cache": False}
    monkeypatch.setattr(
        main, "Automation", lambda *args, **kwargs: Automation(*args, **{**kwargs, **offline})
    )
    monkeypatch.setattr(
        main,
        "Scanner",
        lambda *args, **kwargs: Scanner(
            *args,
            **{
                **kwargs,
                "root": tmp_path / "scanner",
                "news": None,
                "edgar": None,
                "crypto": None,
                "sentiment": None,
            },
        ),
    )
    monkeypatch.setattr(
        main,
        "IcebergDesk",
        lambda *args, **kwargs: IcebergDesk(
            *args, **{**kwargs, "path": tmp_path / "icebergs.json"}
        ),
    )
    monkeypatch.setattr(
        main,
        "BookDesk",
        lambda *args, **kwargs: BookDesk(*args, **{**kwargs, "root": tmp_path / "book"}),
    )
    monkeypatch.setattr(
        main,
        "AgentDesk",
        lambda *args, **kwargs: AgentDesk(*args, **{**kwargs, "root": tmp_path / "ai"}),
    )
    monkeypatch.setattr(
        main,
        "DriftMonitor",
        lambda *args, **kwargs: DriftMonitor(*args, **{**kwargs, "path": tmp_path / "drift.json"}),
    )
    monkeypatch.setattr(
        main,
        "AlgoDesk",
        lambda *args, **kwargs: AlgoDesk(*args, **{**kwargs, "path": tmp_path / "algos.json"}),
    )
    monkeypatch.setattr(
        main,
        "Ledger",
        lambda *args, **kwargs: Ledger(*args, **{**kwargs, "path": tmp_path / "ledger.sqlite"}),
    )
    monkeypatch.setattr(
        main,
        "AlertCenter",
        lambda *args, **kwargs: AlertCenter(
            *args,
            **{**kwargs, "path": tmp_path / "alerts.json", "log_path": tmp_path / "alerts.jsonl"},
        ),
    )
    monkeypatch.setattr(
        main,
        "RiskRecorder",
        lambda *args, **kwargs: RiskRecorder(*args, **{**kwargs, "path": tmp_path / "risk.jsonl"}),
    )
    monkeypatch.setattr(
        main,
        "NavLog",
        lambda *args, **kwargs: NavLog(*args, **{**kwargs, "path": tmp_path / "nav.jsonl"}),
    )
    monkeypatch.setattr(Scanner, "_missed_today", lambda self: False)
    fetched.clear()

    async def fake_ensure(folders: dict[str, Path]) -> None:
        fetched.append(sorted(folders))

    monkeypatch.setattr(main, "ensure_models", fake_ensure)
    monkeypatch.setenv("AUTO_FETCH_MODELS", "true")
    get_settings.cache_clear()
    yield FakeStreams.instances  # type: ignore[misc]


def test_the_market_streams_start_with_the_app_and_stop_with_it(lifecycle: Any) -> None:
    with TestClient(main.app):
        streams = FakeStreams.instances[0]
        assert streams.events == ["start"]
    assert streams.events == ["start", "stop"]


def test_the_daily_rebalance_is_scheduled_with_the_app_and_stopped_with_it(
    lifecycle: Any,
) -> None:
    with TestClient(main.app) as client:
        automation = main.app.state.automation
        assert automation.scheduler.running
        assert automation.scheduler.get_job("daily-rebalance") is not None
        assert client.get("/systematic/automation").json()["config"]["mode"] == "off"
    assert not automation.scheduler.running


def test_the_nightly_scan_refresh_is_scheduled_with_the_app_and_stopped_with_it(
    lifecycle: Any,
) -> None:
    with TestClient(main.app) as client:
        scanner = main.app.state.scanner
        assert scanner.scheduler.running
        assert scanner.scheduler.get_job("scanner-refresh") is not None
        assert client.get("/scanner/status").json()["last"] is None
    assert not scanner.scheduler.running


def test_the_autopilot_takes_the_scanners_picks_for_its_sleeve(lifecycle: Any) -> None:
    with TestClient(main.app):
        assert main.app.state.automation.sleeve is main.app.state.scanner


def test_the_iceberg_desk_slices_with_the_app_and_stops_with_it(lifecycle: Any) -> None:
    with TestClient(main.app) as client:
        desk = main.app.state.icebergs
        assert desk.scheduler.running
        assert desk.scheduler.get_job("iceberg-slices") is not None
        assert client.get("/orders/iceberg").json() == []
    assert not desk.scheduler.running


def test_the_book_desk_follows_the_coins_with_the_app_and_stops_with_it(lifecycle: Any) -> None:
    with TestClient(main.app) as client:
        desk = main.app.state.book
        assert desk.scheduler.running and desk.scheduler.get_job("book-sample") is not None
        assert list(desk.keeper.books) == ["BTC/USD", "ETH/USD"]
        assert desk.renew.__self__ is FakeStreams.instances[0]
        assert client.get("/book/status").json()["books"][0]["symbol"] == "BTC/USD"
    assert not desk.scheduler.running


def test_the_algorithm_desk_runs_with_the_app_and_stops_with_it(lifecycle: Any) -> None:
    with TestClient(main.app) as client:
        desk = main.app.state.algos
        assert desk.scheduler.running
        assert desk.scheduler.get_job("algo-slices") is not None
        assert client.get("/orders/algo").json() == []
    assert not desk.scheduler.running


def test_the_ledger_syncs_with_the_app_and_stops_with_it(lifecycle: Any) -> None:
    with TestClient(main.app):
        ledger = main.app.state.ledger
        assert ledger.scheduler.running
        assert ledger.scheduler.get_job("ledger-sync") is not None
        assert main.app.state.automation.audit is ledger.log
        assert main.app.state.audit is ledger.log
    assert not ledger.scheduler.running
    assert main.app.state.audit is None


def update(event: str) -> OrderUpdate:
    return OrderUpdate(
        event=event,
        order_id="abc",
        symbol="F",
        side="buy",
        status="filled",
        qty=1,
        filled_qty=1,
        filled_avg_price=12.5,
        time=datetime(2026, 11, 3, tzinfo=UTC),
        price=12.5,
        fill_qty=1,
    )


def test_a_fill_on_the_stream_makes_the_ledger_sync_and_other_events_do_not(
    lifecycle: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    with TestClient(main.app):
        ledger = main.app.state.ledger
        monkeypatch.setattr(ledger, "refresh", lambda: None)
        for event in ("new", "canceled", "replaced"):
            main.app.state.hub.publish(update(event))
        assert ledger.scheduler.get_job("ledger-fill") is None
        main.app.state.hub.publish(update("partial_fill"))
        assert ledger.scheduler.get_job("ledger-fill") is not None


def test_the_drift_monitor_checks_daily_with_the_app_and_sizes_the_autopilot(
    lifecycle: Any,
) -> None:
    with TestClient(main.app) as client:
        monitor = main.app.state.drift
        assert monitor.scheduler.running and monitor.scheduler.get_job("drift-check") is not None
        assert main.app.state.automation.sizing is monitor
        found = [type(source).__name__ for source in monitor.sources]
        assert found == ["StrategyReturns", "AgentDesk"]
        monitor.sources = []
        assert client.get("/drift/status").json()["action"] == "act"
    assert not monitor.scheduler.running


def test_the_agents_follow_the_schedule_with_the_app_and_start_off(
    lifecycle: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = get_settings().model_copy(update={"ai_agents": False})
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    with TestClient(main.app) as client:
        desk = main.app.state.agents
        assert desk.scheduler.running and desk.scheduler.get_job("ai-agents") is not None
        assert not desk.enabled and main.app.state.drift.retrain == desk.request_retrain
        assert desk in main.app.state.drift.sources
        main.app.state.drift.sources = []
        body = client.get("/systematic/agents").json()
        assert body["status"]["enabled"] is False and body["evaluation"] is None
    assert not desk.scheduler.running


def test_the_explanations_of_an_agent_are_served_from_the_audit_log(lifecycle: Any) -> None:
    arrays = {
        "groups": ["ret_1"],
        "assets": ["HLAL", "SPUS", "cash"],
        "columns": ["ret_1:HLAL", "ret_1:SPUS", "held:HLAL", "held:SPUS", "held:cash"],
        "background": {},
        "decisions": [
            {
                "weights": [0.5, 0.3, 0.2],
                "base": [0.4, 0.3, 0.3],
                "values": [[0.1, 0.0, -0.1]],
                "residual": 1e-8,
            }
        ],
        "overall": [[0.1, 0.0, 0.1]],
        "sample": 60,
        "shap": "0.54.1",
    }
    with TestClient(main.app) as client:
        main.app.state.drift.sources = []
        desk = main.app.state.agents
        empty = client.get("/systematic/agents/ppo/explanations").json()
        assert empty["decisions"] == [] and empty["overall"] is None
        assert client.get("/systematic/agents/nobody/explanations").status_code == 404
        model = {"version": 1, "trained": "2026-11-01"}
        record = explanations.build(arrays, [(date(2026, 11, 3), [0.1] * 5)], model, 60)
        desk.log.record(explanations.KIND, {"agents": {"ppo": record}, "unexplained": []})
        body = client.get("/systematic/agents/ppo/explanations", params={"limit": 5}).json()
        assert [item["decided"] for item in body["decisions"]] == ["2026-11-03"]
        assert body["version"] == 1 and body["overall"]["days"] == 60
        assert body["decisions"][0]["trained"] == "2026-11-01"
        assert body["decisions"][0]["values"] == {"ret_1": {"HLAL": 0.1, "SPUS": 0.0, "cash": -0.1}}
        assert client.get("/systematic/agents/sac/explanations").json()["decisions"] == []


def test_the_missing_models_are_fetched_when_the_app_starts(lifecycle: Any) -> None:
    with TestClient(main.app):
        pass
    assert fetched == [["finbert", "kronos"]]


def test_no_model_is_fetched_when_the_setting_is_off(
    lifecycle: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTO_FETCH_MODELS", "false")
    get_settings.cache_clear()
    with TestClient(main.app):
        pass
    assert fetched == []
