from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine

from app.core.broker import get_crypto_data_client, get_stock_data_client
from app.features.systematic.automation import Automation
from app.features.systematic.router import get_automation
from app.features.systematic.schemas import AutomationConfig
from app.features.systematic.tests.support import (
    LIVE_SETTINGS,
    SETTINGS,
    fake_history,
    position,
    trading_with,
)
from app.main import app
from app.quant import history


@pytest.fixture
def market(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    return fake_history(monkeypatch)


@pytest.fixture
def automation(client: TestClient, tmp_path: Path) -> Iterator[Automation]:
    built = Automation(
        SETTINGS,
        trading_with(),
        app.dependency_overrides[get_stock_data_client](),
        app.dependency_overrides[get_crypto_data_client](),
        state_path=tmp_path / "automation.json",
    )
    app.dependency_overrides[get_automation] = lambda: built
    yield built


def test_the_overview_has_everything_the_dashboard_shows(
    client: TestClient, automation: Automation, market: dict[str, str]
) -> None:
    automation.configure(AutomationConfig(strategies=["halal-etf-trend"]))
    body = client.get("/systematic/overview").json()
    assert body["status"]["config"]["mode"] == "off"
    assert [s["active"] for s in body["strategies"]] == [False, False, True, False, False]
    first = body["strategies"][0]["research"]
    assert first["metrics"]["total_return"] > 0
    assert len(first["checks"]) == 5
    assert body["plan"]["strategies"] == ["halal-etf-trend"]
    assert body["equity"] == 100_000
    assert body["drawdown"] == 0


def test_the_execution_report_starts_empty_and_lists_settled_fills(
    client: TestClient, automation: Automation, market: dict[str, str]
) -> None:
    assert client.get("/systematic/execution").json() == {"fills": [], "summaries": []}
    automation.configure(
        AutomationConfig(mode="paper", engine="alpaca", strategies=["btc-trend-200"])
    )
    automation.run()
    automation.state.runs[0].executed[0] = (
        automation.state.runs[0]
        .executed[0]
        .model_copy(
            update={"settled": True, "filled_qty": 1.0, "filled_price": 100.0, "cost_bps": 30.0}
        )
    )
    body = client.get("/systematic/execution").json()
    assert [fill["symbol"] for fill in body["fills"]] == ["BTC/USD"]
    assert body["summaries"][0]["cost_bps"] == 30.0


def test_the_sleeve_is_set_through_the_api_and_more_than_a_quarter_is_refused(
    client: TestClient, automation: Automation, market: dict[str, str]
) -> None:
    sleeve = {"mode": "shadow", "share": 0.2, "variant": "mom_12_1 top 10", "override": False}
    saved = client.put("/systematic/automation", json={"sleeve": sleeve}).json()["config"]
    assert saved["sleeve"] == sleeve
    refused = client.put("/systematic/automation", json={"sleeve": {**sleeve, "share": 0.3}})
    assert refused.status_code == 422
    assert client.get("/systematic/automation").json()["config"]["sleeve"] == sleeve


def test_automation_is_configured_then_deployed_through_the_api(
    client: TestClient, automation: Automation, market: dict[str, str]
) -> None:
    assert client.get("/systematic/automation").json()["config"]["mode"] == "off"
    assert client.post("/systematic/automation/run").status_code == 409

    body = {
        "mode": "paper",
        "allocation": 0.25,
        "strategies": ["btc-trend-200"],
        "max_drawdown": 0.2,
        "engine": "alpaca",
    }
    saved = client.put("/systematic/automation", json=body).json()["config"]
    assert saved == {
        **body,
        "automatic": False,
        "crypto": False,
        "reduce_at": 0.5,
        "paused": [],
        "sleeve": {"mode": "off", "share": 0.0, "variant": "rank_avg top 20", "override": False},
    }

    record = client.post("/systematic/automation/run").json()
    assert record["status"] == "ok"
    assert [e["status"] for e in record["executed"]] == ["accepted"]
    assert client.get("/systematic/automation").json()["runs"][0]["time"] == record["time"]


def test_automatic_choice_and_crypto_are_saved_without_naming_strategies(
    client: TestClient, automation: Automation
) -> None:
    body = {"mode": "dry-run", "automatic": True, "crypto": True, "engine": "nautilus"}
    saved = client.put("/systematic/automation", json=body).json()["config"]
    assert (saved["automatic"], saved["crypto"], saved["engine"]) == (True, True, "nautilus")


def test_strategies_are_deduplicated(client: TestClient, automation: Automation) -> None:
    body = {"mode": "off", "strategies": ["btc-trend-200", "btc-trend-200"]}
    assert client.put("/systematic/automation", json=body).json()["config"]["strategies"] == [
        "btc-trend-200"
    ]


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "live"},
        {"allocation": 0},
        {"allocation": 1.1},
        {"strategies": ["martingale"]},
        {"max_drawdown": 0.01},
        {"max_drawdown": 0.9},
    ],
)
def test_invalid_automation_settings_are_rejected(
    client: TestClient, automation: Automation, body: dict[str, Any]
) -> None:
    assert client.put("/systematic/automation", json=body).status_code == 422


def test_trading_a_live_account_is_refused_until_the_settings_open_it(
    client: TestClient, automation: Automation
) -> None:
    automation.settings = LIVE_SETTINGS
    response = client.put("/systematic/automation", json={"mode": "paper"})
    assert response.status_code == 422
    assert "ALPACA_TRADING_MODE=live" in response.json()["detail"]


def test_a_halted_automation_is_resumed_through_the_api(
    client: TestClient, automation: Automation, market: dict[str, str], database: Engine
) -> None:
    automation.trading.get_all_positions.return_value = [position("BTCUSD", 0.1, 80_000)]  # type: ignore[attr-defined]
    client.put("/systematic/automation", json={"mode": "dry-run", "max_drawdown": 0.05})
    automation.state.halted = "drawdown 6.0% reached the 5% limit"
    status = client.get("/systematic/automation").json()
    assert status["halted"].startswith("drawdown")
    assert client.post("/systematic/automation/resume").json()["halted"] is None


@pytest.fixture
def one_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.features.systematic import parallel

    monkeypatch.setattr(parallel, "default_workers", lambda jobs: 1)


def test_the_lab_is_empty_until_it_has_been_run(client: TestClient, automation: Automation) -> None:
    assert client.get("/systematic/lab").json() is None


def test_running_the_lab_returns_every_family_and_is_remembered(
    client: TestClient, automation: Automation, market: dict[str, str], one_worker: None
) -> None:
    result = client.post("/systematic/lab").json()

    assert {f["family"] for f in result["families"]} == {
        "btc-trend-200",
        "halal-etf-trend",
        "halal-multi-trend",
        "halal-etf-rotation",
    }
    first = result["families"][0]
    assert {"reality_check_p", "probability_of_overfitting", "cumulative_trials"} <= set(first)
    assert {"deflated_sharpe", "survives", "t_statistic"} <= set(first["candidates"][0])
    assert client.get("/systematic/lab").json() == result


def test_the_audit_trail_lists_the_newest_entries_first_with_the_chain_status(
    client: TestClient, automation: Automation, market: dict[str, str], one_worker: None
) -> None:
    client.post("/systematic/lab")
    automation.audit.record("note", {"text": "second"})

    body = client.get("/systematic/audit", params={"limit": 5}).json()

    assert body["verification"] == {"valid": True, "entries": 2, "broken_at": None}
    assert [e["kind"] for e in body["entries"]] == ["note", "lab"]
    assert [e["seq"] for e in body["entries"]] == [2, 1]


def test_a_tampered_audit_log_is_reported_as_broken(
    client: TestClient, automation: Automation, tmp_path: Path
) -> None:
    automation.audit.record("note", {"value": 1})
    automation.audit.record("note", {"value": 2})
    path = automation.audit.path
    path.write_text(path.read_text().replace('"value":1', '"value":9'))

    body = client.get("/systematic/audit").json()

    assert body["verification"]["valid"] is False
    assert body["verification"]["broken_at"] == 1


def test_the_finrl_catalog_is_served_even_before_the_sidecar_is_set_up(
    client: TestClient, automation: Automation, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.features.systematic import finrl_lab

    monkeypatch.setattr(finrl_lab, "available", lambda: False)
    body = client.get("/systematic/lab/finrl").json()

    assert body["available"] is False and body["result"] is None
    assert [item["key"] for item in body["catalog"]] == [s.key for s in finrl_lab.CATALOG]
    assert {item["role"] for item in body["catalog"]} == {"used", "excluded", "unavailable"}


def test_running_finrl_without_the_sidecar_is_a_clear_503(
    client: TestClient, automation: Automation, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.features.systematic import finrl_lab

    monkeypatch.setattr(finrl_lab, "available", lambda: False)
    response = client.post("/systematic/lab/finrl")
    assert response.status_code == 503
    assert "poe finrl-setup" in response.json()["detail"]


def test_finrl_failures_map_to_http_errors(
    client: TestClient, automation: Automation, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.features.systematic import finrl_lab

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("Not enough history")

    monkeypatch.setattr(finrl_lab, "run_finrl_lab", fail)
    response = client.post("/systematic/lab/finrl")
    assert (response.status_code, response.json()["detail"]) == (422, "Not enough history")


def test_a_finrl_run_is_returned_remembered_and_audited(
    client: TestClient,
    automation: Automation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.systematic import finrl_lab
    from app.features.systematic.tests import test_finrl_lab as helpers

    closes = helpers.market()

    def ohlcv(_s: object, _c: object, symbols: list[str], *_: object) -> object:
        base = closes if symbols == helpers.SPEC.universe else helpers.market(900, 9, symbols)
        frame = base.reset_index(names="date").melt("date", var_name="tic", value_name="close")
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        for column in ("open", "high", "low"):
            frame[column] = frame["close"]
        frame["volume"] = 1
        return frame

    real = finrl_lab.run_finrl_lab
    monkeypatch.setattr(history, "ohlcv", ohlcv)
    monkeypatch.setattr(finrl_lab, "available", lambda: True)
    monkeypatch.setattr(
        finrl_lab,
        "run_finrl_lab",
        lambda stocks, crypto, log: real(
            stocks, crypto, log, helpers.CONFIG, workers=1, run=helpers.fake_sidecar(closes)
        ),
    )

    result = client.post("/systematic/lab/finrl").json()

    assert {g["name"] for g in result["groups"]} >= {"FinRL A2C", "FinRL ensemble"}
    assert client.get("/systematic/lab/finrl").json()["result"] == result
    assert client.get("/systematic/audit").json()["entries"][0]["kind"] == "finrl"


def test_a_trend_rule_is_checked_by_the_event_driven_engine_and_logged(
    client: TestClient, automation: Automation, market: dict[str, str]
) -> None:
    from app.main import app as main_app

    main_app.state.automation = automation
    body = client.post("/systematic/validate/halal-etf-trend").json()
    assert body["strategy"] == "halal-etf-trend"
    assert body["orders"] > 0 and abs(body["cagr_gap"]) < 0.05
    assert [e.kind for e in automation.audit.entries()][-1] == "event-check"
    refused = client.post("/systematic/validate/halal-etf-rotation")
    assert refused.status_code == 422


def test_a_strategys_stats_signals_and_capital_are_served(
    client: TestClient, automation: Automation, market: dict[str, str]
) -> None:
    automation.configure(AutomationConfig(mode="dry-run", strategies=["halal-etf-trend"]))
    automation.run()
    body = client.get("/systematic/strategies/halal-etf-trend/stats").json()
    assert body["stats"]["trades"] >= 1
    assert body["capital"] > 0
    assert body["signals"][0]["weights"] == {"SPUS": 0.5, "HLAL": 0.5}


def test_signal_feedback_is_kept_in_the_audit_log(
    client: TestClient, automation: Automation
) -> None:
    verdict = {"algorithm": "ppo", "decided": "2026-11-05", "verdict": "flag", "note": "odd"}
    assert client.post("/systematic/signals/feedback", json=verdict).json() == verdict
    assert client.get("/systematic/signals/feedback").json() == [verdict]
    assert (
        client.post(
            "/systematic/signals/feedback", json={**verdict, "verdict": "maybe"}
        ).status_code
        == 422
    )


def test_a_backtest_is_queued_listed_and_read_back(
    client: TestClient, automation: Automation, market: dict[str, str], tmp_path: Path
) -> None:
    from app.features.systematic import runs
    from app.features.systematic.router import get_backtests

    desk = runs.BacktestDesk(
        app.dependency_overrides[get_stock_data_client](),
        app.dependency_overrides[get_crypto_data_client](),
        automation.audit,
        tmp_path / "runs.json",
    )
    app.dependency_overrides[get_backtests] = lambda: desk
    body = {
        "family": "halal-etf-trend",
        "years": 3,
        "window": {"start": 200, "stop": 250, "step": 50},
    }
    queued = client.post("/systematic/backtests", json=body)
    assert queued.status_code == 202
    desk.pool.shutdown(wait=True)
    run_id = queued.json()["id"]
    assert [item["id"] for item in client.get("/systematic/backtests").json()] == [run_id]
    assert client.get(f"/systematic/backtests/{run_id}").json()["status"] == "complete"
    assert client.get("/systematic/backtests/nope").status_code == 404
    refused = client.post("/systematic/backtests", json={"family": "crypto-donchian"})
    assert refused.status_code == 422


def test_a_name_no_strategy_has_is_refused_everywhere_it_can_be_given(
    client: TestClient, automation: Automation, market: dict[str, str]
) -> None:
    assert client.get("/systematic/strategies/martingale/stats").status_code == 404
    settings = client.put("/systematic/automation", json={"strategies": ["martingale"]})
    assert settings.status_code == 422 and "martingale" in settings.json()["detail"]
    from pydantic import ValidationError

    from app.features.systematic import runs

    with pytest.raises(ValidationError, match="No strategy named martingale"):
        runs.BacktestRequest(family="martingale")
