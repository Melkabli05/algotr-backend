from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine

from app.bridge import commands as bridge
from app.core import breakers
from app.core.tables import autopilot, node_positions, node_reconciliation, node_status
from app.features.breakers.service import (
    BreakerMonitor,
    Thresholds,
    portfolio_problem,
    strategy_problem,
)

NOW = datetime(2026, 11, 9, 15, tzinfo=UTC)
LIMITS = Thresholds(
    strategy_drawdown=0.15,
    strategy_sharpe=-0.5,
    portfolio_drawdown=0.10,
    portfolio_var=0.05,
    portfolio_correlation=0.8,
    heartbeat_seconds=60,
)


def monitor(problems: list[str] | None = None, ledger_ok: bool = True) -> BreakerMonitor:
    return BreakerMonitor(
        LIMITS,
        connection_problems=lambda now: problems or [],
        stale_feeds=lambda now: [],
        reconciliation_ok=lambda: ledger_ok,
        strategy_returns=lambda name, since: None,
        equity_history=lambda: pd.Series(dtype=float),
        holdings=pd.DataFrame,
    )


def test_a_breaker_opens_cools_down_half_opens_and_one_probe_decides(database: Engine) -> None:
    name = breakers.strategy_breaker("halal-etf-trend")
    with database.begin() as connection:
        opened = breakers.trip(connection, name, "strategy", "Drawdown 20%", NOW)
        assert (opened.state, opened.retry_at) == ("open", NOW + timedelta(hours=24))
        assert breakers.paused(connection) == {"halal-etf-trend": "Drawdown 20%"}
        assert breakers.due(connection, NOW + timedelta(hours=1)) == []
        [half] = breakers.due(connection, NOW + timedelta(hours=24))
        assert half.state == "half_open" and breakers.paused(connection) == {}
        again = breakers.probe(connection, name, False, "Still down", NOW + timedelta(hours=24))
        assert again.state == "open" and again.cooldown_seconds == 2 * 86400
        breakers.due(connection, NOW + timedelta(hours=72))
        closed = breakers.probe(connection, name, True, "Recovered", NOW + timedelta(hours=72))
        assert closed.state == "closed"


def test_portfolio_and_system_breakers_only_close_by_hand(database: Engine) -> None:
    with database.begin() as connection:
        found = breakers.trip(connection, breakers.SYSTEM, "system", "Heartbeat", NOW)
        assert found.retry_at is None
        assert breakers.due(connection, NOW + timedelta(days=30)) == []
        assert breakers.stopped(connection) == "System breaker: Heartbeat"
        breakers.reset(connection, breakers.SYSTEM, "Checked the node", NOW)
        assert breakers.stopped(connection) is None


def test_the_global_switch_overrides_everything_and_survives_as_a_row(database: Engine) -> None:
    with database.begin() as connection:
        breakers.set_switch(connection, "strategy:a", "strategy", True, "odd fills", "manual", NOW)
        breakers.set_switch(connection, "global", "global", True, "storm", "manual", NOW)
    with database.connect() as connection:
        assert breakers.stopped(connection) == "Global kill switch: storm"
        assert breakers.paused(connection) == {"a": "odd fills"}


def test_a_strategy_pauses_on_its_drawdown_or_a_month_of_poor_sharpe() -> None:
    falling = pd.Series([0.01] * 10 + [-0.03] * 10)
    assert "Drawdown" in (strategy_problem(falling, 252, LIMITS) or "")
    rng = np.random.default_rng(3)
    poor = pd.Series(rng.normal(-0.002, 0.004, 40))
    assert "Sharpe over 30 days" in (strategy_problem(poor, 252, LIMITS) or "")
    assert strategy_problem(pd.Series([0.001, 0.002] * 20), 252, LIMITS) is None


def test_the_portfolio_trips_on_drawdown_var_or_holdings_that_move_together() -> None:
    assert "drawdown" in (
        portfolio_problem(pd.Series([100, 120, 100.0]), pd.DataFrame(), LIMITS) or ""
    )
    rng = np.random.default_rng(4)
    swings = pd.Series(100 * np.cumprod(1 + rng.choice([-0.08, 0.09], 60)))
    swings = swings / swings.cummax() * swings.cummax().iloc[-1]
    found = portfolio_problem(
        swings.iloc[-30:].reset_index(drop=True) * 0 + swings.iloc[-30:].values,
        pd.DataFrame(),
        LIMITS,
    )
    assert found is not None
    common = 100 * np.cumprod(1 + rng.normal(0, 0.01, 60))
    together = pd.DataFrame({"AAPL": common, "MSFT": common * 1.01})
    calm = pd.Series(np.linspace(100, 101, 30))
    assert "move together" in (portfolio_problem(calm, together, LIMITS) or "")


def test_a_silent_node_trips_the_system_breaker_which_halts_and_flattens(
    database: Engine,
) -> None:
    with database.begin() as connection:
        for name, value in (("node", {"state": "running"}), ("heartbeat", {})):
            connection.execute(
                sa.insert(node_status).values(
                    name=name, value=value, updated=NOW - timedelta(minutes=5)
                )
            )
        for symbol, quantity in (("SPUS", 10), ("AAPL", 5)):
            connection.execute(
                sa.insert(node_positions).values(
                    instrument_id=f"{symbol}.ALPACA",
                    side="long",
                    quantity=Decimal(quantity),
                    updated=NOW,
                )
            )
        connection.execute(
            sa.insert(autopilot).values(
                id=1, mode="paper", allocation=0.5, crypto=False, managed=["SPUS", "HLAL"]
            )
        )
    with database.begin() as connection:
        tripped = monitor().check_system(connection, NOW)
    assert tripped is not None and "heartbeat missed" in (tripped.reason or "")
    with database.connect() as connection:
        made = [(c.kind, c.payload, c.source) for c in reversed(bridge.recent(connection))]
        halted = bridge.halted(connection)
    expires = (NOW + timedelta(minutes=2)).isoformat()
    assert made == [
        (
            "rebalance",
            {
                "orders": [{"symbol": "SPUS", "side": "sell", "qty": 10.0}],
                "cancel": ["HLAL", "SPUS"],
                "expires_at": expires,
            },
            "breaker",
        )
    ]
    assert halted[0] and (halted[1] or "").startswith("System breaker")


def test_a_node_stopped_cleanly_is_no_system_problem(database: Engine) -> None:
    with database.begin() as connection:
        connection.execute(
            sa.insert(node_status).values(
                name="node", value={"state": "stopped"}, updated=NOW - timedelta(hours=5)
            )
        )
        assert monitor().check_system(connection, NOW) is None


def test_a_disconnect_must_last_a_minute_and_reconciliation_drift_trips_at_once(
    database: Engine,
) -> None:
    watcher = monitor(problems=["The account stream is disconnected"])
    with database.begin() as connection:
        assert watcher.check_system(connection, NOW) is None
        assert watcher.check_system(connection, NOW + timedelta(seconds=61)) is not None
    with database.begin() as connection:
        breakers.reset(connection, breakers.SYSTEM, "ok", NOW)
        connection.execute(
            sa.insert(node_reconciliation).values(
                kind="policy_resolution", severity="critical", detail="lost order", time=NOW
            )
        )
        found = monitor().check_system(connection, NOW + timedelta(minutes=1))
    assert found is not None and "Reconciliation drift: lost order" in (found.reason or "")


def test_brief_wakes_from_sleep_never_add_up_to_a_disconnect(database: Engine) -> None:
    watcher = monitor(problems=["The account stream is disconnected"])
    with database.begin() as connection:
        assert watcher.check_system(connection, NOW) is None
        assert watcher.check_system(connection, NOW + timedelta(seconds=30)) is None
        later = NOW + timedelta(hours=1)
        assert watcher.check_system(connection, later) is None
        assert watcher.check_system(connection, later + timedelta(seconds=30)) is None
        assert watcher.check_system(connection, later + timedelta(seconds=60)) is not None


def test_switches_and_resets_through_the_api(client: TestClient, database: Engine) -> None:
    body = {"engaged": True, "reason": "Testing the switch"}
    assert client.put("/breakers/switches/global", json=body).status_code == 200
    overview = client.get("/breakers").json()
    assert overview["stopped"] == "Global kill switch: Testing the switch"
    assert client.put("/breakers/switches/nothing", json=body).status_code == 422
    release = {"engaged": False, "reason": "All clear"}
    client.put("/breakers/switches/global", json=release)
    with database.connect() as connection:
        assert bridge.halted(connection) == (False, None)
    assert client.post("/breakers/system/reset", json={"reason": "ok now"}).status_code == 404


@pytest.mark.parametrize(
    ("payload", "last", "found"),
    [
        (
            {"side": "buy", "qty": 10, "type": "limit", "limit_price": 120},
            100.0,
            "20% from the last",
        ),
        ({"side": "buy", "qty": 300, "type": "market"}, 100.0, "more than 25% of the account"),
        ({"side": "buy", "qty": 10, "type": "limit", "limit_price": 101}, 100.0, None),
    ],
)
def test_the_trade_breaker_rejects_one_order_by_size_or_price(
    payload: dict[str, object], last: float, found: str | None
) -> None:
    from trading_node.strategies.commands import TradeLimits, trade_problem

    problem = trade_problem(payload, last, 100_000, None, TradeLimits())
    assert (problem is None) if found is None else (found in (problem or ""))


def test_a_kill_switch_engaged_by_hand_sells_every_position(
    client: TestClient, database: Engine
) -> None:
    with database.begin() as connection:
        for symbol in ("SPUS", "AAPL"):
            connection.execute(
                sa.insert(node_positions).values(
                    instrument_id=f"{symbol}.ALPACA", side="long", quantity=Decimal(3), updated=NOW
                )
            )
    body = {"engaged": True, "reason": "Stop everything"}
    assert client.put("/breakers/switches/global", json=body).status_code == 200
    with database.connect() as connection:
        made = {c.kind: c for c in bridge.recent(connection)}
    assert made["cancel_all"].source == "kill_switch"
    sold = made["rebalance"].payload["orders"]
    assert sorted(order["symbol"] for order in sold) == ["AAPL", "SPUS"]
