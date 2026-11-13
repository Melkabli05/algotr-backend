from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine

from app.core.tables import lifecycle_events, strategies, strategy_decisions
from app.features.lifecycle import gates, service
from app.features.lifecycle.gates import Track
from app.features.lifecycle.router import get_lifecycle
from app.features.lifecycle.service import Lifecycle
from app.features.systematic import signals
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.schemas import StrategyName, StrategySpec
from app.features.systematic.tests.support import downtrend, uptrend
from app.main import app
from app.quant import history

ETF: StrategyName = "halal-etf-trend"


def prices_of(make: Any) -> service.Prices:
    def prices(spec: StrategySpec) -> history.Prices:
        closes = make(spec.universe, 1500)
        return history.Prices(closes, closes.shift(1).fillna(closes))

    return prices


def stage_of(database: Engine, name: str) -> Any:
    with database.connect() as connection:
        return connection.execute(sa.select(strategies).where(strategies.c.name == name)).one()


def put_in(database: Engine, name: str, stage: str, days_ago: int) -> None:
    since = datetime.now(UTC) - timedelta(days=days_ago)
    with database.begin() as connection:
        connection.execute(
            sa.update(strategies)
            .where(strategies.c.name == name)
            .values(stage=stage, stage_since=since)
        )


def test_the_folds_gate_wants_every_fold_above_the_bar() -> None:
    index = pd.date_range("2020-01-01", periods=500)
    steady = Track(
        pd.Series(0.002, index=index) + pd.Series(range(500), index=index) % 2 * 1e-4, 252
    )
    assert gates.research_gate(steady).passed
    flat = Track(pd.Series(0.0, index=index), 252)
    assert not gates.research_gate(flat).passed


def test_divergence_is_half_the_summed_weight_gap_averaged_over_shared_days() -> None:
    day, other = pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-05").date()
    recorded = {day: {"SPUS": 1.0}, other: {"SPUS": 0.5, "HLAL": 0.5}}
    replayed = {day: {"SPUS": 1.0}, other: {"HLAL": 1.0}}
    assert gates.divergence(recorded, replayed) == pytest.approx(0.25)
    assert gates.divergence(recorded, {}) is None


def test_bootstrap_keeps_what_the_autopilot_trades_live_and_starts_the_rest_in_research(
    database: Engine,
) -> None:
    lifecycle = Lifecycle(prices_of(uptrend))
    lifecycle.bootstrap([ETF])
    lifecycle.bootstrap([ETF, "btc-trend-200"])

    states = {state.name: state for state in lifecycle.states()}
    assert set(states) == set(STRATEGIES)
    assert (states[ETF].stage, states[ETF].capital) == ("live", 1.0)
    assert (states["btc-trend-200"].stage, states["btc-trend-200"].capital) == ("research", 0.0)
    assert lifecycle.trading() == [ETF]
    assert {event.actor for event in lifecycle.events()} == {"bootstrap"}


def test_a_rule_that_clears_every_fold_is_validated_and_its_decision_recorded(
    database: Engine,
) -> None:
    lifecycle = Lifecycle(prices_of(uptrend))
    lifecycle.bootstrap([])

    moved = lifecycle.evaluate()

    assert {event.strategy for event in moved} == set(STRATEGIES)
    assert stage_of(database, ETF).stage == "validated"
    [event] = [event for event in moved if event.strategy == ETF]
    assert (event.from_stage, event.to_stage, event.actor) == ("research", "validated", "auto")
    assert event.gates[0].passed
    with database.connect() as connection:
        sessions = connection.execute(
            sa.select(strategy_decisions.c.session).where(strategy_decisions.c.strategy == ETF)
        ).all()
    assert len(sessions) == 1


def test_a_rule_that_fails_its_folds_stays_in_research_and_says_why(database: Engine) -> None:
    lifecycle = Lifecycle(prices_of(downtrend))
    lifecycle.bootstrap([])

    assert lifecycle.evaluate() == []

    state = lifecycle.state(ETF)
    assert state.stage == "research" and state.next_stage == "validated"
    assert not state.gates[0].passed and "Fold Sharpes" in state.gates[0].detail


def test_a_change_of_code_sends_a_live_strategy_back_to_research(
    database: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle = Lifecycle(prices_of(downtrend))
    lifecycle.bootstrap([ETF])
    before = lifecycle.state(ETF).version.id
    monkeypatch.setattr(service, "code_hash", lambda spec: f"changed-{spec.name}")

    lifecycle.evaluate()

    state = lifecycle.state(ETF)
    assert (state.stage, state.capital) == ("research", 0.0)
    assert state.version.id != before
    assert "changed" in lifecycle.events()[0].reason


def reword(**changes: Any) -> None:
    STRATEGIES.specs[ETF] = STRATEGIES[ETF].model_copy(update=changes)


def test_new_wording_keeps_a_live_strategy_live(database: Engine) -> None:
    lifecycle = Lifecycle(prices_of(downtrend))
    lifecycle.bootstrap([ETF])
    before = stage_of(database, ETF)
    reword(description="The same rule, described better")

    lifecycle.evaluate()

    after = stage_of(database, ETF)
    assert (after.stage, after.stage_since) == ("live", before.stage_since)
    assert after.version_id != before.version_id
    assert lifecycle.events()[0].reason == service.REWORDED


def test_a_new_parameter_still_sends_a_live_strategy_back_to_research(database: Engine) -> None:
    lifecycle = Lifecycle(prices_of(downtrend))
    lifecycle.bootstrap([ETF])
    reword(window=150, description="Faster")

    lifecycle.evaluate()

    assert stage_of(database, ETF).stage == "research"


def record_matching(database: Engine, lifecycle: Lifecycle, days: int) -> None:
    spec = STRATEGIES[ETF]
    held = signals.weights(prices_of(uptrend)(spec).closes, spec)
    version = lifecycle.state(ETF).version.id
    with database.begin() as connection:
        for session, weights in service.by_day(held.iloc[-days - 1 : -1]).items():
            connection.execute(
                sa.insert(strategy_decisions).values(
                    strategy=ETF, session=session, version_id=version, weights=weights
                )
            )


def test_a_month_of_decisions_the_backtest_reproduces_moves_it_to_shadow(
    database: Engine,
) -> None:
    lifecycle = Lifecycle(prices_of(uptrend))
    lifecycle.bootstrap([])
    put_in(database, ETF, "validated", 60)
    record_matching(database, lifecycle, 35)

    lifecycle.evaluate()

    assert stage_of(database, ETF).stage == "shadow"


def test_too_few_decisions_keep_it_validated(database: Engine) -> None:
    lifecycle = Lifecycle(prices_of(uptrend))
    lifecycle.bootstrap([])
    put_in(database, ETF, "validated", 60)
    record_matching(database, lifecycle, 10)

    lifecycle.evaluate()

    assert stage_of(database, ETF).stage == "validated"
    assert "of 30 days recorded" in lifecycle.state(ETF).gates[0].detail


def test_a_good_month_in_shadow_starts_a_canary_at_a_tenth_of_the_capital(
    database: Engine,
) -> None:
    lifecycle = Lifecycle(prices_of(uptrend))
    lifecycle.bootstrap([])
    put_in(database, ETF, "shadow", 45)

    lifecycle.evaluate()

    row = stage_of(database, ETF)
    assert (row.stage, row.capital) == ("canary", 0.1)
    assert lifecycle.capital(ETF) == 0.1
    with database.connect() as connection:
        found = Lifecycle.shadow_sharpe(connection, ETF)
    assert found is not None and found > gates.SHADOW_SHARPE


def test_a_canary_that_matches_its_shadow_for_sixty_days_waits_for_approval_to_go_live(
    database: Engine,
) -> None:
    lifecycle = Lifecycle(prices_of(uptrend))
    lifecycle.bootstrap([])
    put_in(database, ETF, "shadow", 200)
    lifecycle.evaluate()
    with database.begin() as connection:
        connection.execute(
            sa.update(lifecycle_events)
            .where(lifecycle_events.c.to_stage == "canary")
            .values(time=datetime.now(UTC) - timedelta(days=90))
        )
    put_in(database, ETF, "canary", 90)

    lifecycle.evaluate()

    assert stage_of(database, ETF).stage == "canary"
    assert lifecycle.state(ETF).awaiting_approval
    lifecycle.approve(ETF, "Reviewed the canary")
    assert (stage_of(database, ETF).stage, lifecycle.capital(ETF)) == ("live", 1.0)
    assert lifecycle.events()[0].actor == "manual"


def test_a_canary_without_a_shadow_sharpe_on_record_never_goes_live(database: Engine) -> None:
    lifecycle = Lifecycle(prices_of(uptrend))
    lifecycle.bootstrap([])
    put_in(database, ETF, "canary", 90)

    lifecycle.evaluate()

    assert stage_of(database, ETF).stage == "canary"
    assert not lifecycle.state(ETF).awaiting_approval
    with pytest.raises(service.ApprovalError, match="has not passed"):
        lifecycle.approve(ETF, "Too early")


@pytest.fixture
def api(database: Engine) -> Lifecycle:
    lifecycle = Lifecycle(prices_of(downtrend))
    lifecycle.bootstrap([ETF])
    app.dependency_overrides[get_lifecycle] = lambda: lifecycle
    return lifecycle


def test_a_person_can_move_a_strategy_with_a_reason(client: TestClient, api: Lifecycle) -> None:
    body = {"stage": "retired", "reason": "Replaced by the rotation"}

    response = client.put(f"/lifecycle/{ETF}", json=body)

    assert response.status_code == 200 and response.json()["stage"] == "retired"
    assert api.trading() == []
    event = api.events()[0]
    assert (event.actor, event.reason) == ("manual", "Replaced by the rotation")
    assert client.put("/lifecycle/nothing", json=body).status_code == 404


def test_the_stages_and_events_are_served(client: TestClient, api: Lifecycle) -> None:
    states = {item["name"]: item for item in client.get("/lifecycle").json()}
    assert states[ETF]["stage"] == "live"
    assert states[ETF]["version"]["engine"] == service.ENGINE
    assert len(client.get("/lifecycle/events").json()) == len(STRATEGIES)


def test_approving_what_does_not_wait_for_approval_is_refused(
    client: TestClient, api: Lifecycle
) -> None:
    response = client.post(f"/lifecycle/{ETF}/approve", json={"reason": "Looks good"})
    assert response.status_code == 422 and "nothing waits for approval" in response.text


def test_decisions_logged_before_a_rewording_still_count(database: Engine) -> None:
    lifecycle = Lifecycle(prices_of(uptrend))
    lifecycle.bootstrap([])
    put_in(database, ETF, "validated", 60)
    record_matching(database, lifecycle, 35)
    reword(title="Shariah ETF trend, 200 days")

    lifecycle.evaluate()
    lifecycle.evaluate()

    assert stage_of(database, ETF).stage == "shadow"
