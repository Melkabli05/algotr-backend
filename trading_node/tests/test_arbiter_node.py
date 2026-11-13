import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import sqlalchemy as sa
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.common import LoggerConfig, LogLevel
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import AccountType, Money, OmsType, TraderId

from app.core.tables import (
    arbiter_runs,
    autopilot,
    compliance,
    order_commands,
    shadow_orders,
    strategies,
    strategy_targets,
    strategy_versions,
)
from app.features.systematic.catalog import TEMPLATES
from trading_node import factory
from trading_node.actors.arbiter import PortfolioArbiter
from trading_node.alpaca_adapter.constants import ALPACA_VENUE
from trading_node.alpaca_adapter.parsing import USD
from trading_node.arbiter import Limits
from trading_node.state_writer import StateWriter
from trading_node.strategies.commands import CommandStrategy
from trading_node.strategies.rule import RuleStrategy
from trading_node.tests.support import bars_of, instrument_for

SPUS, HLAL = instrument_for("SPUS"), instrument_for("HLAL")
DAYS = 300


def rising(start: float) -> list[dict[str, object]]:
    days = pd.bdate_range("2024-01-02", periods=DAYS)
    return [
        {"t": f"{day:%Y-%m-%d}T05:00:00Z", "c": start * (1.001**i), "v": 1e7}
        for i, day in enumerate(days)
    ]


def stage(database: sa.engine.Engine, name: str, where: str, allocation: float = 0.5) -> str:
    version = uuid.uuid4()
    with database.begin() as connection:
        connection.execute(
            sa.insert(strategy_versions).values(
                id=version,
                strategy=name,
                code_hash=name,
                params=TEMPLATES[name].model_dump(mode="json"),
                engine="test",
            )
        )
        connection.execute(
            sa.insert(strategies).values(
                name=name, stage=where, version_id=version, stage_since=datetime.now(UTC)
            )
        )
        connection.execute(sa.delete(autopilot))
        connection.execute(
            sa.insert(autopilot).values(id=1, mode="paper", allocation=allocation, crypto=False)
        )
    return str(version)


def allow(database: sa.engine.Engine) -> None:
    with database.begin() as connection:
        for symbol in ("SPUS", "HLAL"):
            connection.execute(
                sa.insert(compliance).values(
                    symbol=symbol, allowed=True, reason="fund", as_of=datetime.now(UTC)
                )
            )


def run(database: sa.engine.Engine) -> tuple[PortfolioArbiter, list[Any]]:
    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-007"), logging=LoggerConfig(stdout_level=LogLevel.ERROR)
        )
    )
    engine.add_venue(
        venue=ALPACA_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(100_000, USD)],
        fee_model=MakerTakerFeeModel(maker_rate=Decimal(0), taker_rate=Decimal(0)),
    )
    for instrument, start in ((SPUS, 50.0), (HLAL, 40.0)):
        engine.add_instrument(instrument)
        engine.add_data(bars_of(instrument, rising(start)))
    writer = StateWriter(database)
    writer.start()
    hour = timedelta(hours=1)
    engine.add_strategy(CommandStrategy(writer, database, "test-node", hour, hour))
    rules = []
    for item in factory.deployed(database):
        rule = RuleStrategy(
            item.spec,
            item.stage,
            item.version_id,
            item.allocation,
        )
        rules.append(rule)
        engine.add_strategy(rule)
    arbiter = PortfolioArbiter(
        writer, database, lambda: Limits(sector_share=0.4, correlation=0.8), lambda _: "Fund"
    )
    engine.add_actor(arbiter)
    engine.run()
    writer.flush()
    writer.stop()
    positions = list(engine.cache.positions_open())
    engine.dispose()
    return arbiter, positions


def test_the_factory_deploys_shadow_canary_and_live_strategies_at_their_share(
    database: sa.engine.Engine,
) -> None:
    stage(database, "halal-etf-trend", "live")
    stage(database, "halal-etf-rotation", "canary")
    stage(database, "btc-trend-200", "shadow")
    stage(database, "crypto-donchian", "research")

    found = {item.name: item for item in factory.deployed(database)}

    assert set(found) == {"halal-etf-trend", "halal-etf-rotation"}
    assert found["halal-etf-trend"].allocation == 0.25
    assert found["halal-etf-rotation"].allocation == 0.025


def test_a_live_rule_strategy_reaches_its_target_only_through_the_arbiter(
    database: sa.engine.Engine,
) -> None:
    stage(database, "halal-etf-trend", "live")
    allow(database)

    arbiter, positions = run(database)

    with database.connect() as connection:
        runs = connection.execute(sa.select(arbiter_runs).order_by(arbiter_runs.c.id)).all()
        sources = {
            row.strategy: row.source for row in connection.execute(sa.select(strategy_targets))
        }
        made = connection.execute(sa.select(order_commands.c.source, order_commands.c.status)).all()
    assert runs, "the arbiter never ran"
    last = runs[-1]
    assert last.weights == {"SPUS": 0.25, "HLAL": 0.25}
    assert sources == {"halal-etf-trend": "node"}
    assert {source for source, _ in made} == {"arbiter"}
    held = {str(position.instrument_id): float(position.quantity) for position in positions}
    assert set(held) == {"SPUS.ALPACA", "HLAL.ALPACA"}
    assert arbiter.last is not None


def test_a_shadow_strategy_is_filled_on_paper_and_never_sent(
    database: sa.engine.Engine,
) -> None:
    stage(database, "halal-etf-trend", "shadow")

    _, positions = run(database)

    with database.connect() as connection:
        shadow = connection.execute(sa.select(shadow_orders).order_by(shadow_orders.c.id)).all()
        commands_made = connection.execute(sa.select(order_commands)).all()
    assert {row.symbol for row in shadow} == {"SPUS", "HLAL"}
    first = {row.symbol: row.side for row in reversed(shadow)}
    assert first == {"SPUS": "buy", "HLAL": "buy"}
    assert {row.source for row in shadow} <= {"book", "quote", "last"}
    assert commands_made == []
    assert positions == []


def test_a_global_kill_switch_leaves_the_arbiter_wanting_nothing_held(
    database: sa.engine.Engine,
) -> None:
    from app.core import breakers

    stage(database, "halal-etf-trend", "live")
    allow(database)
    with database.begin() as connection:
        breakers.set_switch(
            connection, "global", "global", True, "drill", "manual", datetime.now(UTC)
        )

    arbiter, positions = run(database)

    assert positions == []
    assert arbiter.last is not None and arbiter.last.weights == {}
    assert "Global kill switch: drill; nothing is held" in arbiter.last.notes
