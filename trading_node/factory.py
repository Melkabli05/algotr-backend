import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import redis
import sqlalchemy as sa
from nautilus_trader.common import Environment
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.live import LiveExecutionEngineConfig, LiveNode
from nautilus_trader.model import InstrumentId, TraderId
from sqlalchemy.engine import Connection, Engine

from app.core import data_quality, registry
from app.core.data_quality import Finding
from app.core.feature_store import FeatureValue, read_online
from app.core.limits import LimitStore
from app.core.settings import Settings
from app.core.tables import autopilot, strategies, strategy_versions
from app.features.lifecycle.service import CAPITAL, TRADING
from app.features.risk.sectors import SectorMap
from app.features.systematic.catalog import is_crypto
from app.features.systematic.schemas import StrategySpec
from trading_node.actors.arbiter import PortfolioArbiter
from trading_node.actors.market_relay import MarketRelay
from trading_node.actors.watch import Leases, WatchListener
from trading_node.alpaca_adapter.config import AlpacaDataClientConfig, AlpacaExecClientConfig
from trading_node.alpaca_adapter.constants import ALPACA
from trading_node.alpaca_adapter.factories import (
    AlpacaDataClientFactory,
    AlpacaExecutionClientFactory,
)
from trading_node.alpaca_adapter.parsing import instrument_id_of
from trading_node.arbiter import Limits
from trading_node.exec_algorithms.iceberg import IcebergAlgorithm
from trading_node.exec_algorithms.scheduled import ScheduledAlgorithm
from trading_node.state_writer import StateWriter
from trading_node.strategies.commands import CommandStrategy, TradeLimits
from trading_node.strategies.rule import RuleStrategy

TRADER = TraderId("ALGOTR-001")
RUNNING = ("shadow", *TRADING)
DEFAULT_FACTORY = "trading_node.factory:build"


@dataclass(frozen=True)
class Deployed:
    name: str
    stage: str
    version_id: str
    allocation: float
    spec: StrategySpec


def deployed(engine: Engine) -> list[Deployed]:
    """The strategies the node runs: shadow, canary and live, each at its stage's allocation,
    built from the version the lifecycle promoted rather than whatever the definition says now.

    The autopilot's allocation is split evenly between them; a canary trades a tenth of its
    share and a shadow strategy's share is only the notional of its paper book.
    """
    with engine.connect() as connection:
        settings = connection.execute(sa.select(autopilot)).one_or_none()
        rows = connection.execute(
            sa.select(
                strategies.c.name,
                strategies.c.stage,
                strategy_versions.c.id,
                strategy_versions.c.params,
            )
            .join(strategy_versions, strategies.c.version_id == strategy_versions.c.id)
            .where(strategies.c.stage.in_(RUNNING))
            .order_by(strategies.c.name)
        ).all()
    total = 0.0 if settings is None else settings.allocation
    crypto = settings is not None and settings.crypto
    specs = [(row, StrategySpec.model_validate(row.params)) for row in rows]
    known = [(row, spec) for row, spec in specs if crypto or not is_crypto(spec)]
    share = total / len(known) if known else 0.0
    return [
        Deployed(row.name, row.stage, str(row.id), share * CAPITAL.get(row.stage, 1.0), spec)
        for row, spec in known
    ]


def signature(engine: Engine) -> tuple[object, ...]:
    """What the node loaded: its strategies and the production models; a change restarts it."""
    with engine.connect() as connection:
        models = tuple(sorted(registry.productions(connection).items()))
    return (*((d.name, d.stage, d.version_id, d.allocation) for d in deployed(engine)), models)


def limits_of(settings: Settings) -> Callable[[], Limits]:
    store = LimitStore(settings)

    def read() -> Limits:
        found = store.get()
        return Limits(sector_share=found.sector_share, correlation=found.correlation)

    return read


def exec_engine_config() -> LiveExecutionEngineConfig:
    """Full-history startup reconciliation, then continuous checks of in-flight, open and held."""
    return LiveExecutionEngineConfig(
        reconciliation=True,
        inflight_check_interval_ms=2_000,
        inflight_check_threshold_ms=5_000,
        inflight_check_retries=5,
        open_check_interval_secs=30.0,
        position_check_interval_secs=60.0,
    )


def venue_node(settings: Settings, instrument_ids: Sequence[InstrumentId]) -> LiveNode:
    """The node with Alpaca's data and execution clients and continuous reconciliation."""
    provider = InstrumentProviderConfig(load_ids=[str(i) for i in instrument_ids])
    key, secret, paper = settings.alpaca_api_key, settings.alpaca_api_secret, settings.alpaca_paper
    return (
        LiveNode.builder(f"ALGOTR-{settings.alpaca_trading_mode.upper()}", TRADER, Environment.LIVE)
        .with_exec_engine_config(exec_engine_config())
        .add_data_client(
            ALPACA,
            AlpacaDataClientFactory,
            AlpacaDataClientConfig(
                api_key=key,
                api_secret=secret,
                paper=paper,
                feed=settings.data_feed.value,
                instrument_provider=provider,
            ),
        )
        .add_exec_client(
            ALPACA,
            AlpacaExecutionClientFactory,
            AlpacaExecClientConfig(
                api_key=key, api_secret=secret, paper=paper, instrument_provider=provider
            ),
        )
        .build()
    )


def add_platform(
    node: LiveNode, settings: Settings, engine: Engine, writer: StateWriter, name: str
) -> None:
    """Everything the node runs beside the venue clients, the lifecycle's strategies included."""
    found = LimitStore(settings).get()
    trade = TradeLimits(
        order_share=settings.max_order_share,
        price_deviation=settings.max_price_deviation,
        book_imbalance=found.book_imbalance,
    )
    node.add_strategy(
        CommandStrategy(writer, engine, name, trade_limits=trade, mode=settings.alpaca_trading_mode)
    )
    node.add_exec_algorithm(ScheduledAlgorithm())
    node.add_exec_algorithm(IcebergAlgorithm())
    leases = Leases(settings.stream_symbols, limit=settings.watch_limit)
    node.add_actor(
        MarketRelay(
            writer,
            [instrument_id_of(s) for s in settings.stream_symbols],
            leases,
            WatchListener(settings.database_url, leases),
        )
    )
    online = redis.Redis.from_url(settings.redis_url, socket_timeout=2)

    def features(name: str, symbol: str) -> FeatureValue | None:
        try:
            return read_online(online, name, symbol)
        except redis.RedisError:
            return None

    def quality(found: list[Finding]) -> None:
        def work(connection: Connection) -> None:
            data_quality.record(connection, "node", found)

        writer.put(work)

    for item in deployed(engine):
        node.add_strategy(
            RuleStrategy(
                item.spec,
                item.stage,
                item.version_id,
                item.allocation,
                features,
                quality,
            )
        )
    arbiter = PortfolioArbiter(writer, engine, limits_of(settings), SectorMap().sector_of)
    arbiter.watch = lambda: signature(engine)
    node.add_actor(arbiter)


def load(path: str) -> Callable[..., object]:
    """A `module:callable` reference, as the Helm chart's live.factory names one."""
    module, _, attribute = path.partition(":")
    found = getattr(importlib.import_module(module), attribute)
    if not callable(found):
        raise TypeError(f"{path} is not callable")
    return cast(Callable[..., object], found)


def build(
    settings: Settings,
    engine: Engine,
    writer: StateWriter,
    name: str,
    instrument_ids: Sequence[InstrumentId],
) -> LiveNode:
    """The default factory: Alpaca's clients, the platform's components and the lifecycle's
    strategies with their arbiter."""
    deployed_ids = {
        instrument_id_of(symbol) for item in deployed(engine) for symbol in item.spec.universe
    }
    node = venue_node(settings, sorted({*instrument_ids, *deployed_ids}, key=str))
    add_platform(node, settings, engine, writer, name)
    return node
