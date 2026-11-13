from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest
import sqlalchemy as sa
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from fastapi.testclient import TestClient

from app.bridge import commands as bridge_commands
from app.bridge.commands import Command
from app.core import audit, tables
from app.core import database as database_module
from app.core.book import BookGuard
from app.core.broker import get_crypto_data_client, get_stock_data_client, get_trading_client
from app.core.settings import Settings, get_settings
from app.features.market.hub import Hub
from app.features.market.router import get_hub, get_watchlist
from app.features.market.watch import Watchlist
from app.features.orders.algo import AlgoDesk
from app.features.orders.gate import PreTrade
from app.features.orders.router import get_algos, get_gate, get_guard
from app.features.systematic.catalog import STRATEGIES
from app.main import app
from app.quant import proxy


def settings_with(feed: DataFeed) -> Settings:
    return Settings(
        alpaca_api_key="key",
        alpaca_api_secret="secret",
        alpaca_paper=True,
        data_feed=feed,
    )


@pytest.fixture
def trading() -> Iterator[MagicMock]:
    mock = create_autospec(TradingClient, instance=True)
    mock.get_account.return_value = {"cash": "1000000"}
    mock.get_orders.return_value = []
    app.dependency_overrides[get_trading_client] = lambda: mock
    app.dependency_overrides[get_settings] = lambda: settings_with(DataFeed.IEX)
    yield mock
    app.dependency_overrides.clear()


@pytest.fixture
def choose_feed(trading: MagicMock) -> Callable[[DataFeed], None]:
    def choose(feed: DataFeed) -> None:
        app.dependency_overrides[get_settings] = lambda: settings_with(feed)

    return choose


@pytest.fixture
def stock_data() -> Iterator[MagicMock]:
    mock = create_autospec(StockHistoricalDataClient, instance=True)
    app.dependency_overrides[get_stock_data_client] = lambda: mock
    yield mock


@pytest.fixture
def crypto_data() -> Iterator[MagicMock]:
    mock = create_autospec(CryptoHistoricalDataClient, instance=True)
    app.dependency_overrides[get_crypto_data_client] = lambda: mock
    yield mock


@pytest.fixture
def hub() -> Iterator[Hub]:
    stream_hub = Hub()
    app.dependency_overrides[get_hub] = lambda: stream_hub
    yield stream_hub


@pytest.fixture
def watchlist() -> Iterator[Watchlist]:
    watched = Watchlist(["SPY"])
    app.dependency_overrides[get_watchlist] = lambda: watched
    yield watched


@pytest.fixture
def algos() -> MagicMock:
    mock: MagicMock = create_autospec(AlgoDesk, instance=True)
    mock.algos.return_value = []
    app.dependency_overrides[get_algos] = lambda: mock
    return mock


@pytest.fixture
def guard() -> MagicMock:
    mock: MagicMock = create_autospec(BookGuard, instance=True)
    mock.refusal.return_value = None
    mock.max_size.return_value = None
    app.dependency_overrides[get_guard] = lambda: mock
    return mock


@pytest.fixture
def gate() -> MagicMock:
    """Order tests pass the pre-trade gate unless they set a refusal on it."""
    mock: MagicMock = create_autospec(PreTrade, instance=True)
    mock.check.return_value = None
    app.dependency_overrides[get_gate] = lambda: mock
    return mock


@pytest.fixture
def client(
    trading: MagicMock,
    hub: Hub,
    watchlist: Watchlist,
    stock_data: MagicMock,
    crypto_data: MagicMock,
    algos: MagicMock,
    guard: MagicMock,
    gate: MagicMock,
) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def never_the_real_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """A running trading node carries out what the real database holds; tests use their own."""

    def refused() -> sa.engine.Engine:
        raise AssertionError("A test reached the real database; use the `database` fixture")

    monkeypatch.setattr(database_module, "engine", refused)


@pytest.fixture(autouse=True)
def built_in_strategies() -> Iterator[None]:
    """Each test starts from the templates; a strategy one test saves never reaches the next."""
    STRATEGIES.reset()
    yield
    STRATEGIES.reset()


@pytest.fixture(autouse=True)
def audit_log_in_a_temporary_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tests must never write to the real audit log."""
    monkeypatch.setattr(audit, "AUDIT_PATH", tmp_path / "audit.jsonl")


@pytest.fixture(autouse=True)
def no_market_proxy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tests never read the downloaded market history nor reach Ken French's site."""
    monkeypatch.setattr(proxy, "PATH", tmp_path / "proxy" / "us-market.csv")
    monkeypatch.setattr(proxy, "URL", "http://127.0.0.1:9/factors.zip")


def test_database_url() -> str:
    """The test database beside the real one; never the database the app trades from."""
    url = sa.engine.make_url(get_settings().database_url)
    return url.set(database=f"{url.database}_test").render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def migrated_database() -> Iterator[sa.engine.Engine]:
    url = test_database_url()
    config = AlembicConfig("alembic.ini")
    config.attributes["url"] = url
    config.attributes["configure_logger"] = False
    alembic_command.upgrade(config, "head")
    engine = sa.create_engine(url, future=True)
    yield engine
    engine.dispose()


@pytest.fixture
def database(
    migrated_database: sa.engine.Engine, monkeypatch: pytest.MonkeyPatch
) -> Iterator[sa.engine.Engine]:
    """An empty `platform` and `node` schema in the test database, used by `app.core.database`."""
    names = ", ".join(table.fullname for table in reversed(tables.metadata.sorted_tables))
    with migrated_database.begin() as connection:
        connection.execute(sa.text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
    monkeypatch.setattr(database_module, "engine", lambda: migrated_database)
    yield migrated_database


NodeOrderMaker = Callable[..., dict[str, Any]]


@pytest.fixture
def node_order(database: sa.engine.Engine) -> NodeOrderMaker:
    """Writes an order into the node's tables as the trading node would."""

    def write(**fields: Any) -> dict[str, Any]:
        now = datetime.now(UTC)
        row: dict[str, Any] = {
            "client_order_id": "CMD-0001",
            "venue_order_id": "v-1",
            "command_id": None,
            "primary_id": None,
            "exec_algorithm": None,
            "strategy_id": "CommandStrategy-CMD",
            "instrument_id": "MSFT.ALPACA",
            "side": "buy",
            "order_type": "market",
            "time_in_force": "day",
            "quantity": Decimal(5),
            "filled_qty": Decimal(0),
            "avg_px": None,
            "price": None,
            "trigger_price": None,
            "status": "accepted",
            "is_open": True,
            "is_closed": False,
            "created": now,
            "updated": now,
            "last_event": "OrderAccepted",
            "reason": None,
            **fields,
        }
        with database.begin() as connection:
            connection.execute(sa.insert(tables.node_orders).values(**row))
        return row

    return write


@pytest.fixture
def queued(database: sa.engine.Engine) -> Callable[[], list[Command]]:
    """The commands written so far, oldest first."""

    def read() -> list[Command]:
        with database.connect() as connection:
            return list(reversed(bridge_commands.recent(connection)))

    return read


@pytest.fixture
def node_running(database: sa.engine.Engine) -> None:
    """A trading node that is up: its state and a fresh heartbeat."""
    with database.begin() as connection:
        for name, value in (
            ("node", {"state": "running", "node": "test", "mode": "paper"}),
            ("heartbeat", {"open_orders": 0, "waiting": 0}),
        ):
            connection.execute(sa.insert(tables.node_status).values(name=name, value=value))
