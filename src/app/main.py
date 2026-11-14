import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime

import redis
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator

from app import watch
from app.activity import router as activity_router
from app.bridge import node_state
from app.bridge.fanout import MARKET, WATCH, PostgresFanout, announce
from app.core.broker import (
    get_crypto_data_client,
    get_news_client,
    get_stock_data_client,
    get_trading_client,
)
from app.core.cors import allow_origins
from app.core.database import transaction
from app.core.errors import install_error_handlers
from app.core.feature_store import FeatureStore
from app.core.limits import LimitStore
from app.core.models import ensure as ensure_models
from app.core.settings import get_settings
from app.features.admin.router import router as admin_router
from app.features.alerts.node_events import NodeEventAlerts
from app.features.alerts.pager import pager
from app.features.alerts.router import router as alerts_router
from app.features.alerts.service import AlertCenter, Mail
from app.features.book.router import router as book_router
from app.features.book.service import BookDesk
from app.features.breakers.router import router as breakers_router
from app.features.breakers.service import BreakerMonitor, Thresholds
from app.features.compliance.router import router as compliance_router
from app.features.data_quality.router import router as data_quality_router
from app.features.data_quality.service import DataQualityJob, eodhd_reference
from app.features.drift.monitor import DriftMonitor, Rules
from app.features.drift.router import router as drift_router
from app.features.ledger.router import router as ledger_router
from app.features.ledger.service import Ledger
from app.features.lifecycle.router import router as lifecycle_router
from app.features.lifecycle.service import Lifecycle
from app.features.market.health import StreamHealth
from app.features.market.hub import Hub
from app.features.market.node_feed import NodeFeed
from app.features.market.router import router as market_router
from app.features.market.schemas import FILL_EVENTS, OrderUpdate, StreamEvent
from app.features.market.streams import MarketStreams
from app.features.market.watch import Watchlist
from app.features.market.watch import stop as stop_watching
from app.features.models.features import FeaturePipeline, headline_sentiment
from app.features.models.router import register_finbert
from app.features.models.router import router as models_router
from app.features.orders import trail
from app.features.orders.algo import AlgoDesk
from app.features.orders.iceberg import IcebergDesk
from app.features.orders.router import router as orders_router
from app.features.portfolio.router import router as portfolio_router
from app.features.risk.history import RiskRecorder
from app.features.risk.router import risk_router
from app.features.risk.sectors import SectorMap
from app.features.scanner.crypto import CryptoDesk
from app.features.scanner.filings import Edgar, FilingStore
from app.features.scanner.news import NewsFeed, NewsStore
from app.features.scanner.router import router as scanner_router
from app.features.scanner.sentiment import MODEL_DIR as FINBERT_DIR
from app.features.scanner.sentiment import Scorer
from app.features.scanner.service import ROOT, Scanner
from app.features.strategies.router import router as strategies_router
from app.features.systematic.agent_lab import KRONOS_DIR
from app.features.systematic.agents import AgentDesk
from app.features.systematic.automation import Automation
from app.features.systematic.catalog import STRATEGIES, open_registry
from app.features.systematic.model_registry import AgentRegistry
from app.features.systematic.navlog import NavLog
from app.features.systematic.research import StrategyReturns
from app.features.systematic.router import router as systematic_router
from app.features.systematic.runs import BacktestDesk
from app.features.systematic.targets import PlatformLink
from app.features.tca.router import router as tca_router
from app.freshness import router as freshness_router
from app.health import report as health_report
from app.health import router as health_router
from app.quant import history
from app.system_health import Probe, Timing, time_requests
from app.system_health import router as system_health_router


def alert_on_update(app: FastAPI, event: StreamEvent) -> None:
    if isinstance(event, OrderUpdate):
        app.state.alerts.on_update(event)


def nudge_on_fill(app: FastAPI, event: StreamEvent) -> None:
    """The ledger syncs when a fill arrives, as well as on its own timer."""
    if isinstance(event, OrderUpdate) and event.event in FILL_EVENTS:
        app.state.ledger.nudge()


logger = logging.getLogger(__name__)


def node_running() -> bool:
    with transaction() as connection:
        return node_state.health(connection).up


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await asyncio.to_thread(open_registry)
    app.state.hub = Hub()
    app.state.stream_health = StreamHealth()
    app.state.stream_health.install()
    app.state.limits = LimitStore(get_settings())
    app.state.sectors = SectorMap()
    await asyncio.to_thread(app.state.sectors.refresh)
    coins = [symbol for symbol in get_settings().stream_symbols if "/" in symbol]
    settings = get_settings()
    app.state.book = BookDesk(
        coins,
        max_spread_multiple=settings.max_spread_multiple,
        max_slippage_bps=settings.max_slippage_bps,
        slice_slippage_bps=settings.slice_slippage_bps,
        max_imbalance=app.state.limits.get().book_imbalance,
    )
    streams = MarketStreams(
        app.state.hub,
        get_settings(),
        asyncio.get_running_loop(),
        on_book=app.state.book.keeper.handle,
    )
    app.state.book.renew = streams.renew_book
    app.state.streams = streams
    streams.start()
    app.state.book.start()
    app.state.automation = Automation(
        get_settings(),
        get_trading_client(),
        get_stock_data_client(),
        get_crypto_data_client(),
        warm_cache=True,
    )
    app.state.automation.link = PlatformLink()
    app.state.automation.publish()
    app.state.automation.start()
    app.state.audit = app.state.automation.audit
    app.state.backtests = BacktestDesk(
        get_stock_data_client(), get_crypto_data_client(), app.state.automation.audit
    )
    app.state.drift = DriftMonitor(
        Rules(
            action=settings.drift_action,
            threshold=settings.drift_threshold,
            significance=settings.drift_significance,
            scale=settings.drift_scale,
            hold_days=settings.drift_hold_days,
            baseline_days=settings.drift_baseline_days,
        ),
        log=app.state.automation.audit,
    )
    app.state.drift.sources = [StrategyReturns(get_stock_data_client(), get_crypto_data_client())]
    app.state.automation.sizing = app.state.drift
    app.state.agents = AgentDesk(
        get_stock_data_client(), get_crypto_data_client(), log=app.state.automation.audit
    )
    app.state.agents.enabled = settings.ai_agents
    app.state.drift.sources.append(app.state.agents)
    app.state.drift.retrain = app.state.agents.request_retrain
    app.state.automation.agents = app.state.agents
    app.state.agents.registry = AgentRegistry()
    try:
        register_finbert()
    except Exception:
        logger.exception("Could not register FinBERT in the model registry")
    app.state.agents.start()
    app.state.drift.start()
    app.state.lifecycle = Lifecycle(
        lambda spec: history.cached_prices(
            get_stock_data_client(), get_crypto_data_client(), spec.universe
        )
    )
    try:
        app.state.lifecycle.bootstrap(app.state.automation.state.active or [])
        app.state.automation.lifecycle = app.state.lifecycle
        app.state.lifecycle.start()
    except Exception:
        logger.exception("The strategy lifecycle could not start; the autopilot keeps its streaks")
    contact = get_settings().sec_contact
    headlines = NewsStore(ROOT / "news.sqlite")
    app.state.scanner = Scanner(
        get_trading_client(),
        get_stock_data_client(),
        log=app.state.automation.audit,
        news=NewsFeed(get_news_client(), headlines),
        sentiment=Scorer(headlines),
        edgar=Edgar(contact, FilingStore(ROOT / "filings.sqlite")) if contact else None,
        crypto=CryptoDesk(get_crypto_data_client(), get_settings().crypto_coins, ROOT),
    )
    app.state.automation.sleeve = app.state.scanner
    app.state.agents.mood = app.state.scanner
    app.state.scanner.start()
    app.state.icebergs = IcebergDesk(get_trading_client())
    app.state.icebergs.start()
    app.state.algos = AlgoDesk(
        get_trading_client(),
        get_stock_data_client(),
        get_crypto_data_client(),
        get_settings().data_feed,
        guard=app.state.book,
    )
    app.state.algos.start()
    app.state.ledger = Ledger(get_trading_client(), log=app.state.automation.audit)
    app.state.ledger.start()
    app.state.hub.listen(lambda event: nudge_on_fill(app, event))
    settings_now = get_settings()
    app.state.alerts = AlertCenter(
        Mail(
            host=settings_now.smtp_host,
            port=settings_now.smtp_port,
            user=settings_now.smtp_user,
            password=settings_now.smtp_password,
            sender=settings_now.alert_email_from,
            to=settings_now.alert_email_to,
        )
    )
    app.state.alerts.audit = app.state.audit
    app.state.alerts.status = lambda: health_report(app.state)
    app.state.alerts.problems = lambda: watch.connection_problems(app.state, datetime.now(UTC))
    app.state.alerts.ledger = lambda: app.state.ledger.last
    app.state.alerts.daily_returns = watch.daily_returns
    app.state.alerts.overdue = lambda now: watch.overdue_jobs(app.state, now)
    app.state.hub.listen(lambda event: alert_on_update(app, event))
    app.state.hub.listen(app.state.alerts.on_quote)
    app.state.alerts.start()
    app.state.risk_history = RiskRecorder(lambda: watch.risk_snapshot(app.state))
    app.state.risk_history.listen(app.state.alerts.on_risk)
    app.state.navlog = NavLog()
    app.state.risk_history.listen(
        lambda _: app.state.navlog.record(app.state.automation.state.books, date.today())
    )
    app.state.risk_history.start()
    app.state.breakers = BreakerMonitor(
        Thresholds.of(settings_now),
        connection_problems=lambda now: watch.connection_problems(app.state, now),
        stale_feeds=lambda now: watch.stale_feeds(app.state, now),
        reconciliation_ok=lambda: app.state.ledger.last is None or app.state.ledger.last.ok,
        strategy_returns=watch.strategy_returns,
        equity_history=watch.equity_series,
        holdings=lambda: watch.holdings_closes(app.state.sectors.sector_of),
        alert=lambda title, detail: app.state.alerts.raise_alert(
            "system", f"{title}: {detail}", force=True
        ),
        drift_scores=lambda: watch.drift_scores(app.state.drift),
        retrain=app.state.agents.request_retrain,
    )
    app.state.breakers.start()
    if settings_now.pagerduty_routing_key:
        app.state.alerts.pager = pager(settings_now.pagerduty_routing_key)
    app.state.node_alerts = NodeEventAlerts(
        PostgresFanout(settings_now.database_url).listen,
        app.state.alerts.raise_alert,
        app.state.hub.last_price,
    )
    app.state.node_alerts.start()
    app.state.node_feed = NodeFeed(
        node_up=node_running,
        relay=PostgresFanout(settings_now.database_url, MARKET).listen,
        pause=streams.pause_data,
        resume=streams.resume_data,
        hub=app.state.hub,
        on_book=app.state.book.keeper.handle,
    )
    app.state.node_feed.start()

    def announce_watch(symbols: list[str]) -> None:
        if symbols:
            announce(settings_now.database_url, WATCH, {"symbols": symbols})

    app.state.watchlist = Watchlist(
        settings_now.stream_symbols,
        [streams.watch, announce_watch],
        total=settings_now.watch_limit,
    )
    watching = asyncio.create_task(app.state.watchlist.run())
    app.state.features = FeaturePipeline(
        FeatureStore(redis.Redis.from_url(settings_now.redis_url, socket_timeout=2)),
        symbols=lambda: {
            *settings_now.stream_symbols,
            *(symbol for spec in STRATEGIES.values() for symbol in spec.universe),
        },
        closes=lambda names: history.cached_closes(
            get_stock_data_client(), get_crypto_data_client(), names
        ),
        sentiment=headline_sentiment(headlines),
        books=lambda: dict(app.state.book.keeper.books),
    )
    app.state.features.start()
    app.state.data_quality = DataQualityJob(
        symbols=lambda: {
            *settings_now.stream_symbols,
            *(symbol for spec in STRATEGIES.values() for symbol in spec.universe),
        },
        prices=lambda names: history.cached_prices(
            get_stock_data_client(), get_crypto_data_client(), names
        ),
        reference=eodhd_reference(settings_now.eodhd_api_key)
        if settings_now.eodhd_api_key
        else None,
    )
    app.state.data_quality.start()
    fetching = (
        asyncio.create_task(ensure_models({"finbert": FINBERT_DIR, "kronos": KRONOS_DIR}))
        if settings_now.auto_fetch_models
        else None
    )
    yield
    if fetching is not None:
        await stop_watching(fetching)
    await stop_watching(watching)
    await app.state.node_feed.stop()
    app.state.data_quality.stop()
    app.state.features.stop()
    app.state.node_alerts.stop()
    app.state.audit = None
    app.state.ledger.stop()
    app.state.book.stop()
    app.state.drift.stop()
    if getattr(app.state, "breakers", None) is not None:
        app.state.breakers.stop()
    if getattr(app.state, "lifecycle", None) is not None:
        app.state.lifecycle.stop()
    app.state.agents.stop()
    app.state.algos.stop()
    app.state.icebergs.stop()
    app.state.scanner.stop()
    app.state.automation.stop()
    app.state.backtests.stop()
    app.state.alerts.stop()
    app.state.risk_history.stop()
    streams.stop()


app = FastAPI(title="algotr", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
app.state.timing = Timing()
app.state.probe = Probe()
app.middleware("http")(time_requests)

allow_origins(app, get_settings().cors_origins)

install_error_handlers(app)
app.exception_handler(RequestValidationError)(trail.refused_before_routing)
for router in (
    portfolio_router,
    orders_router,
    ledger_router,
    market_router,
    book_router,
    drift_router,
    lifecycle_router,
    strategies_router,
    breakers_router,
    models_router,
    data_quality_router,
    systematic_router,
    scanner_router,
    compliance_router,
    admin_router,
    alerts_router,
    freshness_router,
    system_health_router,
    tca_router,
    risk_router,
    health_router,
    activity_router,
):
    app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
