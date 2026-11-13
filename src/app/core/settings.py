from functools import lru_cache
from typing import Literal

from alpaca.data.enums import DataFeed
from pydantic_settings import BaseSettings, SettingsConfigDict

COINS = ["BTC", "ETH", "SOL", "XRP", "LINK", "AVAX", "DOT", "LTC", "BCH", "UNI"]


class Settings(BaseSettings):
    """Application settings loaded from the environment and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    alpaca_api_key: str
    alpaca_api_secret: str
    alpaca_paper: bool = True
    alpaca_trading_mode: Literal["paper", "live"] = "paper"
    node_mode: Literal["validate", "live"] = "validate"
    node_factory: str = "trading_node.factory:build"
    database_url: str = "postgresql+psycopg://localhost:5432/algotr"
    data_feed: Literal[DataFeed.IEX, DataFeed.SIP] = DataFeed.IEX
    stream_symbols: list[str] = ["SPY", "AAPL", "MSFT", "BTC/USD", "ETH/USD"]
    watch_limit: int = 20
    max_spread_multiple: float = 3.0
    max_slippage_bps: float = 100.0
    slice_slippage_bps: float = 20.0
    max_book_imbalance: float = 0.6
    max_order_share: float = 0.25
    strategy_max_drawdown: float = 0.15
    strategy_min_sharpe: float = -0.5
    portfolio_max_drawdown: float = 0.10
    portfolio_var_limit: float = 0.05
    portfolio_max_correlation: float = 0.8
    heartbeat_stale_seconds: int = 60
    pagerduty_routing_key: str | None = None
    node_metrics_port: int = 9101
    redis_url: str = "redis://localhost:6379/0"
    eodhd_api_key: str | None = None
    celery_broker_url: str = "redis://localhost:6379/1"
    max_price_deviation: float = 0.10
    max_position_share: float = 0.25
    daily_loss_limit: float = 0.03
    max_adv_share: float = 0.01
    max_stock_spread_bps: float = 50.0
    max_sector_share: float = 0.40
    max_correlation: float = 0.90
    drift_action: Literal["off", "log", "act"] = "act"
    drift_threshold: float = 0.1
    drift_significance: float = 0.001
    drift_scale: float = 0.5
    drift_hold_days: int = 30
    drift_baseline_days: int = 500
    ai_agents: bool = False
    sec_contact: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    alert_email_from: str | None = None
    alert_email_to: str | None = None
    crypto_coins: list[str] = COINS


@lru_cache
def get_settings() -> Settings:
    return Settings()
