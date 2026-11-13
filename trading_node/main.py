import logging
import os
import socket
from typing import cast

import truststore
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.models import Order, Position, TradeAccount
from alpaca.trading.requests import GetOrdersRequest
from nautilus_trader.live import LiveNode
from nautilus_trader.model import InstrumentId

from app.core.database import engine
from app.core.settings import Settings, get_settings
from trading_node import metrics
from trading_node.alpaca_adapter.parsing import instrument_id_of, normalize_symbol
from trading_node.credentials import ModeMismatchError, check_mode
from trading_node.factory import deployed, load
from trading_node.state_writer import StateWriter

logger = logging.getLogger(__name__)


def known_instruments(trading: TradingClient) -> set[InstrumentId]:
    """What the account holds or has working: reconciliation must be able to price all of it."""
    positions = cast(list[Position], trading.get_all_positions())
    orders = cast(list[Order], trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)))
    held = {normalize_symbol(p.symbol, p.asset_class) for p in positions}
    working = {normalize_symbol(o.symbol or "", o.asset_class) for o in orders if o.asset_class}
    return {instrument_id_of(symbol) for symbol in held | working}


def validate(settings: Settings) -> None:
    """The safe default: load the factory and say what it would deploy, never touch a venue."""
    load(settings.node_factory)
    for item in deployed(engine()):
        logger.info(
            "Would run %s (%s, version %s) at %.1f%% of the account",
            item.name,
            item.stage,
            item.version_id,
            item.allocation * 100,
        )
    logger.info("NODE_MODE=validate: nothing connects or trades; set NODE_MODE=live to run")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    truststore.inject_into_ssl()
    settings = get_settings()
    if settings.node_mode != "live":
        validate(settings)
        return
    trading = TradingClient(
        settings.alpaca_api_key, settings.alpaca_api_secret, paper=settings.alpaca_paper
    )
    try:
        account = cast(TradeAccount, trading.get_account())
        mode, key = settings.alpaca_trading_mode, settings.alpaca_api_key
        check_mode(mode, settings.alpaca_paper, key, account)
    except (APIError, ModeMismatchError) as exc:
        logger.error("The trading node will not start: %s", exc)
        raise SystemExit(2) from exc
    metrics.serve(settings.node_metrics_port)
    name = f"{socket.gethostname()}-{os.getpid()}"
    writer = StateWriter(engine())
    writer.start()
    writer.status(
        "node",
        {"state": "starting", "node": name, "mode": settings.alpaca_trading_mode},
        announce=True,
    )
    streamed = [instrument_id_of(symbol) for symbol in settings.stream_symbols]
    instruments = known_instruments(trading) | set(streamed)
    factory = load(settings.node_factory)
    node = cast(LiveNode, factory(settings, engine(), writer, name, sorted(instruments, key=str)))
    try:
        node.run()
    finally:
        writer.status("node", {"state": "stopped", "node": name}, announce=True)
        writer.flush()
        writer.stop()


if __name__ == "__main__":
    main()
