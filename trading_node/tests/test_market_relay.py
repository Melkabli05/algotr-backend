import json
from decimal import Decimal

import psycopg
import sqlalchemy as sa
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.common import LoggerConfig, LogLevel
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.core import dt_to_unix_nanos
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import (
    AccountType,
    AggressorSide,
    Money,
    OmsType,
    QuoteTick,
    TradeId,
    TraderId,
    TradeTick,
)

from app.bridge.fanout import MARKET, plain_dsn
from trading_node.actors.market_relay import MarketRelay
from trading_node.alpaca_adapter import parsing
from trading_node.alpaca_adapter.constants import ALPACA_VENUE
from trading_node.alpaca_adapter.parsing import USD
from trading_node.alpaca_adapter.tests import support
from trading_node.state_writer import StateWriter
from trading_node.tests.support import BTC

SECOND = 10**9


def market_data() -> list[object]:
    start = dt_to_unix_nanos(support.NOW)
    data: list[object] = []
    snapshot = support.book([(81000.0, 1.5), (80999.0, 2.0)], [(81002.0, 0.75)], reset=True)
    data.append(parsing.parse_book(snapshot, BTC, start))
    for step, bid in enumerate((81000.0, 81000.5, 81001.0)):
        nanos = start + step * SECOND // 10
        data.append(
            QuoteTick(
                instrument_id=BTC.id,
                bid_price=BTC.make_price(bid),
                ask_price=BTC.make_price(81002.0),
                bid_size=BTC.make_qty(1.0),
                ask_size=BTC.make_qty(1.0),
                ts_event=nanos,
                ts_init=nanos,
            )
        )
        data.append(
            TradeTick(
                instrument_id=BTC.id,
                price=BTC.make_price(bid),
                size=BTC.make_qty(0.01),
                aggressor_side=AggressorSide.NO_AGGRESSOR,
                trade_id=TradeId(str(step)),
                ts_event=nanos,
                ts_init=nanos,
            )
        )
    end = start + 2 * SECOND
    data.append(
        QuoteTick(
            instrument_id=BTC.id,
            bid_price=BTC.make_price(81001.0),
            ask_price=BTC.make_price(81002.0),
            bid_size=BTC.make_qty(1.0),
            ask_size=BTC.make_qty(1.0),
            ts_event=end,
            ts_init=end,
        )
    )
    return data


def test_quotes_trades_and_books_are_relayed_on_the_market_channel(
    database: sa.engine.Engine,
) -> None:
    listener = psycopg.connect(
        plain_dsn(database.url.render_as_string(hide_password=False)), autocommit=True
    )
    listener.execute(f"LISTEN {MARKET}")
    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-006"), logging=LoggerConfig(stdout_level=LogLevel.ERROR)
        )
    )
    engine.add_venue(
        venue=ALPACA_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(100_000, USD)],
        fee_model=MakerTakerFeeModel(maker_rate=Decimal(0), taker_rate=Decimal(0)),
    )
    engine.add_instrument(BTC)
    engine.add_data(market_data())
    writer = StateWriter(database)
    writer.start()
    engine.add_actor(MarketRelay(writer, [BTC.id]))
    engine.run()
    writer.flush()
    writer.stop()
    engine.dispose()
    events = [json.loads(note.payload) for note in listener.notifies(timeout=1, stop_after=50)]
    listener.close()

    trades = [event for event in events if event["type"] == "trade"]
    quotes = [event for event in events if event["type"] == "quote"]
    books = [event for event in events if event["type"] == "book"]
    assert [trade["price"] for trade in trades] == [81000.0, 81000.5, 81001.0]
    assert quotes[0]["bid_price"] == 81001.0
    assert len(quotes) < 4
    assert books[0]["symbol"] == "BTC/USD"
    assert books[0]["bids"][0] == [81000.0, 1.5]
    assert books[0]["asks"] == [[81002.0, 0.75]]
    assert Decimal(str(quotes[0]["ask_price"])) == Decimal("81002.0")
