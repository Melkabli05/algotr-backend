"""Run a trend rule through NautilusTrader's event-driven backtest and write what it did."""

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.common import LoggerConfig, LogLevel
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import (
    AccountType,
    Bar,
    BarType,
    Currency,
    CurrencyPair,
    Equity,
    InstrumentId,
    Money,
    OmsType,
    Price,
    Quantity,
    Symbol,
    TraderId,
)

from trading_node.alpaca_adapter.constants import ALPACA_VENUE
from trading_node.alpaca_adapter.parsing import USD, instrument_id_of
from trading_node.strategies.plan import daily_bar_type
from trading_node.strategies.trend import TrendRule

CASH = 100_000.0
SESSION_END = pd.Timedelta(hours=21)
COIN_SIZE = Decimal("0.000001")
VOLUME = 10**9


def instrument_of(symbol: str) -> Equity | CurrencyPair:
    """An instrument for the backtest: whole shares for a stock, a millionth for a coin."""
    instrument_id = instrument_id_of(symbol)
    if "/" in symbol:
        base, _ = symbol.split("/")
        return CurrencyPair(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            base_currency=Currency.from_str(base),
            quote_currency=USD,
            price_precision=2,
            size_precision=6,
            price_increment=Price.from_str("0.01"),
            size_increment=Quantity.from_str(str(COIN_SIZE)),
            ts_event=0,
            ts_init=0,
        )
    return Equity(
        instrument_id=instrument_id,
        raw_symbol=Symbol(symbol),
        currency=USD,
        price_precision=4,
        price_increment=Price.from_str("0.0001"),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )


def bars_of(instrument: Equity | CurrencyPair, rows: pd.DataFrame) -> list[Bar]:
    """A day's open and close as a bar stamped at the session's end; the bar's range spans both."""
    bar_type = daily_bar_type(instrument.id)
    found = []
    for row in rows.itertuples():
        stamp = (pd.Timestamp(row.date, tz="UTC") + SESSION_END).value
        high, low = max(row.open, row.close), min(row.open, row.close)
        found.append(
            Bar(
                bar_type=bar_type,
                open=instrument.make_price(row.open),
                high=instrument.make_price(high),
                low=instrument.make_price(low),
                close=instrument.make_price(row.close),
                volume=instrument.make_qty(VOLUME),
                ts_event=stamp,
                ts_init=stamp,
            )
        )
    return found


def paid(commissions: object) -> float:
    """The dollars in a fills report's commission cell, a list like ['49.99 USD']."""
    items = commissions if isinstance(commissions, list) else [commissions]
    return sum(float(str(item).strip("[]'").split()[0]) for item in items if str(item).strip("[]"))


def run(prices: pd.DataFrame, window: int, fee_bps: float) -> dict[str, Any]:
    """The rule's daily account value, its orders and its fees over `prices` (date, symbol,
    open, close)."""
    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId("VALIDATOR-001"), logging=LoggerConfig(stdout_level=LogLevel.ERROR)
        )
    )
    fee = Decimal(str(fee_bps)) / Decimal(10_000)
    engine.add_venue(
        venue=ALPACA_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(CASH, USD)],
        default_leverage=Decimal(1),
        fee_model=MakerTakerFeeModel(maker_rate=fee, taker_rate=fee),
    )
    bar_types: list[BarType] = []
    steps: dict[InstrumentId, Decimal] = {}
    for symbol, rows in prices.groupby("symbol"):
        instrument = instrument_of(str(symbol))
        engine.add_instrument(instrument)
        engine.add_data(bars_of(instrument, rows.sort_values("date")))
        bar_types.append(daily_bar_type(instrument.id))
        steps[instrument.id] = COIN_SIZE if "/" in str(symbol) else Decimal(1)
    strategy = TrendRule(bar_types, window, steps)
    engine.add_strategy(strategy)
    engine.run()
    fills = engine.generate_order_fills_report()
    commissions = 0.0 if fills.empty else sum(paid(item) for item in fills["commissions"])
    result = {
        "equity": [
            [pd.Timestamp(stamp, tz="UTC").strftime("%Y-%m-%d"), value]
            for stamp, value in strategy.equity
        ],
        "orders": strategy.orders,
        "fills": 0 if fills.empty else len(fills),
        "commissions": commissions,
    }
    engine.dispose()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True, help="CSV: date, symbol, open, close")
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--fee-bps", type=float, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    found = run(pd.read_csv(args.prices), args.window, args.fee_bps)
    args.result.write_text(json.dumps(found))


if __name__ == "__main__":
    main()
