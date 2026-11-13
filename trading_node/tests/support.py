from typing import Any

import pandas as pd
from nautilus_trader.model import Bar, Quantity

from trading_node.alpaca_adapter import parsing
from trading_node.alpaca_adapter.tests import support as alpaca
from trading_node.strategies.plan import daily_bar_type


def instrument_for(symbol: str) -> parsing.AlpacaInstrument:
    instrument = parsing.parse_instrument(alpaca.asset(symbol), 0)
    assert instrument is not None
    return instrument


BTC = instrument_for("BTC/USD")
AAPL = instrument_for("AAPL")


def bars_of(instrument: Any, rows: list[dict[str, object]]) -> list[Bar]:
    """Daily rows as Nautilus bars stamped at the evening of their session."""
    bar_type = daily_bar_type(instrument.id)
    bars = []
    for row in rows:
        nanos = (pd.Timestamp(str(row["t"])[:10], tz="UTC") + pd.Timedelta(hours=21)).value
        price = instrument.make_price(float(row["c"]))  # type: ignore[arg-type]
        bars.append(
            Bar(
                bar_type=bar_type,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=Quantity(float(row["v"]), instrument.size_precision),  # type: ignore[arg-type]
                ts_event=nanos,
                ts_init=nanos,
            )
        )
    return bars


def rows(price: float, days: int = 3) -> list[dict[str, object]]:
    return [
        {"t": f"{day:%Y-%m-%d}T05:00:00Z", "o": price, "h": price, "l": price, "c": price, "v": 1e7}
        for day in pd.bdate_range("2024-01-02", periods=days)
    ]
