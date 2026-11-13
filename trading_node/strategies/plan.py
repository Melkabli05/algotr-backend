from decimal import ROUND_DOWN, Decimal

from nautilus_trader.model import (
    BarType,
    CurrencyPair,
    InstrumentId,
)

from trading_node.alpaca_adapter.parsing import AlpacaInstrument

DAILY_BARS = "1-DAY-LAST-EXTERNAL"


def daily_bar_type(instrument_id: InstrumentId) -> BarType:
    return BarType.from_str(f"{instrument_id}-{DAILY_BARS}")


def whole_units(instrument: AlpacaInstrument, quantity: float) -> Decimal:
    """Round down to what the account can hold, so a buy never exceeds the cash."""
    step = instrument.size_increment.as_decimal()
    if isinstance(instrument, CurrencyPair):
        step = max(step, Decimal(1).scaleb(-instrument.base_currency.precision))
    return (Decimal(str(quantity)) / step).to_integral_value(ROUND_DOWN) * step
