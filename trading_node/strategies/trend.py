from collections import deque
from decimal import ROUND_DOWN, Decimal
from typing import Any, Self

from nautilus_trader.model import Bar, BarType, InstrumentId, OrderSide, TimeInForce
from nautilus_trader.trading import Strategy, StrategyConfig

from trading_node.alpaca_adapter.parsing import USD


class TrendRule(Strategy):
    """The 200-day trend rule as events: each close, hold every asset above its average.

    The rule decides on a day's closes once every asset's bar has arrived, and sends market
    orders that the venue fills at the next bar's open, the way the autopilot's orders wait for
    the next session. The capital is split equally; each asset buys whole units within its share,
    and a change smaller than `band` of that share is left alone, as the autopilot's is.
    """

    def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
        return super().__new__(cls)

    def __init__(
        self,
        bar_types: list[BarType],
        window: int,
        steps: dict[InstrumentId, Decimal],
        band: float = 0.05,
    ) -> None:
        super().__init__(StrategyConfig(order_id_tag="003"))
        self.bar_types = bar_types
        self.window = window
        self.steps = steps
        self.band = band
        self.closes: dict[InstrumentId, deque[float]] = {
            bar_type.instrument_id: deque(maxlen=window) for bar_type in bar_types
        }
        self.day: dict[InstrumentId, int] = {}
        self.equity: list[tuple[int, float]] = []
        self.orders = 0

    def on_start(self) -> None:
        for bar_type in self.bar_types:
            self.subscribe_bars(bar_type)

    def on_bar(self, bar: Bar) -> None:
        instrument_id = bar.bar_type.instrument_id
        self.closes[instrument_id].append(float(bar.close))
        self.day[instrument_id] = bar.ts_event
        if len(set(self.day.values())) != 1 or len(self.day) != len(self.bar_types):
            return
        self.equity.append((bar.ts_event, self.value()))
        self.rebalance()

    def value(self) -> float:
        """The account's balance plus what its open positions have gained, in dollars."""
        found = self.portfolio.equity(self.bar_types[0].instrument_id.venue)
        money = found.get(USD) if isinstance(found, dict) else found
        return float(money) if money is not None else 0.0

    def rebalance(self) -> None:
        if any(len(closes) < self.window for closes in self.closes.values()):
            return
        capital = self.value()
        share = capital / len(self.closes)
        for instrument_id, closes in self.closes.items():
            above = closes[-1] > sum(closes) / len(closes)
            wanted = Decimal(str(share / closes[-1])) if above else Decimal(0)
            step = self.steps[instrument_id]
            wanted = (wanted / step).to_integral_value(ROUND_DOWN) * step
            held = Decimal(str(self.portfolio.net_position(instrument_id)))
            change = wanted - held
            small = float(abs(change)) * closes[-1] < self.band * share and wanted > 0
            if abs(change) < step or small:
                continue
            instrument = self.cache.instrument(instrument_id)
            assert instrument is not None
            self.submit_order(
                self.order_factory.market(
                    instrument_id,
                    OrderSide.BUY if change > 0 else OrderSide.SELL,
                    instrument.make_qty(float(abs(change))),
                    time_in_force=TimeInForce.GTC,
                )
            )
            self.orders += 1
