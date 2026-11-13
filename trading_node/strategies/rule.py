from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Self

import numpy as np
import pandas as pd
from nautilus_trader.model import InstrumentId
from nautilus_trader.trading import Strategy, StrategyConfig

from app.core import data_quality
from app.core.data_quality import Finding
from app.core.feature_store import FeatureValue
from app.features.systematic import signals
from app.features.systematic.schemas import StrategySpec
from trading_node.alpaca_adapter.parsing import instrument_id_of, symbol_of
from trading_node.arbiter import TARGETS_TOPIC, StrategyTarget
from trading_node.strategies.plan import daily_bar_type

FeatureReader = Callable[[str, str], FeatureValue | None]
MOMENTUM = "momentum_126"
MOMENTUM_WINDOW = 126
DECIDE_AT = timedelta(minutes=5)
HISTORY_WAIT = timedelta(seconds=30)
CLOSES_SENT = 90
EXTRA_DAYS = 30


def next_daily(now: datetime, offset: timedelta) -> datetime:
    """The next time of day `offset` after midnight UTC, strictly after `now`."""
    midnight = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    found = midnight + offset
    return found if found > now else found + timedelta(days=1)


def frame_of(bars: dict[str, list[Any]], before: datetime) -> pd.DataFrame:
    """Daily closes of every symbol, one row per finished session before `before`."""
    columns = {}
    for symbol, found in bars.items():
        rows = {
            pd.Timestamp(bar.ts_event, tz="UTC").normalize(): float(bar.close)
            for bar in found
            if pd.Timestamp(bar.ts_event, tz="UTC").normalize() < pd.Timestamp(before).normalize()
        }
        columns[symbol] = pd.Series(rows).sort_index()
    return pd.DataFrame(columns).sort_index().ffill().dropna()


class RuleStrategy(Strategy):
    """One lifecycle strategy in the node: the same rule code its version was validated with.

    Once a day it reads its symbols' daily bars and sends the arbiter the weights it wants held;
    it never places an order itself.
    """

    def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
        return super().__new__(cls)

    def __init__(
        self,
        spec: StrategySpec,
        stage: str,
        version_id: str,
        allocation: float,
        features: FeatureReader | None = None,
        quality: Callable[[list[Finding]], None] | None = None,
    ) -> None:
        super().__init__(StrategyConfig(order_id_tag=spec.name.replace("-", "_")))
        self.spec = spec
        self.stage = stage
        self.version_id = version_id
        self.allocation = allocation
        self.instrument_ids: list[InstrumentId] = [instrument_id_of(s) for s in spec.universe]
        self.sent: list[StrategyTarget] = []
        self.features = features
        self.quality = quality
        self.source = "closes"

    @property
    def history(self) -> timedelta:
        days = signals.warmup_days(self.spec) + CLOSES_SENT + EXTRA_DAYS
        return timedelta(days=days * 1.6)

    def on_start(self) -> None:
        self.ask()
        now = self.clock.utc_now()
        self.clock.set_time_alert(
            "decide-now", now + HISTORY_WAIT, callback=lambda _: self.decide()
        )
        self.clock.set_timer(
            "decide-daily",
            timedelta(days=1),
            start_time=next_daily(now, DECIDE_AT),
            callback=lambda _: self.ask_then_decide(),
        )

    def ask(self) -> None:
        start = self.clock.utc_now() - self.history
        for instrument_id in self.instrument_ids:
            self.request_bars(daily_bar_type(instrument_id), start=start)

    def ask_then_decide(self) -> None:
        self.ask()
        later = self.clock.utc_now() + HISTORY_WAIT
        self.clock.set_time_alert(
            f"decide-{later.isoformat()}", later, callback=lambda _: self.decide()
        )

    def validated(self, closes: pd.DataFrame) -> pd.DataFrame:
        """The closes with every flagged bar treated as missing, and what was found recorded."""
        now = self.clock.utc_now()
        flags, found = {}, []
        for symbol in closes.columns:
            marked, findings = data_quality.validate(closes[symbol], str(symbol), now)
            flags[str(symbol)] = marked
            found.extend(findings)
        if found and self.quality is not None:
            self.quality(found)
        return data_quality.mask(closes, flags).ffill().dropna()

    def from_features(self, closes: pd.DataFrame) -> pd.Series | None:
        """The momentum rule's choice from the online store's momentum, never recomputed here.

        Only for a rule whose window is the feature's: any other window is a different feature.
        Every symbol's value must be there, unflagged and of the session being decided;
        otherwise the rule runs on its closes as in the backtest.
        """
        if self.features is None or self.spec.signal != "momentum" or self.spec.target_vol:
            return None
        if self.spec.window != MOMENTUM_WINDOW:
            return None
        session = closes.index[-1].date()
        gains = {}
        for symbol in closes.columns:
            found = self.features(MOMENTUM, str(symbol))
            if found is None or not found.usable or found.time.date() != session:
                return None
            gains[str(symbol)] = found.value
        window = self.spec.window
        frame = pd.DataFrame(np.nan, index=range(window + 1), columns=list(gains))
        frame.iloc[0] = 1.0
        frame.iloc[-1] = [1.0 + gains[name] for name in frame.columns]
        self.source = "features"
        return signals.momentum_selection(frame.ffill(), window, self.spec.top_n).iloc[-1]

    def bars(self) -> dict[str, list[Any]]:
        return {
            symbol_of(instrument_id): list(self.cache.bars(daily_bar_type(instrument_id)) or [])
            for instrument_id in self.instrument_ids
        }

    def decide(self) -> StrategyTarget | None:
        """The rule's weights for the last finished session, sent to the arbiter."""
        closes = self.validated(frame_of(self.bars(), self.clock.utc_now()))
        if len(closes) <= signals.warmup_days(self.spec):
            self.log.warning(
                f"{self.spec.name} has {len(closes)} sessions; it needs more to decide"
            )
            return None
        weights = self.from_features(closes)
        if weights is None:
            self.source = "closes"
            weights = signals.weights(closes, self.spec).iloc[-1]
        target = StrategyTarget(
            strategy=self.spec.name,
            stage=self.stage,
            allocation=self.allocation,
            weights={str(name): float(value) for name, value in weights.items() if value},
            version_id=self.version_id,
            session=closes.index[-1].date().isoformat(),
            closes={str(name): closes[name].iloc[-CLOSES_SENT:].tolist() for name in closes},
            inputs=self.source,
        )
        self.sent.append(target)
        self.publish_message(TARGETS_TOPIC, target)
        return target
