from collections.abc import Iterable, Mapping
from typing import Any

from prometheus_client import Counter, Gauge

STATES = {"closed": 0, "half_open": 1, "open": 2}
STAGES = ("research", "validated", "shadow", "canary", "live", "retired")

BREAKER_STATE = Gauge(
    "algotr_breaker_state", "Circuit breaker: 0 closed, 1 half-open, 2 open", ["name", "level"]
)
KILL_SWITCH = Gauge("algotr_kill_switch_engaged", "1 while a kill switch is engaged", ["name"])
STRATEGY_STAGE = Gauge(
    "algotr_strategy_stage", "1 for the lifecycle stage a strategy is in", ["strategy", "stage"]
)
STRATEGY_SHARPE = Gauge(
    "algotr_strategy_sharpe", "Rolling annualised Sharpe of a strategy", ["strategy", "window"]
)
DRIFT_SCORE = Gauge("algotr_drift_score", "Latest drift statistic of a watched stream", ["stream"])
ALERTS = Counter("algotr_alerts", "Alerts raised", ["kind", "tier"])
ORDER_REJECTIONS = Counter("algotr_order_rejections", "Orders the node reported refused")


def breakers(found: Iterable[Any]) -> None:
    for breaker in found:
        BREAKER_STATE.labels(breaker.name, breaker.level).set(STATES[breaker.state])


def switches(found: Iterable[Any]) -> None:
    for switch in found:
        KILL_SWITCH.labels(switch.name).set(1 if switch.engaged else 0)


def stages(found: Mapping[str, str]) -> None:
    for name, stage in found.items():
        for each in STAGES:
            STRATEGY_STAGE.labels(name, each).set(1 if each == stage else 0)


def sharpes(name: str, windows: Mapping[str, float]) -> None:
    for window, value in windows.items():
        STRATEGY_SHARPE.labels(name, window).set(value)


def drift(scores: Mapping[str, float]) -> None:
    for stream, value in scores.items():
        DRIFT_SCORE.labels(stream).set(value)
