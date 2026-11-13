from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.features.portfolio.schemas import Position
from app.features.risk.schemas import Correlation, Exposure, Limit, RiskEvent, Var
from app.quant import risk

WINDOW = 252
MATERIAL = 0.05
MANUAL = "By hand"
TRIGGERS: dict[str, str] = {
    "Daily loss": "today, past the",
    "Position size": "one position may be",
    "Order size against daily volume": "trades on an average day",
    "Sector": "sector",
    "Spread": "a market buy may cross",
    "Book imbalance": "leans",
    "Kill switch": "kill switch",
    "Shariah screen": "Shariah screen",
    "Cash": "does not borrow",
}


def pair(position: Position) -> str:
    """The symbol prices are read under: a coin as BTC/USD."""
    if position.asset_class == "crypto" and "/" not in position.symbol:
        return f"{position.symbol.removesuffix('USD')}/USD"
    return position.symbol


def var_table(returns: pd.Series, equity: float) -> list[Var]:
    rows = []
    for confidence in risk.CONFIDENCES:
        for horizon in risk.HORIZONS:
            found = risk.value_at_risk(returns, confidence, horizon)
            rows.append(
                Var(
                    confidence=confidence,
                    horizon=horizon,
                    dollars=found["historical"] * equity,
                    **found,
                )
            )
    return rows


def grouped(values: Mapping[str, float], equity: float) -> list[Exposure]:
    return sorted(
        (
            Exposure(name=name, value=value, share=value / equity if equity else 0.0)
            for name, value in values.items()
        ),
        key=lambda item: -item.value,
    )


def sector_exposure(positions: Iterable[Position], sector_of: Any, equity: float) -> list[Exposure]:
    totals: dict[str, float] = {}
    for position in positions:
        name = sector_of(pair(position))
        totals[name] = totals.get(name, 0.0) + abs(position.market_value)
    return grouped(totals, equity)


def strategy_exposure(
    positions: list[Position], books: Mapping[str, Any], equity: float
) -> list[Exposure]:
    """What each autopilot book holds at today's prices, and the rest as held by hand."""
    prices = {pair(position): position.current_price for position in positions}
    invested = sum(abs(position.market_value) for position in positions)
    totals: dict[str, float] = {}
    for name, book in books.items():
        value = sum(
            held.qty * prices.get(symbol, held.price) for symbol, held in book.holdings.items()
        )
        if value > 0:
            totals[name] = value
    rest = invested - sum(totals.values())
    if rest > 1:
        totals[MANUAL] = rest
    return grouped(totals, equity)


def correlation(returns: pd.DataFrame, weights: pd.Series) -> Correlation:
    material = [symbol for symbol in weights.index if abs(weights[symbol]) >= MATERIAL / 10]
    matrix = returns[material].corr().fillna(0.0) if material else pd.DataFrame()
    return Correlation(symbols=list(matrix.columns), matrix=matrix.round(3).to_numpy().tolist())


def highest_pair(returns: pd.DataFrame, weights: pd.Series) -> float | None:
    """The highest correlation between two holdings that are each at least 5% of the account."""
    material = [symbol for symbol in weights.index if abs(weights[symbol]) >= MATERIAL]
    if len(material) < 2:
        return None
    matrix = returns[material].corr().to_numpy()
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    return float(np.nanmax(upper)) if upper.size else None


def last_triggered(entries: Iterable[Any]) -> dict[str, datetime]:
    """When each limit last refused an order, from the audit log."""
    found: dict[str, datetime] = {}
    for entry in entries:
        if entry.kind != "order" or entry.payload.get("outcome") != "refused":
            continue
        reason = str(entry.payload.get("reason", ""))
        for name, words in TRIGGERS.items():
            if words.lower() in reason.lower():
                found[name] = datetime.fromisoformat(entry.time)
    return found


def events(entries: Iterable[Any], drift: Iterable[Any]) -> list[RiskEvent]:
    found: list[RiskEvent] = []
    for entry in entries:
        payload = entry.payload
        if entry.kind == "order" and payload.get("outcome") == "refused":
            request = payload.get("request") or {}
            found.append(
                RiskEvent(
                    time=datetime.fromisoformat(entry.time),
                    severity="warning",
                    title=f"Refused {request.get('side', '')} {request.get('symbol', '')}".strip(),
                    detail=payload.get("reason"),
                )
            )
        elif entry.kind == "run" and (payload.get("run") or {}).get("status") == "halted":
            found.append(
                RiskEvent(
                    time=datetime.fromisoformat(entry.time),
                    severity="critical",
                    title="Kill switch: positions closed and autopilot halted",
                    detail=(payload.get("run") or {}).get("detail"),
                )
            )
    for event in drift:
        if event.adverse:
            found.append(
                RiskEvent(
                    time=datetime.combine(event.time, datetime.min.time()).astimezone(),
                    severity="info",
                    title=f"Drift in {event.stream}: {event.kind}",
                    detail=f"distance {event.distance:.2f}, p {event.p_value:.4f}",
                )
            )
    return sorted(found, key=lambda event: event.time, reverse=True)


def limit(
    name: str,
    value: float | None,
    threshold: float,
    unit: str,
    triggered: Mapping[str, datetime],
    scope: Literal["account", "order"] = "account",
) -> Limit:
    used = None if value is None or not threshold else value / threshold
    return Limit(
        name=name,
        value=value,
        threshold=threshold,
        used=used,
        unit=unit,
        scope=scope,
        last_triggered=triggered.get(name),
    )
