import sqlite3
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.features.systematic.explanations import AgentExplanations
from app.quant.regimes import Regimes
from app.quant.regimes import fit as fit_regimes

MOVE = 0.02
DIRECTIONAL = 0.2
WINDOWS = (30, 90, 365)
SHAP_WINDOWS = (30, 60, 90)
BINS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)
Record = tuple[list[str], list[date], list[list[float]], Mapping[str, list[float]]]
AGENT_BINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)
NEWS_PATH = Path("data/scanner/news.sqlite")
SENTIMENT = """
SELECT t.symbol AS symbol, a.created AS created, a.headline AS headline,
       s.score AS score, s.confidence AS confidence
FROM tagged t
JOIN articles a ON a.id = t.article
JOIN article_sentiment x ON x.article = a.id
JOIN sentiment_cache s ON s.text_hash = x.text_hash
WHERE a.created >= ? AND s.fallback = 0
"""


class Outcome(BaseModel):
    source: str
    day: date
    symbol: str
    direction: str
    confidence: float | None
    detail: str
    realized: float | None
    right: bool | None


class Accuracy(BaseModel):
    source: str
    window: int
    signals: int
    hit_rate: float | None


class RegimeAccuracy(BaseModel):
    source: str
    regime: str
    signals: int
    hit_rate: float | None


class CalibrationBin(BaseModel):
    source: str
    low: float
    high: float
    headlines: int
    hit_rate: float | None


class Importance(BaseModel):
    algorithm: str
    group: str
    windows: dict[int, float | None]
    rank: dict[int, int | None]


class Outcomes(BaseModel):
    regime: "RegimeNow | None" = None
    accuracy: list[Accuracy]
    regimes: list[RegimeAccuracy]
    calibration: list[CalibrationBin]
    importance: list[Importance]
    recent: list[Outcome]
    notes: list[str]


def agent_outcomes(
    records: Mapping[
        str, tuple[list[str], list[date], list[list[float]], Mapping[str, list[float]]]
    ],
    closes: Mapping[str, pd.Series],
) -> list[Outcome]:
    """Each day an agent moved a weight by 2 points or more, against that day's return.

    `applied[i]` is the weights the agent held through `dates[i]`, chosen at the close before,
    so the move from the day before is judged by the asset's return on `dates[i]`. Its
    confidence is the share of the policy's own samples that made the same move that close.
    """
    rows = []
    for name, (assets, dates, applied, sure) in records.items():
        for index in range(1, len(dates)):
            for position, symbol in enumerate(assets):
                if symbol not in closes:
                    continue
                change = applied[index][position] - applied[index - 1][position]
                if abs(change) < MOVE:
                    continue
                series = closes[symbol].dropna()
                day = pd.Timestamp(dates[index])
                earlier = series[series.index < day]
                realized = None
                if day in series.index and not earlier.empty:
                    realized = float(series[day] / earlier.iloc[-1] - 1)
                direction = "Buy" if change > 0 else "Sell"
                shares = sure.get(dates[index - 1].isoformat())
                rows.append(
                    Outcome(
                        source=name.upper(),
                        day=dates[index],
                        symbol=symbol,
                        direction=direction,
                        confidence=shares[position] if shares else None,
                        detail=f"weight {applied[index - 1][position]:.0%} to "
                        f"{applied[index][position]:.0%}",
                        realized=realized,
                        right=None if realized is None else (realized > 0) == (direction == "Buy"),
                    )
                )
    return rows


def news_outcomes(
    path: Path, since: date, read_panel: Callable[[set[str]], pd.DataFrame]
) -> list[Outcome]:
    """Directional headlines against their symbol's next-day return."""
    if not path.exists():
        return []
    with sqlite3.connect(path, timeout=30) as connection:
        frame = pd.read_sql_query(SENTIMENT, connection, params=(since.isoformat(),))
    frame = frame[frame["score"].abs() >= DIRECTIONAL]
    if frame.empty:
        return []
    panel = read_panel(set(frame["symbol"]))
    if panel.empty:
        return []
    frame["day"] = (
        pd.to_datetime(frame["created"], utc=True, format="ISO8601")
        .dt.tz_convert(None)
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    prices = panel[["symbol", "date", "close"]].sort_values(["symbol", "date"]).copy()
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    prices["next"] = prices.groupby("symbol")["close"].shift(-1) / prices["close"] - 1
    merged = pd.merge_asof(
        frame.sort_values("day"),
        prices.rename(columns={"date": "day"}).sort_values("day"),
        on="day",
        by="symbol",
        direction="backward",
        tolerance=pd.Timedelta(days=4),
    )
    rows = []
    for row in merged.itertuples(index=False):
        realized = None if pd.isna(row.next) else float(row.next)
        direction = "Positive" if row.score > 0 else "Negative"
        rows.append(
            Outcome(
                source="FinBERT",
                day=row.day.date(),
                symbol=str(row.symbol),
                direction=direction,
                confidence=float(row.confidence),
                detail=str(row.headline)[:140],
                realized=realized,
                right=None if realized is None else (realized > 0) == (row.score > 0),
            )
        )
    return rows


def accuracy(rows: Sequence[Outcome], today: date) -> list[Accuracy]:
    found = []
    for source in sorted({row.source for row in rows}):
        mine = [row for row in rows if row.source == source and row.right is not None]
        for window in WINDOWS:
            recent = [row for row in mine if row.day >= today - timedelta(days=window)]
            found.append(
                Accuracy(
                    source=source,
                    window=window,
                    signals=len(recent),
                    hit_rate=sum(bool(row.right) for row in recent) / len(recent)
                    if recent
                    else None,
                )
            )
    return found


def by_regime(rows: Sequence[Outcome], labels: pd.Series) -> list[RegimeAccuracy]:
    groups: dict[tuple[str, str], list[bool]] = {}
    for row in rows:
        if row.right is None:
            continue
        stamp = pd.Timestamp(row.day)
        known = labels[labels.index <= stamp]
        if known.empty:
            continue
        groups.setdefault((row.source, str(known.iloc[-1])), []).append(bool(row.right))
    return [
        RegimeAccuracy(
            source=source, regime=regime, signals=len(hits), hit_rate=sum(hits) / len(hits)
        )
        for (source, regime), hits in sorted(groups.items())
    ]


def calibration(rows: Sequence[Outcome]) -> list[CalibrationBin]:
    """Hit rate by stated confidence, per source: a calibrated source sits on the diagonal."""
    scored = [row for row in rows if row.confidence is not None and row.right is not None]
    found = []
    for source in sorted({row.source for row in scored}):
        found += bins_of(source, [row for row in scored if row.source == source])
    return found


class RegimeState(BaseModel):
    label: str
    mean: float
    volatility: float
    expected_days: float
    probability: float


class RegimeNow(BaseModel):
    day: date | None
    label: str | None
    states: list[RegimeState]
    history: list[tuple[date, str]]


def regime_now(found: Regimes, days: int = 120) -> RegimeNow:
    latest = found.probabilities.iloc[-1] if len(found.probabilities) else None
    labels = found.labels()
    return RegimeNow(
        day=found.probabilities.index[-1].date() if latest is not None else None,
        label=str(latest.idxmax()) if latest is not None else None,
        states=[
            RegimeState(
                label=state.label,
                mean=state.mean,
                volatility=state.volatility,
                expected_days=state.expected_days,
                probability=float(latest[state.label]) if latest is not None else 0.0,
            )
            for state in found.states
        ],
        history=[(stamp.date(), str(label)) for stamp, label in labels.tail(days).items()],
    )


def bins_of(source: str, scored: Sequence[Outcome]) -> list[CalibrationBin]:
    edges = BINS if source == "FinBERT" else AGENT_BINS
    found = []
    for low, high in zip(edges, edges[1:], strict=False):
        inside = [row for row in scored if low <= (row.confidence or 0) < high]
        found.append(
            CalibrationBin(
                source=source,
                low=low,
                high=min(high, 1.0),
                headlines=len(inside),
                hit_rate=sum(bool(row.right) for row in inside) / len(inside) if inside else None,
            )
        )
    return found


def ranked(values: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: -item[1])
    return {group: index + 1 for index, (group, _) in enumerate(ordered)}


def importance(found: Sequence[AgentExplanations], today: date) -> list[Importance]:
    """Mean absolute SHAP push of each indicator over the last 30, 60 and 90 days, and its rank."""
    rows = []
    for agent in found:
        means: dict[int, dict[str, float]] = {}
        for window in SHAP_WINDOWS:
            recent = [
                item for item in agent.decisions if item.decided >= today - timedelta(days=window)
            ]
            totals: dict[str, list[float]] = {}
            for decision in recent:
                for group, pushes in decision.values.items():
                    totals.setdefault(group, []).append(
                        sum(abs(value) for asset, value in pushes.items() if asset != "cash")
                    )
            means[window] = {group: float(np.mean(values)) for group, values in totals.items()}
        ranks = {
            window: {
                group: index + 1
                for index, (group, _) in enumerate(
                    sorted(found_.items(), key=lambda item: -item[1])
                )
            }
            for window, found_ in means.items()
        }
        for group in agent.groups:
            rows.append(
                Importance(
                    algorithm=agent.algorithm.upper(),
                    group=group,
                    windows={window: means[window].get(group) for window in SHAP_WINDOWS},
                    rank={window: ranks[window].get(group) for window in SHAP_WINDOWS},
                )
            )
    return rows


def compute(
    found: Sequence[AgentExplanations],
    records: Mapping[
        str, tuple[list[str], list[date], list[list[float]], Mapping[str, list[float]]]
    ],
    closes: Mapping[str, pd.Series],
    market: pd.Series,
    read_panel: Callable[[set[str]], pd.DataFrame],
    today: date,
    news_path: Path = NEWS_PATH,
) -> Outcomes:
    agents_rows = agent_outcomes(records, closes)
    news_rows = news_outcomes(news_path, today - timedelta(days=max(WINDOWS)), read_panel)
    rows = agents_rows + news_rows
    model = fit_regimes(market)
    recent = sorted(rows, key=lambda row: row.day, reverse=True)[:50]
    return Outcomes(
        regime=regime_now(model),
        accuracy=accuracy(rows, today),
        regimes=by_regime(rows, model.labels()),
        calibration=calibration(rows),
        importance=importance(found, today),
        recent=recent,
        notes=[
            "A signal is right when its symbol moved its way over the next trading day.",
            "Agents are judged on their whole paper record: each day's weights against that "
            "day's return.",
            "Agents choose weights; a move of 2 points or more counts as a signal. Its confidence "
            "is the share of 256 draws from the agent's own stochastic policy that make the same "
            "move, recorded at each decision from 7 November 2026 on.",
            "FinBERT counts headlines scoring at least 0.2 either way, against the tagged "
            "symbol's next close.",
            "Regimes come from a three-state hidden Markov model of SPUS daily returns "
            "(statsmodels' Markov-switching regression with a mean and variance per state): the "
            "highest variance is volatile, the higher mean of the others trending, the last "
            "ranging. Each day takes its most likely state from returns up to that day.",
        ],
    )


CACHE: dict[str, tuple[datetime, Any]] = {}
CACHE_SECONDS = 3600


def cached(key: str, make: Callable[[], Outcomes]) -> Outcomes:
    now = datetime.now(UTC)
    found = CACHE.get(key)
    if found and (now - found[0]).total_seconds() < CACHE_SECONDS:
        result: Outcomes = found[1]
        return result
    made = make()
    CACHE[key] = (now, made)
    return made
