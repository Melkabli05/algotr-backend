import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.features.scanner.news import NewsStore
from app.features.scanner.sentiment import MODEL

EXCHANGE = ZoneInfo("America/New_York")
CLOSE = time(16, 0)
SCORED = """
SELECT DISTINCT a.id, a.created, c.score, c.risk, c.uncertainty
FROM tagged t
JOIN articles a ON a.id = t.article
JOIN article_sentiment s ON s.article = a.id
JOIN sentiment_cache c ON c.text_hash = s.text_hash AND c.model = ?
WHERE t.symbol IN (SELECT value FROM json_each(?)) AND a.created >= ? AND a.created < ?
"""
COLUMNS = ["sentiment", "news_risk", "headlines"]


def session_of(created: datetime) -> date:
    """The session a headline can first inform: its own day before the close, else the next."""
    local = created.astimezone(EXCHANGE)
    day = local.date()
    return day if local.time() < CLOSE else (pd.Timestamp(day) + pd.Timedelta(days=1)).date()


def gate_weights(spread: pd.Series, tau: float | None) -> pd.Series:
    """The uncertainty gate: a headline whose score spread under dropout is above `tau` is
    suppressed, one below it counts with weight 1 - spread / tau, and one never measured or
    without a calibrated gate counts for nothing."""
    if tau is None or tau <= 0:
        return pd.Series(0.0, index=spread.index)
    values = pd.to_numeric(spread, errors="coerce")
    open_ = values.notna() & (values <= tau)
    return (1 - values / tau).where(open_, 0.0).clip(lower=0.0).fillna(0.0)


def tau_of(news: NewsStore) -> float | None:
    with closing(news.connect()) as connection:
        try:
            row = connection.execute(
                "SELECT tau FROM sentiment_gate WHERE model = ?", (MODEL,)
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    return None if row is None else float(row[0])


def scored(news: NewsStore, symbols: Sequence[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Every scored headline tagged to `symbols`, once each, with the session it informs and
    the weight the uncertainty gate gives its score."""
    with closing(news.connect()) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            SCORED, (MODEL, json.dumps(list(symbols)), start.isoformat(), end.isoformat())
        ).fetchall()
    frame = pd.DataFrame(
        [dict(row) for row in rows], columns=["id", "created", "score", "risk", "uncertainty"]
    )
    frame["weight"] = gate_weights(frame["uncertainty"], tau_of(news))
    frame["session"] = [
        session_of(datetime.fromisoformat(value.replace("Z", "+00:00")))
        for value in frame["created"]
    ]
    return frame


def daily(headlines: pd.DataFrame, sessions: Sequence[date]) -> pd.DataFrame:
    """For each trading session, the gated FinBERT score (weighted by how sure FinBERT was, the
    unsure ones left out) and the mean rule risk (1 to 5) of the headlines that came out since
    the session before it closed.

    A weekend's or holiday's headlines fold into the next session. A session with no headline
    has no value (NaN), which the agents read as neutral; nothing after a session's close is in
    its row.
    """
    days = pd.DatetimeIndex(sorted(pd.Timestamp(day) for day in sessions))
    if headlines.empty or days.empty:
        return pd.DataFrame(np.nan, index=days, columns=COLUMNS)
    position = days.searchsorted(pd.to_datetime(headlines["session"]), side="left")
    inside = position < len(days)
    keyed = headlines.loc[inside].assign(day=days[position[inside]])
    if "weight" not in keyed:
        keyed = keyed.assign(weight=1.0)
    keyed = keyed.assign(weighted=keyed["score"] * keyed["weight"])
    grouped = keyed.groupby("day")
    trusted = grouped["weight"].sum()
    found = pd.DataFrame(
        {
            "sentiment": (grouped["weighted"].sum() / trusted).where(trusted > 0),
            "news_risk": grouped["risk"].mean(),
            "headlines": grouped["id"].count(),
        }
    )
    return found.reindex(days)
