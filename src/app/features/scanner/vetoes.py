import json
import sqlite3
from collections import defaultdict
from collections.abc import Collection
from contextlib import closing
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from app.core import audit
from app.features.scanner import mood, news
from app.features.scanner.headlines import Event
from app.features.scanner.news import NewsStore, parse

KIND = "veto-study"
HORIZONS = (1, 5, 21)
MIN_KIND = 30
HIGH = """
SELECT t.symbol AS symbol, a.created AS created, a.tags AS tags, a.kind AS kind,
       a.direction AS direction, a.severity AS severity, a.label AS label
FROM articles a JOIN tagged t ON t.article = a.id
WHERE a.severity = 'high' AND t.symbol IN (SELECT value FROM json_each(?))
"""
NOTES = [
    "A veto is a serious negative headline, or a pending deal, tagged to at most three names, "
    "in the last five days: the rule the scanner applies to a new buy.",
    "Each event is the first session its name was vetoed while the scanner would have considered "
    "it (listed, liquid, trending). The return runs from the next session's open, where an order "
    "placed after the close fills, to the close of the horizon's session.",
    "The comparison is the same session's other considered names that were not vetoed. Excess is "
    "averaged within a session before it is averaged across sessions, since names on one day move "
    "together.",
    "The Shariah lists are today's, and delisted names are missing from the prices: both gaps "
    "flatter the names that stayed, vetoed or not.",
]


class Horizon(BaseModel):
    sessions: int
    events: int
    days: int
    vetoed_return: float
    others_return: float
    mean_excess: float
    t_statistic: float
    share_below: float


class Kind(BaseModel):
    kind: str
    events: int
    excess: dict[int, float]


class VetoStudy(BaseModel):
    start: str
    end: str
    events: int
    names: int
    horizons: list[Horizon]
    kinds: list[Kind]
    notes: list[str]


def onsets(store: NewsStore, members: Collection[str], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Each name's veto events: the session it was first vetoed on, with the kind of event."""
    with closing(store.connect()) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(HIGH, (json.dumps(sorted(members)),)).fetchall()
    found = []
    for row in rows:
        event = Event(
            kind=row["kind"],
            direction=row["direction"],
            severity=row["severity"],
            label=row["label"],
        )
        if event.veto and row["tags"] <= news.MAX_TAGS_FOR_VETO:
            found.append((row["symbol"], row["kind"], parse(row["created"])))
    frame = pd.DataFrame(found, columns=["symbol", "kind", "time"])
    if frame.empty:
        return frame.assign(session=pd.Series(dtype="datetime64[ns]"))
    frame["session"] = [pd.Timestamp(mood.session_of(time)) for time in frame["time"]]
    frame = frame[frame["session"].isin(sessions)]
    return frame.sort_values("time").drop_duplicates(["symbol", "session"])


def active(events: pd.DataFrame, sessions: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """Per session and name whether a veto event is inside its five-day window."""
    cutoff = pd.Series(
        [(day + timedelta(days=1)).tz_localize("UTC") + timedelta(minutes=1) for day in sessions],
        index=sessions,
    )
    flags = pd.DataFrame(False, index=sessions, columns=symbols)
    by_symbol: dict[str, list[datetime]] = defaultdict(list)
    for symbol, time in zip(events["symbol"], events["time"], strict=True):
        by_symbol[symbol].append(time)
    for symbol, times in by_symbol.items():
        for time in times:
            window = (cutoff >= time) & (cutoff < time + news.VETO_WINDOW)
            flags.loc[window.to_numpy(), symbol] = True
    return flags


def horizon_returns(close: pd.DataFrame, opens: pd.DataFrame, sessions: int) -> pd.DataFrame:
    """From the open of the next session to the close `sessions` sessions after the decision."""
    return close.shift(-sessions) / opens.shift(-1) - 1


def study(
    close: pd.DataFrame,
    opens: pd.DataFrame,
    ok: pd.DataFrame,
    store: NewsStore,
    members: Collection[str],
) -> VetoStudy:
    """Whether names the veto blocks went on to do worse than the names it let through."""
    symbols = [symbol for symbol in close.columns if symbol in set(members)]
    sessions = close.index
    events = onsets(store, symbols, sessions)
    considered_then = np.array(
        [bool(ok.at[r.session, r.symbol]) for r in events.itertuples()], dtype=bool
    )
    events = events.loc[considered_then]
    flags = active(events, sessions, symbols)
    considered = ok[symbols] & ~flags
    result = []
    excess_by_kind: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for horizon in HORIZONS:
        moved = horizon_returns(close[symbols], opens[symbols], horizon)
        peers = moved.where(considered).mean(axis=1)
        rows = []
        for row in events.itertuples():
            value = moved.at[row.session, row.symbol]
            peer = peers.at[row.session]
            if pd.notna(value) and pd.notna(peer):
                rows.append((row.session, row.kind, float(value), float(peer)))
        if not rows:
            continue
        frame = pd.DataFrame(rows, columns=["session", "kind", "vetoed", "others"])
        frame["excess"] = frame["vetoed"] - frame["others"]
        for kind, group in frame.groupby("kind"):
            excess_by_kind[str(kind)][horizon] = group["excess"].tolist()
        per_day = frame.groupby("session")["excess"].mean()
        spread = per_day.std(ddof=1)
        result.append(
            Horizon(
                sessions=horizon,
                events=len(frame),
                days=len(per_day),
                vetoed_return=float(frame["vetoed"].mean()),
                others_return=float(frame["others"].mean()),
                mean_excess=float(per_day.mean()),
                t_statistic=float(per_day.mean() / (spread / np.sqrt(len(per_day))))
                if spread > 0
                else 0.0,
                share_below=float((frame["excess"] < 0).mean()),
            )
        )
    kinds = [
        Kind(
            kind=kind,
            events=len(values[HORIZONS[0]]),
            excess={h: float(np.mean(v)) for h, v in sorted(values.items())},
        )
        for kind, values in sorted(excess_by_kind.items())
        if HORIZONS[0] in values and len(values[HORIZONS[0]]) >= MIN_KIND
    ]
    return VetoStudy(
        start=sessions[0].date().isoformat(),
        end=sessions[-1].date().isoformat(),
        events=len(events),
        names=int(events["symbol"].nunique()),
        horizons=result,
        kinds=kinds,
        notes=NOTES,
    )


def latest(log: audit.AuditLog) -> VetoStudy | None:
    """The newest stored study; one written by an incompatible older version reads as none."""
    for entry in reversed(log.entries()):
        if entry.kind == KIND:
            try:
                return VetoStudy.model_validate(entry.payload["result"])
            except ValidationError:
                return None
    return None
