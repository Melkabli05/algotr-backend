import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from functools import cache

import exchange_calendars as xc
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from app.core.feature_store import ANOMALOUS, GAP, MISSING, STALE
from app.core.tables import daily_bars, data_quality

CROSS_SOURCE = 16
SPIKE_SIGMAS = 8.0
SPIKE_WINDOW = 60
MIN_SPIKE = 0.05
CROSS_TOLERANCE = 0.005
STALE_SESSIONS = 1
GENESIS = "0" * 64
REASONS = {
    GAP: "gap",
    STALE: "stale",
    ANOMALOUS: "anomalous",
    MISSING: "missing",
    CROSS_SOURCE: "cross_source",
}


@dataclass(frozen=True)
class Finding:
    symbol: str
    session: date
    flag: int
    detail: str

    @property
    def reason_code(self) -> str:
        return REASONS[self.flag]


@cache
def exchange() -> xc.ExchangeCalendar:
    return xc.get_calendar("XNYS")


def expected_sessions(symbol: str, start: date, end: date) -> pd.DatetimeIndex:
    """Every session the symbol should have a daily bar for: each day for a coin, each NYSE
    session for a stock or fund."""
    if "/" in symbol:
        return pd.date_range(start, end, freq="D")
    if start > end:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(exchange().sessions_in_range(start, end))


def last_finished(symbol: str, now: datetime) -> date | None:
    """The last session whose daily bar is complete before `now`."""
    today = pd.Timestamp(now).tz_convert("UTC").normalize().tz_localize(None)
    if "/" in symbol:
        previous: date = (today - pd.Timedelta(days=1)).date()
        return previous
    found = expected_sessions(symbol, (today - pd.Timedelta(days=10)).date(), today.date())
    finished: list[date] = [s.date() for s in found if s < today]
    return finished[-1] if finished else None


def validate(
    closes: pd.Series,
    symbol: str,
    now: datetime,
    reference: pd.Series | None = None,
) -> tuple[pd.Series, list[Finding]]:
    """Each expected session's quality flags, and why: missing sessions, a late last bar,
    impossible or extreme moves, and closes another source disagrees with."""
    series = closes.dropna()
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    findings: list[Finding] = []
    if series.empty:
        return pd.Series(dtype=int), findings
    end = last_finished(symbol, now) or series.index[-1].date()
    sessions = expected_sessions(symbol, series.index[0].date(), end)
    flags = pd.Series(0, index=sessions.union(series.index), dtype=int)
    inside = sessions[sessions <= series.index[-1]]
    for day in inside.difference(series.index):
        flags[day] |= GAP
        findings.append(Finding(symbol, day.date(), GAP, "No daily bar for an expected session"))
    late = sessions[sessions > series.index[-1]]
    if len(late) >= STALE_SESSIONS:
        flags[late[-1]] |= STALE
        behind = f"{len(late)} session{'s' if len(late) != 1 else ''}"
        detail = f"The last bar is from {series.index[-1].date()}, {behind} behind"
        findings.append(Finding(symbol, late[-1].date(), STALE, detail))
    for day, value in series[series <= 0].items():
        flags[day] |= ANOMALOUS
        findings.append(Finding(symbol, day.date(), ANOMALOUS, f"Close {value:g} is not positive"))
    moves = np.log(series[series > 0]).diff()
    spread = moves.rolling(SPIKE_WINDOW, min_periods=20).std().shift(1)
    spikes = moves[
        (moves.abs() > SPIKE_SIGMAS * spread) & (moves.abs() > MIN_SPIKE) & spread.notna()
    ]
    for day, move in spikes.items():
        flags[day] |= ANOMALOUS
        detail = f"A {move:+.1%} move is past {SPIKE_SIGMAS:g} times its recent spread"
        findings.append(Finding(symbol, day.date(), ANOMALOUS, detail))
    if reference is not None and not reference.empty:
        other = reference.copy()
        other.index = pd.DatetimeIndex(other.index).tz_localize(None).normalize()
        both = series.index.intersection(other.index)
        gaps = (series[both] / other[both] - 1).abs()
        for day, gap in gaps[gaps > CROSS_TOLERANCE].items():
            flags[day] |= CROSS_SOURCE
            detail = f"Alpaca {series[day]:g} and the second source {other[day]:g} differ {gap:.2%}"
            findings.append(Finding(symbol, day.date(), CROSS_SOURCE, detail))
    return flags.sort_index(), findings


def mask(closes: pd.DataFrame, flags: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Flagged bars become missing: a strategy never reads a bad close as a price."""
    found = closes.copy()
    stamps = pd.DatetimeIndex(found.index).tz_localize(None).normalize()
    for symbol, marked in flags.items():
        if symbol not in found.columns:
            continue
        bad = set(marked[marked != 0].index)
        found.loc[[stamp in bad for stamp in stamps], symbol] = np.nan
    return found


def digest(previous: str, source: str, finding: Finding) -> str:
    body = {
        "previous": previous,
        "source": source,
        "symbol": finding.symbol,
        "session": finding.session.isoformat(),
        "flag": finding.flag,
        "detail": finding.detail,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def record(connection: Connection, source: str, findings: Iterable[Finding]) -> int:
    """Keep each new finding once, chained to the one before it so the record cannot be edited
    without breaking every digest after."""
    connection.execute(sa.text("lock table platform.data_quality in share row exclusive mode"))
    last = connection.execute(
        sa.select(data_quality.c.digest).order_by(data_quality.c.id.desc()).limit(1)
    ).scalar_one_or_none()
    previous = last or GENESIS
    written = 0
    for finding in findings:
        known = connection.execute(
            sa.select(data_quality.c.id).where(
                data_quality.c.source == source,
                data_quality.c.symbol == finding.symbol,
                data_quality.c.session == finding.session,
                data_quality.c.reason_code == finding.reason_code,
            )
        ).first()
        if known is not None:
            continue
        current = digest(previous, source, finding)
        connection.execute(
            sa.insert(data_quality).values(
                source=source,
                symbol=finding.symbol,
                session=finding.session,
                quality_flags=finding.flag,
                reason_code=finding.reason_code,
                detail=finding.detail,
                previous_digest=previous,
                digest=current,
            )
        )
        previous = current
        written += 1
    return written


def verify(connection: Connection) -> bool:
    """Recompute the chain: True when no finding was changed, removed or reordered."""
    previous = GENESIS
    rows = connection.execute(sa.select(data_quality).order_by(data_quality.c.id))
    for row in rows:
        finding = Finding(row.symbol, row.session, row.quality_flags, row.detail)
        if row.previous_digest != previous or digest(previous, row.source, finding) != row.digest:
            return False
        previous = row.digest
    return True


def store_bars(
    connection: Connection,
    symbol: str,
    closes: pd.Series,
    opens: pd.Series | None,
    flags: pd.Series,
    source: str,
) -> None:
    """The offline store backtests read: each session's bar with its quality flags."""
    for stamp, close in closes.dropna().items():
        day = pd.Timestamp(stamp).tz_localize(None).normalize()
        values = {
            "open": None if opens is None else float(opens.get(stamp, np.nan)),
            "close": float(close),
            "source": source,
            "quality_flags": int(flags.get(day, 0)),
        }
        statement = insert(daily_bars).values(symbol=symbol, session=day.date(), **values)
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=[daily_bars.c.symbol, daily_bars.c.session], set_=values
            )
        )
