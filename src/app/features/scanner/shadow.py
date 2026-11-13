import numpy as np
import pandas as pd

from app.core import audit
from app.features.scanner import features
from app.features.scanner.schemas import Scan, Shadow, ShadowReport, ShadowSummary

KIND = "scan"
HORIZONS = (1, 5, 21)
ETF = "SPUS"
SHOWN = 60


def logged(log: audit.AuditLog) -> list[audit.Entry]:
    return [entry for entry in log.entries() if entry.kind == KIND]


def record(log: audit.AuditLog, scan: Scan, manifest: dict[str, object]) -> audit.Entry | None:
    """Log the day's picks once per session, without the candidate table, which can be rerun."""
    session = scan.session.isoformat()
    if any(entry.payload["scan"]["session"] == session for entry in logged(log)):
        return None
    payload = {"scan": scan.model_dump(mode="json", exclude={"candidates"}), "manifest": manifest}
    return log.record(KIND, payload)


def forward(close: pd.DataFrame, session: pd.Timestamp, horizon: int) -> pd.Series | None:
    """Each symbol's return from the close of `session` to `horizon` sessions on, if known."""
    if session not in close.index:
        return None
    position = close.index.get_loc(session)
    if not isinstance(position, int) or position + horizon >= len(close):
        return None
    return close.iloc[position + horizon] / close.iloc[position] - 1


def average(values: list[float | None]) -> float | None:
    found = [value for value in values if value is not None]
    return float(np.mean(found)) if found else None


def needed(log: audit.AuditLog) -> set[str]:
    """Every symbol the logged scans refer to, and the ETF they are compared with."""
    symbols = {ETF}
    for entry in logged(log):
        item = entry.payload["scan"]
        symbols.update(item["benchmark"])
        for weights in item["picks"].values():
            symbols.update(weights)
    return symbols


def shadow(log: audit.AuditLog, panel: pd.DataFrame) -> ShadowReport:
    """How each day's picks did afterwards against the same universe and against SPUS.

    Only stored prices are used. A horizon that has not happened yet is left empty.
    """
    scans = [entry.payload["scan"] for entry in logged(log)]
    close = features.wide(panel[panel["symbol"].isin(needed(log))], "close")
    rows: list[Shadow] = []
    for item in scans:
        session = pd.Timestamp(item["session"])
        ahead = {horizon: forward(close, session, horizon) for horizon in HORIZONS}
        for variant, weights in item["picks"].items():
            returns: dict[int, float | None] = {}
            equal: dict[int, float | None] = {}
            etf: dict[int, float | None] = {}
            for horizon, moved in ahead.items():
                if moved is None:
                    returns[horizon] = equal[horizon] = etf[horizon] = None
                    continue
                held = moved.reindex(list(weights)).fillna(0.0)
                mine = float((held * pd.Series(weights)).sum())
                basket = moved.reindex(item["benchmark"]).mean()
                spus = moved.get(ETF, np.nan)
                returns[horizon] = mine
                equal[horizon] = mine - float(basket) if pd.notna(basket) else None
                etf[horizon] = mine - float(spus) if pd.notna(spus) else None
            rows.append(
                Shadow(
                    session=session.date(),
                    variant=variant,
                    picks=len(weights),
                    returns=returns,
                    excess_equal_weight=equal,
                    excess_spus=etf,
                )
            )
    return ShadowReport(horizons=list(HORIZONS), summary=summarise(rows), rows=rows[-SHOWN * 8 :])


def summarise(rows: list[Shadow]) -> list[ShadowSummary]:
    variants = list(dict.fromkeys(row.variant for row in rows))
    summary = []
    for variant in variants:
        mine = [row for row in rows if row.variant == variant]
        summary.append(
            ShadowSummary(
                variant=variant,
                scans=len(mine),
                completed={h: sum(row.returns[h] is not None for row in mine) for h in HORIZONS},
                mean_return={h: average([row.returns[h] for row in mine]) for h in HORIZONS},
                mean_excess_equal_weight={
                    h: average([row.excess_equal_weight[h] for row in mine]) for h in HORIZONS
                },
                mean_excess_spus={
                    h: average([row.excess_spus[h] for row in mine]) for h in HORIZONS
                },
            )
        )
    return summary
