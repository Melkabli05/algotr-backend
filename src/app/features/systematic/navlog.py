import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from pydantic import BaseModel

PATH = Path("data/strategy_nav.jsonl")


class TrackPoint(BaseModel):
    day: date
    paper: float
    model: float | None


class Tracking(BaseModel):
    name: str
    since: date | None
    points: list[TrackPoint]
    gap: float | None
    note: str


class NavLog:
    """The value index of each strategy's paper book, one reading per day, kept for good."""

    def __init__(self, path: Path = PATH) -> None:
        self.path = path
        self.lock = Lock()

    def readings(self) -> dict[str, dict[date, float]]:
        found: dict[str, dict[date, float]] = {}
        if not self.path.exists():
            return found
        for line in self.path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            found.setdefault(row["name"], {})[date.fromisoformat(row["day"])] = float(row["nav"])
        return found

    def record(self, books: Mapping[str, Any], today: date) -> None:
        """Keep today's NAV of every book; a later reading on the same day replaces it."""
        if not books:
            return
        with self.lock:
            kept = [
                line
                for line in (self.path.read_text().splitlines() if self.path.exists() else [])
                if line and json.loads(line)["day"] != today.isoformat()
            ]
            kept += [
                json.dumps({"day": today.isoformat(), "name": name, "nav": float(book.nav)})
                for name, book in books.items()
            ]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("\n".join(kept) + "\n")


def tracking(name: str, paper: Mapping[date, float], model_returns: pd.Series) -> Tracking:
    """Paper NAV against the rule's own backtest over the same days, both from 1 at the start."""
    if not paper:
        return Tracking(
            name=name,
            since=None,
            points=[],
            gap=None,
            note="No paper record yet: the log starts the first day the strategy holds a book",
        )
    days = sorted(paper)
    first = paper[days[0]]
    returns = model_returns[model_returns.index > pd.Timestamp(days[0])]
    model = (1 + returns).cumprod()
    points = []
    for day in days:
        upto = model[model.index <= pd.Timestamp(day)]
        points.append(
            TrackPoint(
                day=day,
                paper=paper[day] / first if first else 1.0,
                model=float(upto.iloc[-1]) if not upto.empty else 1.0,
            )
        )
    last = points[-1]
    gap = None if last.model is None else last.paper - last.model
    return Tracking(
        name=name,
        since=days[0],
        points=points,
        gap=gap,
        note="Paper is the strategy's own book marked at real prices; model is its rule "
        "backtested over the same days with the backtest's costs",
    )
