import io
import logging
import re
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd
from pydantic import BaseModel

from app.features.scanner.schemas import ListHealth

logger = logging.getLogger(__name__)

STALE_AFTER = timedelta(days=7)
KEEP_AT_LEAST = 0.5
TICKER = re.compile(r"[A-Z]{1,5}(?:\.[A-Z])?")
TICKER_COLUMNS = ("stockticker", "ticker", "symbol")
DATE_COLUMNS = ("date", "as of date", "asofdate")
USER_AGENT = "algotr-scanner/0.1"


class ListFormatError(ValueError):
    """Raised when a downloaded or supplied list is not a usable holdings file."""


class Source(BaseModel, frozen=True):
    name: str
    url: str | None = None


class Snapshot(BaseModel):
    name: str
    as_of: date
    symbols: list[str]


SPUS = Source(
    name="SPUS", url="https://www.sp-funds.com/wp-content/uploads/data/TidalFG_Holdings_SPUS.csv"
)
HLAL = Source(name="HLAL")
SOURCES = (SPUS, HLAL)


def client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30,
        follow_redirects=True,
        transport=httpx.HTTPTransport(retries=2),
    )


def parse(name: str, text: str, fallback: date) -> Snapshot:
    """Tickers of a holdings CSV with cash, CUSIP-style ids and CVRs dropped."""
    try:
        frame = pd.read_csv(io.StringIO(text), dtype=str)
    except ValueError as exc:
        raise ListFormatError(f"{name}: not a CSV file") from exc
    columns = {str(column).strip().lower(): column for column in frame.columns}
    ticker = next((columns[key] for key in TICKER_COLUMNS if key in columns), None)
    if ticker is None:
        raise ListFormatError(f"{name}: no ticker column")
    stamp = next((columns[key] for key in DATE_COLUMNS if key in columns), None)
    as_of = fallback
    if stamp is not None:
        dates = pd.to_datetime(frame[stamp], errors="coerce")
        if dates.notna().any():
            as_of = dates.max().date()
    tickers = frame[ticker].dropna().str.strip().str.upper().str.replace("/", ".")
    symbols = sorted(set(tickers[tickers.str.fullmatch(TICKER)]))
    if not symbols:
        raise ListFormatError(f"{name}: no tickers")
    return Snapshot(name=name, as_of=as_of, symbols=symbols)


def newest(source: Source, folder: Path) -> tuple[Snapshot | None, list[str]]:
    """The most recent usable snapshot on disk and a note for every file that was not usable."""
    files = [
        *sorted(folder.glob(f"{source.name}-*.csv"))[-1:],
        *folder.glob(f"{source.name}.csv"),
    ]
    found, problems = [], []
    for path in files:
        try:
            fallback = date.fromtimestamp(path.stat().st_mtime)
            found.append(parse(source.name, path.read_text(), fallback))
        except (ListFormatError, UnicodeDecodeError) as exc:
            problems.append(f"{path.name}: {exc}")
    return max(found, key=lambda snapshot: snapshot.as_of, default=None), problems


def current(sources: Sequence[Source], folder: Path) -> list[Snapshot]:
    """The newest usable snapshot of every source that has one."""
    found = (newest(source, folder)[0] for source in sources)
    return [snapshot for snapshot in found if snapshot is not None]


def versions(snapshots: Sequence[Snapshot]) -> list[str]:
    return [f"{snapshot.name} {snapshot.as_of.isoformat()}" for snapshot in snapshots]


def download(source: Source, folder: Path, http: httpx.Client, today: date) -> Snapshot:
    """Fetch a list and keep a dated copy; a non-list or a list that lost most names is refused."""
    assert source.url is not None
    response = http.get(source.url)
    response.raise_for_status()
    snapshot = parse(source.name, response.text, today)
    previous, _ = newest(source, folder)
    if previous and len(snapshot.symbols) < len(previous.symbols) * KEEP_AT_LEAST:
        raise ListFormatError(
            f"{source.name}: {len(snapshot.symbols)} tickers against {len(previous.symbols)} before"
        )
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{source.name}-{snapshot.as_of.isoformat()}.csv"
    scratch = target.with_suffix(".csv.tmp")
    scratch.write_text(response.text)
    scratch.replace(target)
    return snapshot


def collect(
    source: Source, folder: Path, http: httpx.Client | None, today: date
) -> tuple[Snapshot | None, list[str]]:
    """The best list available now: a fresh download when one works, else the last good file."""
    problems: list[str] = []
    if source.url and http is not None:
        try:
            download(source, folder, http, today)
        except (httpx.HTTPError, ListFormatError) as exc:
            logger.warning("Could not download the %s list: %s", source.name, exc)
            problems.append(f"download: {exc}")
    snapshot, unusable = newest(source, folder)
    return snapshot, [*problems, *unusable]


def health(
    name: str,
    snapshot: Snapshot | None,
    problems: list[str],
    tradable: set[str],
    today: date,
) -> ListHealth:
    error = "; ".join(problems) or None
    if snapshot is None:
        return ListHealth(name=name, error=error or "no file")
    age = (today - snapshot.as_of).days
    return ListHealth(
        name=name,
        as_of=snapshot.as_of,
        age_days=age,
        stale=age > STALE_AFTER.days,
        symbols=len(snapshot.symbols),
        unmatched=sorted(set(snapshot.symbols) - tradable),
        error=error,
    )


def listed(snapshots: list[Snapshot], tradable: set[str], today: date) -> set[str]:
    """Names on at least one list that is not stale, limited to what Alpaca can trade."""
    names: set[str] = set()
    for snapshot in snapshots:
        if (today - snapshot.as_of) <= STALE_AFTER:
            names.update(snapshot.symbols)
    return names & tradable
