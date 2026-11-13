import io
import logging
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
PATH = Path("data/proxy/us-market.csv")
MAX_AGE = timedelta(days=35)
USER_AGENT = "algotr-research/0.1"
NAME = "US market (Fama-French)"


class ProxyFormatError(ValueError):
    """Raised when the downloaded file is not the daily factor table."""


def parse(archive: bytes) -> pd.Series:
    """The US stock market's daily total return from Ken French's daily factor file.

    The market's return is the excess return over T-bills plus the T-bill rate, both in percent.
    """
    with zipfile.ZipFile(io.BytesIO(archive)) as found:
        text = found.read(found.namelist()[0]).decode("latin-1")
    rows = [line for line in text.splitlines() if line[:8].isdigit() and "," in line]
    if not rows:
        raise ProxyFormatError("No daily rows in the factor file")
    table = pd.read_csv(
        io.StringIO("\n".join(rows)), header=None, names=["day", "excess", "smb", "hml", "rf"]
    )
    index = pd.to_datetime(table["day"].astype(str), format="%Y%m%d")
    return pd.Series(((table["excess"] + table["rf"]) / 100).to_numpy(), index=index)


def download(path: Path = PATH, client: httpx.Client | None = None) -> pd.Series:
    """Fetch the factor file and keep the market's returns as a CSV."""
    http = client or httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True
    )
    response = http.get(URL)
    response.raise_for_status()
    returns = parse(response.content)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".csv.tmp")
    returns.rename("return").to_csv(scratch, index_label="day")
    scratch.replace(path)
    return returns


def stale(path: Path = PATH, now: datetime | None = None) -> bool:
    if not path.exists():
        return True
    moment = now or datetime.now(UTC)
    return moment - datetime.fromtimestamp(path.stat().st_mtime, UTC) > MAX_AGE


def refresh(path: Path = PATH) -> bool:
    """Download the file when it is missing or older than `MAX_AGE`; a failure keeps the old one."""
    if not stale(path):
        return False
    try:
        download(path)
    except (httpx.HTTPError, ProxyFormatError, zipfile.BadZipFile, OSError) as exc:
        logger.warning("Could not refresh the market proxy: %s", exc)
        return False
    return True


def market(path: Path | None = None) -> pd.Series | None:
    """The stored daily returns of the US market, or None until they have been downloaded."""
    found = path or PATH
    if not found.exists():
        return None
    table = pd.read_csv(found, index_col="day", parse_dates=True)
    return table["return"]
