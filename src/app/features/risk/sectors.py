import io
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from app.core.compliance import SHARIAH_FUNDS

logger = logging.getLogger(__name__)

URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
PATH = Path("data/risk/sectors.csv")
MAX_AGE = timedelta(days=30)
FUND = "Fund"
CRYPTO = "Crypto"
UNKNOWN = "Unclassified"


def parse(text: str) -> dict[str, str]:
    """Each S&P 500 member's GICS sector, tickers written the way Alpaca writes them."""
    frame = pd.read_csv(io.StringIO(text), dtype=str)
    if not {"Symbol", "GICS Sector"} <= set(frame.columns):
        raise ValueError("The constituents file has no Symbol or GICS Sector column")
    return dict(zip(frame["Symbol"].str.replace("-", "."), frame["GICS Sector"], strict=True))


class SectorMap:
    """The GICS sector of a symbol: S&P 500 members from a monthly file, funds and coins by kind."""

    def __init__(self, path: Path = PATH, client: httpx.Client | None = None) -> None:
        self.path = path
        self.client = client
        self.found: dict[str, str] | None = None

    def refresh(self) -> bool:
        """Download the file when it is missing or a month old; a failure keeps the old one."""
        if self.path.exists():
            age = datetime.now(UTC) - datetime.fromtimestamp(self.path.stat().st_mtime, UTC)
            if age < MAX_AGE:
                return False
        try:
            http = self.client or httpx.Client(timeout=30, follow_redirects=True)
            response = http.get(URL)
            response.raise_for_status()
            parse(response.text)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Could not refresh the sector map: %s", exc)
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(response.text)
        self.found = None
        return True

    def sectors(self) -> dict[str, str]:
        if self.found is None:
            self.found = parse(self.path.read_text()) if self.path.exists() else {}
        return self.found

    def sector_of(self, symbol: str) -> str:
        plain = symbol.upper()
        if "/" in plain or (
            plain.endswith("USD") and len(plain) > 5 and plain not in self.sectors()
        ):
            return CRYPTO
        if plain in SHARIAH_FUNDS:
            return FUND
        return self.sectors().get(plain, UNKNOWN)
