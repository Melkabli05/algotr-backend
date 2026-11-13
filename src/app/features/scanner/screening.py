from collections.abc import Collection, Sequence
from datetime import date

from app.core.compliance import SHARIAH_FUNDS, Screening
from app.features.scanner import lists
from app.features.scanner.lists import Snapshot

CRYPTO_NOTE = "a listed coin; scholarly opinion on crypto is divided"


def screen(
    symbol: str, snapshots: Sequence[Snapshot], coins: Collection[str], today: date
) -> Screening:
    """The Shariah verdict for `symbol`, from the screened funds' current holdings.

    A Shariah-screened fund itself is allowed. A stock must be held by a fund whose list is
    no older than the lists' staleness limit, so a list that stopped updating cannot keep
    approving names. A coin must be in the configured coin list. Anything else is refused.
    """
    plain = symbol.upper()
    if "/" in plain:
        coin = plain.split("/")[0]
        if coin in coins:
            return Screening(symbol=symbol, allowed=True, reason=CRYPTO_NOTE)
        return Screening(symbol=symbol, allowed=False, reason=f"{coin} is not a listed coin")
    if plain in SHARIAH_FUNDS:
        return Screening(symbol=symbol, allowed=True, reason="a Shariah-screened fund")
    fresh = [s for s in snapshots if (today - s.as_of) <= lists.STALE_AFTER]
    holders = [f"{s.name} ({s.as_of.isoformat()})" for s in fresh if plain in s.symbols]
    if holders:
        return Screening(symbol=symbol, allowed=True, reason=f"held by {', '.join(holders)}")
    if not fresh:
        return Screening(
            symbol=symbol,
            allowed=False,
            reason="no Shariah list is current, so the stock cannot be confirmed",
        )
    names = ", ".join(f"{s.name} ({s.as_of.isoformat()})" for s in fresh)
    return Screening(symbol=symbol, allowed=False, reason=f"not held by {names}")
