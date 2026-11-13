from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal

from app.features.ledger.lots import ZERO, Book

QTY_TOLERANCE = Decimal("1e-9")
COST_TOLERANCE = Decimal("0.005")
DUST_VALUE = Decimal(1)


@dataclass(frozen=True)
class Held:
    """A position as the broker reports it."""

    symbol: str
    qty: Decimal
    average_cost: Decimal | None


@dataclass(frozen=True)
class Difference:
    symbol: str
    kind: Literal["quantity", "pending", "cost"]
    ledger_qty: Decimal
    broker_qty: Decimal
    ledger_cost: Decimal | None
    broker_cost: Decimal | None
    note: str


def compare(
    book: Book,
    held: Sequence[Held],
    pending: Mapping[str, Decimal],
    actions: Mapping[str, str],
) -> list[Difference]:
    """Where the ledger and the broker disagree; `book` pools each symbol, so its lots are whole."""
    ledger = {lot.symbol: lot for lot in book.lots}
    broker = {position.symbol: position for position in held}
    found = []
    for symbol in sorted({*ledger, *broker}):
        lot, position = ledger.get(symbol), broker.get(symbol)
        row = Difference(
            symbol,
            "cost",
            lot.signed_qty if lot else ZERO,
            position.qty if position else ZERO,
            lot.unit_cost if lot else None,
            position.average_cost if position else None,
            "",
        )
        verdict = judge(row, pending.get(symbol, ZERO), actions.get(symbol))
        if verdict is not None:
            found.append(verdict)
    return found


def judge(row: Difference, pending: Decimal, action: str | None) -> Difference | None:
    gap = row.ledger_qty - row.broker_qty
    if abs(gap) > QTY_TOLERANCE:
        if ZERO < gap <= pending:
            return replace(row, kind="pending", note="Coin fees post the next day")
        return replace(row, kind="quantity", note=explain(gap, action))
    ours, theirs = row.ledger_cost, row.broker_cost
    if not ours or not theirs or abs(row.broker_qty * theirs) < DUST_VALUE:
        return None
    if abs(ours - theirs) / theirs > COST_TOLERANCE:
        return replace(row, note="Average cost differs; Alpaca's method is not documented")
    return None


def explain(gap: Decimal, action: str | None) -> str:
    if action:
        return f"A {action} is on record for this symbol and is not applied to the lots"
    side = "The ledger holds" if gap > ZERO else "Alpaca holds"
    return f"{side} {abs(gap):f} more than the other"
