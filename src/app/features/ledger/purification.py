from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal

from pydantic import BaseModel

from app.features.ledger.activities import Activity

DIVIDENDS = ("DIV", "DIVTXEX")
INTEREST = ("INT",)
ZAKAT_RATE = Decimal("0.025")
ZERO = Decimal(0)


class DividendRow(BaseModel):
    symbol: str
    received: Decimal
    ratio: Decimal | None
    owed: Decimal | None


class Statement(BaseModel):
    """What a year's income owes to charity, and an estimate of the zakat on the account.

    A dividend owes its non-compliant share, the ratio the fund or a screening service publishes
    for that holding; a holding without one is listed in `missing`. Interest received is owed in
    full. Interest paid cannot be purified away and is reported so it can be stopped. Zakat is
    estimated by the trading-asset method, 2.5% of the account's value today, and is due once
    a lunar year has passed on it.
    """

    year: int
    dividends: list[DividendRow]
    interest_received: Decimal
    interest_paid: Decimal
    owed: Decimal
    missing: list[str]
    zakat_base: Decimal | None
    zakat: Decimal | None


def in_year(row: Activity, year: int) -> bool:
    return row.moment.year == year


def statement(
    rows: Iterable[Activity],
    ratios: Mapping[str, Decimal],
    year: int,
    equity: Decimal | None = None,
) -> Statement:
    received: dict[str, Decimal] = defaultdict(Decimal)
    earned = paid = ZERO
    for row in rows:
        if not in_year(row, year) or row.net_amount is None:
            continue
        if row.activity_type in DIVIDENDS and row.symbol and row.net_amount > 0:
            received[row.key] += row.net_amount
        elif row.activity_type in INTEREST:
            if row.net_amount > 0:
                earned += row.net_amount
            else:
                paid -= row.net_amount
    dividends = []
    for symbol in sorted(received):
        ratio = ratios.get(symbol)
        owed = None if ratio is None else received[symbol] * ratio
        dividends.append(
            DividendRow(symbol=symbol, received=received[symbol], ratio=ratio, owed=owed)
        )
    owed_total = earned + sum((row.owed for row in dividends if row.owed is not None), ZERO)
    return Statement(
        year=year,
        dividends=dividends,
        interest_received=earned,
        interest_paid=paid,
        owed=owed_total,
        missing=[row.symbol for row in dividends if row.ratio is None],
        zakat_base=equity,
        zakat=None if equity is None else equity * ZAKAT_RATE,
    )
