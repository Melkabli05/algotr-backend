from decimal import Decimal
from typing import Any

import pytest

from app.features.ledger import purification
from app.features.ledger.activities import Activity


def row(kind: str, day: str, amount: str, symbol: str | None = None, n: int = 0) -> Activity:
    data: dict[str, Any] = {
        "id": f"{day}-{kind}-{n}",
        "activity_type": kind,
        "date": day,
        "net_amount": amount,
        "symbol": symbol,
    }
    return Activity.model_validate(data)


ROWS = [
    row("DIV", "2026-03-20", "40.00", "SPUS", 1),
    row("DIV", "2026-06-20", "60.00", "SPUS", 2),
    row("DIVNRA", "2026-06-20", "-9.00", "SPUS", 3),
    row("DIV", "2026-06-20", "10.00", "HLAL", 4),
    row("INT", "2026-07-01", "2.50", None, 5),
    row("INT", "2026-08-01", "-1.25", None, 6),
    row("DIV", "2025-12-20", "999.00", "SPUS", 7),
    row("FILL", "2026-06-20", "-5000", "SPUS", 8),
]


def test_a_dividend_owes_its_ratio_and_interest_received_is_owed_in_full() -> None:
    found = purification.statement(ROWS, {"SPUS": Decimal("0.012")}, 2026)
    [hlal, spus] = found.dividends
    assert (spus.symbol, spus.received, spus.owed) == ("SPUS", Decimal("100.00"), Decimal("1.2"))
    assert (hlal.ratio, hlal.owed) == (None, None)
    assert found.interest_received == Decimal("2.50")
    assert found.owed == Decimal("3.70")
    assert found.missing == ["HLAL"]


def test_interest_paid_is_reported_apart_since_it_cannot_be_purified() -> None:
    found = purification.statement(ROWS, {}, 2026)
    assert found.interest_paid == Decimal("1.25")


def test_only_the_asked_year_counts() -> None:
    found = purification.statement(ROWS, {"SPUS": Decimal("0.01")}, 2025)
    assert [(d.symbol, d.received) for d in found.dividends] == [("SPUS", Decimal("999.00"))]


def test_zakat_is_two_and_a_half_percent_of_the_value_given() -> None:
    found = purification.statement([], {}, 2026, Decimal("100000"))
    assert found.zakat == Decimal("2500.000")
    assert purification.statement([], {}, 2026).zakat is None


@pytest.mark.parametrize("kind", ["DIVNRA", "DIVTW", "DIVROC", "FILL"])
def test_withholding_return_of_capital_and_fills_are_not_income(kind: str) -> None:
    found = purification.statement([row(kind, "2026-01-02", "50", "SPUS")], {}, 2026)
    assert found.dividends == []
