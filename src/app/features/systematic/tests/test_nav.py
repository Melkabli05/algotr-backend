import pytest

from app.features.systematic import nav
from app.features.systematic.nav import Book, Holding


def holding(qty: float, price: float) -> Holding:
    return Holding(qty=qty, price=price)


def test_a_book_gains_what_its_holdings_gain_over_its_capital() -> None:
    book = Book(capital=1_000, holdings={"A": holding(5, 100)})
    moved = nav.marked(book, {"A": 110})
    assert moved.nav == pytest.approx(1.05)
    assert moved.peak == pytest.approx(1.05)
    assert moved.holdings["A"].price == 110


def test_a_fall_from_the_peak_is_the_drawdown() -> None:
    book = Book(nav=1.2, peak=1.2, capital=1_000, holdings={"A": holding(10, 100)})
    moved = nav.marked(book, {"A": 88})
    assert moved.nav == pytest.approx(1.2 * 0.88)
    assert moved.drawdown == pytest.approx(0.12)


def test_marking_twice_at_the_same_prices_changes_nothing() -> None:
    book = nav.marked(Book(capital=1_000, holdings={"A": holding(5, 100)}), {"A": 90})
    assert nav.marked(book, {"A": 90}) == book


def test_a_symbol_without_a_price_keeps_its_last_one() -> None:
    book = Book(capital=1_000, holdings={"A": holding(5, 100)})
    assert nav.marked(book, {}).nav == 1.0


def test_the_total_weighs_every_book_by_its_capital() -> None:
    books = {
        "a": Book(capital=1_000, holdings={"A": holding(10, 100)}),
        "b": Book(capital=3_000, holdings={}),
    }
    whole = nav.total(Book(), books, {"A": 90})
    assert whole.nav == pytest.approx(1 - 100 / 4_000)
    assert whole.capital == 4_000


def test_a_book_without_capital_never_moves() -> None:
    assert nav.marked(Book(holdings={"A": holding(1, 10)}), {"A": 5}).nav == 1.0


def test_retargeting_holds_the_dollars_at_the_prices_given() -> None:
    book = nav.retargeted(Book(nav=0.9, peak=1.0), 2_000, {"A": 500, "B": 0}, {"A": 50, "B": 9})
    assert book.holdings == {"A": holding(10, 50)}
    assert (book.capital, book.nav, book.peak) == (2_000, 0.9, 1.0)


def test_half_size_from_the_chosen_share_of_the_limit() -> None:
    assert nav.scale(0.07, 0.15, 0.5) == 1.0
    assert nav.scale(0.075, 0.15, 0.5) == 0.5
    assert nav.scale(0.14, 0.15, 1.0) == 1.0


def test_restarting_sets_the_peak_to_the_current_value() -> None:
    assert nav.restarted(Book(nav=0.8, peak=1.0)).drawdown == 0


@pytest.mark.parametrize(
    ("drawdown", "halted", "state"),
    [
        (0.0, False, "normal"),
        (0.0375, False, "watch"),
        (0.075, False, "reduce"),
        (0.1125, False, "defensive"),
        (0.15, False, "flatten"),
        (0.0, True, "flatten"),
    ],
)
def test_the_kill_switch_escalates_in_five_steps(drawdown: float, halted: bool, state: str) -> None:
    assert nav.kill_state(drawdown, 0.15, 0.5, halted) == state


def test_without_the_size_cut_reduce_is_skipped() -> None:
    assert nav.kill_state(0.08, 0.15, 1.0, False) == "watch"


def test_capped_targets_never_add_to_a_holding() -> None:
    book = Book(holdings={"A": holding(10, 50)})
    assert nav.capped({"A": 900, "B": 100}, book) == {"A": 500, "B": 0.0}
    assert nav.capped({"A": 100}, book) == {"A": 100}
