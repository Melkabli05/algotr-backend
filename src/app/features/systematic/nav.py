from collections.abc import Mapping

from pydantic import BaseModel, computed_field

from app.core.compliance import KillState

REDUCED = 0.5
WATCH_AT = 0.25
DEFENSIVE_AT = 0.75


class Holding(BaseModel):
    qty: float
    price: float


class Book(BaseModel):
    """What a strategy meant to hold, and a value index of that since it started.

    The index moves only with the prices of those holdings, so deposits, withdrawals and trades
    made by hand in the same account never move it.
    """

    nav: float = 1.0
    peak: float = 1.0
    capital: float = 0.0
    holdings: dict[str, Holding] = {}

    @computed_field
    @property
    def drawdown(self) -> float:
        return 1 - self.nav / self.peak if self.peak > 0 else 0.0


def gain(book: Book, marks: Mapping[str, float]) -> float:
    """Dollars the holdings made since they were last priced."""
    return sum(
        held.qty * (marks.get(symbol, held.price) - held.price)
        for symbol, held in book.holdings.items()
    )


def grown(book: Book, made: float, capital: float) -> Book:
    nav = book.nav * (1 + made / capital) if capital > 0 else book.nav
    return book.model_copy(update={"nav": nav, "peak": max(book.peak, nav)})


def marked(book: Book, marks: Mapping[str, float]) -> Book:
    """The book after its holdings move to `marks`; a symbol without a mark keeps its price."""
    holdings = {
        symbol: Holding(qty=held.qty, price=marks.get(symbol, held.price))
        for symbol, held in book.holdings.items()
    }
    return grown(book, gain(book, marks), book.capital).model_copy(update={"holdings": holdings})


def total(previous: Book, books: Mapping[str, Book], marks: Mapping[str, float]) -> Book:
    """The whole autopilot: every strategy's gain over every strategy's capital."""
    made = sum(gain(book, marks) for book in books.values())
    capital = sum(book.capital for book in books.values())
    return grown(previous, made, capital).model_copy(update={"capital": capital})


def retargeted(
    book: Book, capital: float, targets: Mapping[str, float], prices: Mapping[str, float]
) -> Book:
    """The book holding `targets` (dollars) at `prices` from now on."""
    holdings = {
        symbol: Holding(qty=value / prices[symbol], price=prices[symbol])
        for symbol, value in targets.items()
        if value > 0 and prices.get(symbol, 0) > 0
    }
    return book.model_copy(update={"capital": capital, "holdings": holdings})


def scale(drawdown: float, limit: float, reduce_at: float) -> float:
    """Half size once the drawdown reaches `reduce_at` of the limit; 1 otherwise."""
    return REDUCED if reduce_at < 1 and drawdown >= limit * reduce_at else 1.0


def restarted(book: Book) -> Book:
    return book.model_copy(update={"peak": book.nav})


def kill_state(drawdown: float, limit: float, reduce_at: float, halted: bool) -> KillState:
    """Where a drawdown stands on the way to the limit, in five steps.

    normal; watch from a quarter of the limit (shown, nothing changes); reduce from `reduce_at`
    of it (half size); defensive from three quarters (sells only, no new buys anywhere);
    flatten at the limit or while halted (everything sold, trading stopped until Resume).
    """
    if halted or drawdown >= limit:
        return "flatten"
    if drawdown >= limit * DEFENSIVE_AT:
        return "defensive"
    if reduce_at < 1 and drawdown >= limit * reduce_at:
        return "reduce"
    if drawdown >= limit * WATCH_AT:
        return "watch"
    return "normal"


def capped(targets: Mapping[str, float], book: Book) -> dict[str, float]:
    """Targets no larger than what the book holds now: in defensive, nothing is added."""
    held = {symbol: h.qty * h.price for symbol, h in book.holdings.items()}
    return {symbol: min(value, held.get(symbol, 0.0)) for symbol, value in targets.items()}
