from datetime import UTC, datetime, timedelta
from decimal import Decimal
from decimal import Decimal as D

from app.features.ledger.lots import Execution, replay

START = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)


def at(day: int) -> datetime:
    return START + timedelta(days=day)


def buy(order: str, day: int, qty: int, price: int, symbol: str = "ACME") -> Execution:
    return Execution(order, symbol, "buy", at(day), Decimal(qty), Decimal(price))


def sell(order: str, day: int, qty: int, price: int, symbol: str = "ACME") -> Execution:
    return Execution(order, symbol, "sell", at(day), Decimal(qty), Decimal(price))


def test_a_loss_sale_then_a_buy_within_30_days_moves_the_loss_into_the_new_basis() -> None:
    book = replay(
        [buy("b1", 0, 10, 50), sell("s1", 40, 10, 40), buy("b2", 50, 10, 42)],
        "fifo",
        wash_exempt=(),
    )
    [close] = book.closes
    assert (close.gain, close.disallowed, close.reported) == (D(-100), D(100), D(0))
    [lot] = book.lots
    assert (lot.cost, lot.wash) == (D(520), D(100))


def test_a_buy_more_than_30_days_later_is_not_a_replacement() -> None:
    book = replay(
        [buy("b1", 0, 10, 50), sell("s1", 40, 10, 40), buy("b2", 71, 10, 42)],
        "fifo",
        wash_exempt=(),
    )
    assert book.closes[0].disallowed == 0
    assert book.lots[0].cost == 420


def test_shares_bought_in_the_30_days_before_a_loss_sale_replace_it() -> None:
    book = replay(
        [buy("b1", 0, 10, 50), buy("b2", 35, 10, 45), sell("s1", 40, 10, 40)],
        "fifo",
        wash_exempt=(),
    )
    [close] = book.closes
    assert close.lot == "b1" and close.disallowed == 100
    [lot] = book.lots
    assert (lot.id, lot.cost) == ("b2", D(550))


def test_only_as_many_shares_as_were_bought_back_are_disallowed() -> None:
    book = replay(
        [buy("b1", 0, 10, 50), sell("s1", 40, 10, 40), buy("b2", 45, 4, 41)], "fifo", wash_exempt=()
    )
    assert book.closes[0].disallowed == 40
    assert book.closes[0].reported == -60


def test_a_replacement_share_replaces_one_sold_share_once() -> None:
    book = replay(
        [
            buy("b1", 0, 10, 50),
            sell("s1", 40, 10, 40),
            buy("b2", 45, 10, 41),
            buy("b3", 46, 10, 41),
        ],
        "fifo",
        wash_exempt=(),
    )
    assert book.closes[0].disallowed == 100
    assert sorted(lot.wash for lot in book.lots) == [0, 100]


def test_a_gain_is_never_washed_and_coins_are_exempt() -> None:
    gain = replay(
        [buy("b1", 0, 10, 50), sell("s1", 10, 10, 60), buy("b2", 12, 10, 61)],
        "fifo",
        wash_exempt=(),
    )
    assert gain.closes[0].disallowed == 0
    coin = [
        buy("b1", 0, 1, 50, "BTCUSD"),
        sell("s1", 10, 1, 40, "BTCUSD"),
        buy("b2", 12, 1, 41, "BTCUSD"),
    ]
    assert replay(coin, "fifo", wash_exempt={"BTCUSD"}).closes[0].disallowed == 0


def test_without_the_rule_nothing_changes() -> None:
    book = replay([buy("b1", 0, 10, 50), sell("s1", 40, 10, 40), buy("b2", 50, 10, 42)], "fifo")
    assert book.closes[0].disallowed == 0 and book.lots[0].cost == 420
