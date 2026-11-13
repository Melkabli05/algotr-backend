from app.features.book.book import STALE_AFTER, Book, BookKeeper
from app.features.book.tests.support import ASKS, BIDS, message, snapshot


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_snapshot_replaces_the_book_and_makes_it_ready() -> None:
    book = Book("BTC/USD")
    book.apply(message(bids=[(1.0, 1.0)], asks=[(2.0, 1.0)], reset=True))
    book.apply(snapshot())
    assert book.ready
    assert book.top(10) == (BIDS, ASKS)


def test_an_update_changes_adds_and_removes_levels() -> None:
    book = Book("BTC/USD")
    book.apply(snapshot())
    book.apply(message(bids=[(100.0, 3.0), (97.0, 1.0)], asks=[(100.5, 0.0), (103.0, 5.0)]))
    bids, asks = book.top(10)
    assert bids[0] == (100.0, 3.0) and (97.0, 1.0) in bids
    assert (100.5, 1.0) not in asks and (103.0, 5.0) in asks
    assert asks[0] == (101.0, 2.0)


def test_an_update_before_any_snapshot_is_not_applied() -> None:
    book = Book("BTC/USD")
    assert not book.apply(message(bids=[(100.0, 1.0)]))
    assert not book.ready and book.bids == {}


def test_the_best_levels_come_first_and_only_as_many_as_asked() -> None:
    book = Book("BTC/USD")
    book.apply(snapshot())
    bids, asks = book.top(2)
    assert bids == [(100.0, 1.0), (99.5, 2.0)] and asks == [(100.5, 1.0), (101.0, 2.0)]


def test_a_book_whose_best_bid_reaches_the_best_ask_is_crossed() -> None:
    book = Book("BTC/USD")
    book.apply(snapshot())
    assert not book.crossed()
    book.apply(message(bids=[(100.5, 1.0)]))
    assert book.crossed()
    assert not Book("ETH/USD").crossed()


def test_the_keeper_routes_a_message_to_its_coin_and_ignores_others() -> None:
    keeper = BookKeeper(["BTC/USD"])
    keeper.handle(snapshot("BTC/USD"))
    keeper.handle(snapshot("DOGE/USD"))
    assert keeper.books["BTC/USD"].ready and list(keeper.books) == ["BTC/USD"]


def test_the_feed_is_stale_after_five_seconds_of_silence() -> None:
    clock = Clock()
    keeper = BookKeeper(["BTC/USD"], clock)
    keeper.handle(snapshot())
    clock.now += STALE_AFTER - 0.1
    assert not keeper.stale
    clock.now += 0.2
    assert keeper.stale and keeper.silence > STALE_AFTER
    keeper.handle(message(bids=[(100.0, 2.0)]))
    assert not keeper.stale


def test_a_change_at_the_top_adds_to_the_flow_and_taking_it_empties_it() -> None:
    book = Book("BTC/USD")
    book.apply(snapshot())
    book.apply(message(bids=[(100.0, 3.0)]))
    book.apply(message(asks=[(100.5, 0.0)]))
    assert book.flow == 2.0 + 1.0
    assert book.take_flow() == 3.0 and book.flow == 0.0


def test_the_snapshot_and_changes_below_the_top_are_not_flow() -> None:
    book = Book("BTC/USD")
    book.apply(snapshot())
    book.apply(message(bids=[(99.0, 9.0)], asks=[(104.0, 1.0)]))
    assert book.flow == 0.0
    book.apply(snapshot())
    assert book.flow == 0.0


def test_an_update_before_a_snapshot_is_not_flow() -> None:
    book = Book("BTC/USD")
    book.apply(message(bids=[(100.0, 3.0)]))
    assert book.flow == 0.0 and book.best() is None


def test_the_best_levels_are_the_highest_bid_and_the_lowest_ask() -> None:
    book = Book("BTC/USD")
    book.apply(snapshot())
    assert book.best() == ((100.0, 1.0), (100.5, 1.0))
