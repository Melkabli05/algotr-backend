from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from app.features.book.store import BookStore, Flow, Sample, pack, unpack
from app.features.book.tests.support import ASKS, BIDS


def sample(at: datetime, symbol: str = "BTC/USD", mid: float = 100.25) -> Sample:
    return Sample(at, symbol, mid, 0.5, BIDS, ASKS)


NOON = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> BookStore:
    return BookStore(tmp_path / "book")


def test_prices_survive_packing_exactly_and_sizes_to_single_precision() -> None:
    levels = [(77604.12, 0.001007), (77593.8, 0.001989), (77564.1, 1.109)]
    found = unpack(pack(levels))
    assert [price for price, _ in found] == [price for price, _ in levels]
    assert [size for _, size in found] == pytest.approx([size for _, size in levels], rel=1e-6)


def test_no_levels_pack_to_nothing() -> None:
    assert unpack(pack([])) == []


def test_a_sample_is_read_back_whole(store: BookStore) -> None:
    store.add([sample(NOON)])
    [found] = store.read("BTC/USD", NOON - timedelta(minutes=1), NOON + timedelta(minutes=1))
    assert found == sample(NOON)


def test_samples_are_kept_in_a_file_for_each_utc_day(store: BookStore) -> None:
    late = datetime(2026, 11, 3, 23, 59, 59, tzinfo=UTC)
    early = late + timedelta(seconds=2)
    store.add([sample(late), sample(early)])
    assert sorted(path.name for path in store.root.glob("*.sqlite")) == [
        "2026-11-03.sqlite",
        "2026-11-04.sqlite",
    ]
    found = store.read("BTC/USD", late - timedelta(seconds=1), early + timedelta(seconds=1))
    assert [item.at for item in found] == [late, early]


def test_a_read_picks_one_coin_and_the_window_only(store: BookStore) -> None:
    moments = [NOON + timedelta(seconds=n) for n in range(5)]
    store.add([sample(at) for at in moments] + [sample(moments[2], "ETH/USD")])
    found = store.read("BTC/USD", moments[1], moments[3])
    assert [item.at for item in found] == moments[1:4]
    assert {item.symbol for item in found} == {"BTC/USD"}


def test_the_same_second_is_stored_once(store: BookStore) -> None:
    store.add([sample(NOON, mid=100.0)])
    store.add([sample(NOON, mid=101.0)])
    [found] = store.read("BTC/USD", NOON - timedelta(seconds=1), NOON + timedelta(seconds=1))
    assert found.mid == 101.0


def test_reading_a_day_with_no_file_is_empty(store: BookStore) -> None:
    assert store.read("BTC/USD", NOON, NOON + timedelta(days=2)) == []


def test_days_older_than_the_ones_kept_are_deleted(store: BookStore) -> None:
    for days_ago in (0, 1, 2, 3, 4):
        store.add([sample(NOON - timedelta(days=days_ago))])
    removed = store.prune(date(2026, 11, 3))
    assert removed == 2
    assert sorted(path.stem for path in store.root.glob("*.sqlite")) == [
        "2026-11-01",
        "2026-11-02",
        "2026-11-03",
    ]


def test_pruning_ignores_files_that_are_not_days(store: BookStore) -> None:
    store.root.mkdir(parents=True)
    (store.root / "notes.sqlite").write_text("x")
    assert store.prune(date(2026, 11, 3)) == 0


def test_flows_are_stored_with_their_window_and_read_back(store: BookStore) -> None:
    flows = [Flow(NOON + timedelta(seconds=5 * n), "BTC/USD", 5.0, 0.5 * n) for n in range(4)]
    store.add_flows(flows + [Flow(NOON, "ETH/USD", 5.0, 9.0)])
    found = store.read_flows("BTC/USD", NOON + timedelta(seconds=4), NOON + timedelta(seconds=11))
    assert found == flows[1:3]


def test_a_day_file_made_before_flows_existed_gains_the_table(store: BookStore) -> None:
    store.add([sample(NOON)])
    store.add_flows([Flow(NOON, "BTC/USD", 5.0, 1.0)])
    assert len(store.read("BTC/USD", NOON - timedelta(seconds=1), NOON + timedelta(seconds=1))) == 1
    assert (
        len(store.read_flows("BTC/USD", NOON - timedelta(seconds=1), NOON + timedelta(seconds=1)))
        == 1
    )


def test_the_spreads_are_read_in_basis_points_for_one_coin_since_a_moment(store: BookStore) -> None:
    moments = [NOON + timedelta(seconds=n) for n in range(3)]
    store.add(
        [Sample(at, "BTC/USD", 100.0, 0.5 * (n + 1), BIDS, ASKS) for n, at in enumerate(moments)]
        + [Sample(moments[1], "ETH/USD", 100.0, 9.0, BIDS, ASKS)]
    )
    assert sorted(store.spreads("BTC/USD", moments[1])) == pytest.approx([100.0, 150.0])
    assert store.spreads("DOGE/USD", moments[0]) == []


def test_the_spreads_cross_midnight(store: BookStore) -> None:
    midnight = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    before, after = midnight - timedelta(seconds=1), midnight + timedelta(seconds=1)
    store.add([sample(before), sample(after)])
    assert len(store.spreads("BTC/USD", before - timedelta(seconds=1))) == 2


def test_the_spreads_of_a_day_with_no_file_are_empty(store: BookStore) -> None:
    assert store.spreads("BTC/USD", NOON) == []
