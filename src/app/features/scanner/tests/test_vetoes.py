from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core import audit
from app.features.scanner import vetoes
from app.features.scanner.news import NewsStore
from app.features.scanner.tests.support import NOW, article

SESSIONS = pd.bdate_range(end="2026-11-02", periods=150)
NAMES = ["BAD", "A", "B", "C", "D", "E"]
BAD_NEWS = "First Solar Lowers Full-Year Guidance"


@pytest.fixture
def store(tmp_path: Path) -> NewsStore:
    return NewsStore(tmp_path / "news.sqlite")


def prices(drop: float, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flat noisy prices; BAD loses `drop` over the five sessions after session 100's close."""
    rng = np.random.default_rng(seed)
    moves = rng.normal(0.0, 0.004, (len(SESSIONS), len(NAMES)))
    moves[101:106, 0] += drop / 5
    close = pd.DataFrame(100 * np.cumprod(1 + moves, axis=0), index=SESSIONS, columns=NAMES)
    opens = close.shift(1).fillna(close.iloc[0])
    return close, opens, pd.DataFrame(True, index=SESSIONS, columns=NAMES)


def news_on(session: int, store: NewsStore, symbol: str = "BAD", text: str = BAD_NEWS) -> None:
    day = SESSIONS[session]
    hours = (NOW - day.tz_localize("UTC").replace(hour=14)).total_seconds() / 3600
    store.add([article(session, text, [symbol], timedelta(hours=hours))])


def test_a_name_that_falls_after_a_veto_does_worse_than_the_names_let_through(
    store: NewsStore,
) -> None:
    close, opens, ok = prices(drop=-0.10)
    news_on(100, store)
    result = vetoes.study(close, opens, ok, store, NAMES)
    assert result.events == 1 and result.names == 1
    five = next(item for item in result.horizons if item.sessions == 5)
    assert five.mean_excess < -0.05 and five.share_below == 1.0
    assert five.vetoed_return < five.others_return


def test_a_veto_with_no_fall_afterwards_shows_no_excess_loss(store: NewsStore) -> None:
    close, opens, ok = prices(drop=0.0)
    news_on(100, store)
    five = next(i for i in vetoes.study(close, opens, ok, store, NAMES).horizons if i.sessions == 5)
    assert abs(five.mean_excess) < 0.03


def test_the_return_starts_at_the_next_opening_not_the_close_that_carried_the_news(
    store: NewsStore,
) -> None:
    close, opens, ok = prices(drop=0.0)
    close.iloc[101, 0] *= 0.8
    opens.iloc[101, 0] = close.iloc[100, 0]
    news_on(100, store)
    one = next(i for i in vetoes.study(close, opens, ok, store, NAMES).horizons if i.sessions == 1)
    assert one.vetoed_return < -0.15


def test_only_serious_negative_news_tagged_to_few_names_is_a_veto(store: NewsStore) -> None:
    close, opens, ok = prices(drop=-0.10)
    news_on(100, store, text="Acme Wins Contract")
    store.add([article(900, BAD_NEWS, ["BAD", "A", "B", "C"], timedelta(days=30))])
    assert vetoes.study(close, opens, ok, store, NAMES).events == 0


def test_a_name_the_scanner_would_not_consider_is_left_out(store: NewsStore) -> None:
    close, opens, ok = prices(drop=-0.10)
    ok["BAD"] = False
    news_on(100, store)
    assert vetoes.study(close, opens, ok, store, NAMES).events == 0


def test_a_vetoed_name_is_not_one_of_the_names_it_is_compared_with(store: NewsStore) -> None:
    close, opens, ok = prices(drop=0.0)
    news_on(100, store)
    news_on(100, store, symbol="A")
    two = vetoes.study(close, opens, ok, store, NAMES)
    assert two.events == 2
    flags = vetoes.active(
        vetoes.onsets(store, NAMES, close.index), close.index, [str(name) for name in NAMES]
    )
    assert flags.iloc[101][["BAD", "A"]].all() and not flags.iloc[101][["B", "C"]].any()


def test_one_name_vetoed_by_two_headlines_in_a_session_is_one_event(store: NewsStore) -> None:
    close, opens, ok = prices(drop=-0.10)
    news_on(100, store)
    store.add([article(777, "Acme Faces Class Action", ["BAD"], timedelta(hours=1))])
    found = vetoes.onsets(store, NAMES, close.index)
    assert not found.duplicated(["symbol", "session"]).any()


def test_a_horizon_that_has_not_happened_yet_is_left_out(store: NewsStore) -> None:
    close, opens, ok = prices(drop=-0.10)
    news_on(len(SESSIONS) - 3, store)
    result = vetoes.study(close, opens, ok, store, NAMES)
    assert [item.sessions for item in result.horizons] == [1]


def test_the_study_reads_back_from_the_audit_log(tmp_path: Path, store: NewsStore) -> None:
    close, opens, ok = prices(drop=-0.10)
    news_on(100, store)
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    assert vetoes.latest(log) is None
    result = vetoes.study(close, opens, ok, store, NAMES)
    log.record(vetoes.KIND, {"result": result.model_dump(mode="json")})
    assert vetoes.latest(log) == result
