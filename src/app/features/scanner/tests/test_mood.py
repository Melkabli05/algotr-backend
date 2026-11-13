import json
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.features.scanner import backfill, mood


def test_a_headline_before_the_close_informs_its_day_and_one_after_it_the_next() -> None:
    assert mood.session_of(datetime(2026, 11, 4, 20, 59, tzinfo=UTC)) == date(2026, 11, 4)
    assert mood.session_of(datetime(2026, 11, 4, 21, 1, tzinfo=UTC)) == date(2026, 11, 5)


def headlines(*rows: tuple[int, str, float, int]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["id", "session", "score", "risk"])
    frame["session"] = pd.to_datetime(frame["session"]).dt.date
    return frame


SESSIONS = [date(2026, 11, 6), date(2026, 11, 9), date(2026, 11, 10)]


def test_each_session_averages_the_headlines_since_the_one_before() -> None:
    found = mood.daily(
        headlines(
            (1, "2026-11-06", 0.5, 2),
            (2, "2026-11-06", -0.1, 4),
            (3, "2026-11-07", -0.6, 5),
            (4, "2026-11-08", 0.0, 1),
        ),
        SESSIONS,
    )
    assert found.loc["2026-11-06", "sentiment"] == pytest.approx(0.2)
    assert found.loc["2026-11-06", "news_risk"] == pytest.approx(3.0)
    assert found.loc["2026-11-09", "sentiment"] == pytest.approx(-0.3)
    assert found.loc["2026-11-09", "headlines"] == 2
    assert pd.isna(found.loc["2026-11-10", "sentiment"])


def test_headlines_after_the_last_session_are_left_out_and_none_gives_nan() -> None:
    late = mood.daily(headlines((1, "2026-11-11", 0.9, 1)), SESSIONS)
    assert late["sentiment"].isna().all()
    assert mood.daily(pd.DataFrame(), SESSIONS)["sentiment"].isna().all()


def test_the_backfill_goes_back_a_month_at_a_time_and_resumes(tmp_path: Path) -> None:
    feed = MagicMock()
    feed.fetch.return_value = 3
    path = tmp_path / "progress.json"
    added = backfill.backfill(feed, ["AAPL"], date(2026, 8, 1), date(2026, 10, 30), path)
    starts = [call.args[1].date() for call in feed.fetch.call_args_list]
    assert starts == [date(2026, 9, 30), date(2026, 8, 31), date(2026, 8, 1)]
    assert added == 9
    assert json.loads(path.read_text())["done_from"] == "2026-08-01"
    feed.fetch.reset_mock()
    assert backfill.backfill(feed, ["AAPL"], date(2026, 8, 1), date(2026, 10, 30), path) == 0
    feed.fetch.assert_not_called()
    backfill.backfill(feed, ["AAPL"], date(2026, 7, 1), date(2026, 10, 30), path)
    assert [call.args[1].date() for call in feed.fetch.call_args_list] == [
        date(2026, 7, 2),
        date(2026, 7, 1),
    ]
