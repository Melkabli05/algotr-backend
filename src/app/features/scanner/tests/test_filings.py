from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.features.scanner import filings
from app.features.scanner.filings import Edgar, FilingStore

NOW = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
TODAY = NOW.date()
CONTACT = "Test Person test@example.test"


def payload(rows: list[tuple[str, str, str, date]]) -> dict[str, Any]:
    return {
        "cik": "320193",
        "filings": {
            "recent": {
                "accessionNumber": [row[0] for row in rows],
                "form": [row[1] for row in rows],
                "items": [row[2] for row in rows],
                "filingDate": [row[3].isoformat() for row in rows],
            }
        },
    }


def website(
    companies: dict[str, int], submissions: dict[int, dict[str, Any]]
) -> tuple[httpx.Client, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        url = str(request.url)
        if url == filings.TICKERS:
            body = {
                str(index): {"cik_str": cik, "ticker": ticker, "title": ticker}
                for index, (ticker, cik) in enumerate(companies.items())
            }
            return httpx.Response(200, json=body)
        for cik, found in submissions.items():
            if url == filings.SUBMISSIONS.format(cik=cik):
                return httpx.Response(200, json=found)
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def days_ago(count: int) -> date:
    return TODAY - timedelta(days=count)


QUARTERLY = [
    ("a1", "8-K", "2.02,9.01", days_ago(10)),
    ("a2", "10-Q", "", days_ago(9)),
    ("a3", "8-K", "2.02,9.01", days_ago(101)),
    ("a4", "8-K", "5.02", days_ago(60)),
    ("a5", "8-K", "2.02", days_ago(193)),
    ("a6", "4", "", days_ago(3)),
]


@pytest.fixture
def store(tmp_path: Path) -> FilingStore:
    return FilingStore(tmp_path / "filings.sqlite")


def test_only_8k_filings_are_kept_with_their_item_numbers() -> None:
    found = filings.read("AAPL", payload(QUARTERLY))
    assert [(f.accession, f.items) for f in found] == [
        ("a1", ["2.02", "9.01"]),
        ("a3", ["2.02", "9.01"]),
        ("a4", ["5.02"]),
        ("a5", ["2.02"]),
    ]


def test_the_fetch_skips_tickers_the_sec_does_not_know(store: FilingStore) -> None:
    http, seen = website({"AAPL": 320193}, {320193: payload(QUARTERLY)})
    edgar = Edgar(CONTACT, store, http, pause=0)
    assert edgar.refresh(["AAPL", "NOPE"], NOW) == 1
    assert store.checked().keys() == {"AAPL"}
    assert len(store.of(["AAPL"], days_ago(400))["AAPL"]) == 4
    assert len(seen) == 2


def test_the_requests_are_made_under_the_contact_the_user_gave(
    store: FilingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    http, _ = website({}, {})
    asked: list[str] = []

    def named(contact: str) -> httpx.Client:
        asked.append(contact)
        return http

    monkeypatch.setattr(filings, "client", named)
    Edgar(CONTACT, store)
    assert asked == [CONTACT]


def test_share_classes_use_the_sec_s_dash_spelling(store: FilingStore) -> None:
    http, _ = website({"BRK-B": 1067983}, {1067983: payload([("b1", "8-K", "2.02", days_ago(5))])})
    assert Edgar(CONTACT, store, http).ciks() == {"BRK.B": 1067983}


def test_a_company_without_a_submissions_file_is_skipped(store: FilingStore) -> None:
    http, _ = website({"AAPL": 320193, "MSFT": 789019}, {789019: payload(QUARTERLY)})
    assert Edgar(CONTACT, store, http, pause=0).refresh(["AAPL", "MSFT"], NOW) == 1
    assert store.checked().keys() == {"MSFT"}


def test_a_refusal_from_the_sec_stops_the_run_instead_of_being_hidden(store: FilingStore) -> None:
    refusing = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403)))
    with pytest.raises(httpx.HTTPStatusError):
        Edgar(CONTACT, store, refusing, pause=0).ciks()


def test_the_last_results_and_the_next_estimate_come_from_the_item_2_02_filings() -> None:
    result = filings.badge(filings.read("AAPL", payload(QUARTERLY)), TODAY)
    assert result.last_results == days_ago(10)
    assert result.days_since_results == 10
    assert result.next_results == days_ago(10) + timedelta(days=91)
    assert result.soon is False


def test_results_expected_within_a_week_are_flagged() -> None:
    rows = [("x", "8-K", "2.02", days_ago(85)), ("y", "8-K", "2.02", days_ago(176))]
    result = filings.badge(filings.read("AAPL", payload(rows)), TODAY)
    assert result.next_results is not None
    assert (result.next_results - TODAY).days == 6 and result.soon is True


def test_a_missed_quarter_moves_the_estimate_forward_by_whole_quarters() -> None:
    rows = [("x", "8-K", "2.02", days_ago(200))]
    result = filings.badge(filings.read("AAPL", payload(rows)), TODAY)
    assert result.next_results is not None and result.next_results > TODAY


def test_a_name_that_never_reported_has_no_dates() -> None:
    result = filings.badge([], TODAY)
    assert (result.last_results, result.next_results, result.soon) == (None, None, False)


def test_a_serious_item_filed_in_the_last_five_days_is_an_event_and_a_veto() -> None:
    rows = [("x", "8-K", "4.02", days_ago(2)), ("y", "8-K", "3.01", days_ago(9))]
    result = filings.badge(filings.read("ACME", payload(rows)), TODAY)
    assert [event.label for event in result.events] == ["Restated financials"]
    assert result.events[0].veto


def test_an_ordinary_item_is_no_veto() -> None:
    result = filings.badge(
        filings.read("ACME", payload([("x", "8-K", "5.02", days_ago(1))])), TODAY
    )
    assert [event.veto for event in result.events] == [False]
