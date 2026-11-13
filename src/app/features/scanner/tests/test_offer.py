from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.sleeve import Criterion, Sleeve
from app.features.scanner import offer, rank, scan
from app.features.scanner.router import get_scanner
from app.features.scanner.schemas import Refresh
from app.features.scanner.service import Scanner
from app.features.scanner.tests.support import listed_scanner, noisy_market
from app.features.scanner.tests.test_watch import recent, scanner_with
from app.main import app

LABEL = scan.PAGE.label
CHECKS = 5
SESSIONS = "At least 40 sessions logged"


def refreshed(tmp_path: Path) -> Scanner:
    scanner = listed_scanner(tmp_path, noisy_market())
    scanner.refresh()
    return scanner


def offered(scanner: Scanner, variant: str = LABEL) -> Sleeve:
    found = scanner.sleeve(variant)
    assert found is not None
    return found


def outcome(gate: list[Criterion], name: str) -> bool:
    return next(item.passed for item in gate if item.name.startswith(name))


def test_every_variant_has_scheduled_holdings_within_the_caps(tmp_path: Path) -> None:
    found = refreshed(tmp_path).market()
    assert found is not None
    assert set(found.analysis.scheduled) == {variant.label for variant in rank.VARIANTS}
    for weights in found.analysis.scheduled.values():
        assert sum(weights.values()) <= 1 + 1e-9
        assert all(0 < weight <= rank.CAP + 1e-9 for weight in weights.values())


def test_the_offer_holds_what_the_variant_is_scheduled_to_hold_largest_first(
    tmp_path: Path,
) -> None:
    scanner = refreshed(tmp_path)
    found = scanner.market()
    assert found is not None
    result = offered(scanner)
    weights = {item.symbol: item.weight for item in result.holdings}
    close = found.analysis.found.close.iloc[-1]
    assert weights == found.analysis.scheduled[LABEL]
    assert [item.weight for item in result.holdings] == sorted(weights.values(), reverse=True)
    assert result.session == found.analysis.found.close.index[-1].date()
    assert all(item.price == pytest.approx(float(close[item.symbol])) for item in result.holdings)
    assert all(item.name == f"{item.symbol} Inc" for item in result.holdings)
    assert all(item.blocked is None for item in result.holdings)


def test_there_is_no_offer_without_data_or_for_a_variant_that_does_not_exist(
    tmp_path: Path,
) -> None:
    scanner = listed_scanner(tmp_path, noisy_market())
    assert scanner.sleeve(LABEL) is None
    scanner.refresh()
    assert scanner.sleeve("no such variant") is None


def test_a_serious_setback_in_the_news_blocks_the_name_and_names_the_event(tmp_path: Path) -> None:
    scanner = scanner_with(tmp_path, [])
    scanner.refresh()
    symbol = offered(scanner).holdings[0].symbol
    scanner = scanner_with(tmp_path, [recent(1, "Acme Lowers Full-Year Outlook", symbol, 1)])
    scanner.poll_news()
    blocked = {item.symbol: item.blocked for item in offered(scanner).holdings if item.blocked}
    assert blocked == {symbol: "Guidance cut"}


def test_before_any_evaluation_the_record_fails_and_says_where_to_run_it(tmp_path: Path) -> None:
    result = offered(refreshed(tmp_path))
    assert [(item.name, item.passed) for item in result.criteria] == [
        ("The variants have been evaluated", False),
        (SESSIONS, False),
        ("Audit log intact", True),
    ]
    assert result.criteria[0].detail == "run it on the Evaluation tab"
    assert result.criteria[1].detail == "1 logged" and not result.passed


def test_after_an_evaluation_the_record_has_the_five_checks_of_that_variant(
    tmp_path: Path,
) -> None:
    scanner = refreshed(tmp_path)
    mine = next(item for item in scanner.evaluate().variants if item.label == LABEL)
    gate = offered(scanner).criteria
    assert gate[0].name.startswith("Evaluated in the last 30 days") and gate[0].passed
    assert [(item.name, item.passed) for item in gate[1 : 1 + CHECKS]] == [
        (check.name, check.passed) for check in mine.checks
    ]
    assert [item.name for item in gate[-2:]] == [SESSIONS, "Audit log intact"]


def test_forty_sessions_an_intact_log_and_a_recent_evaluation_are_each_needed(
    tmp_path: Path,
) -> None:
    evaluation = refreshed(tmp_path).evaluate()
    now = datetime.now(UTC)
    enough = offer.criteria(evaluation, LABEL, 40, True, now)
    assert outcome(enough, SESSIONS) and outcome(enough, "Audit log")
    assert not outcome(offer.criteria(evaluation, LABEL, 39, True, now), SESSIONS)
    assert not outcome(offer.criteria(evaluation, LABEL, 40, False, now), "Audit log")
    old = evaluation.model_copy(update={"end": now.date() - timedelta(days=31)})
    assert not outcome(offer.criteria(old, LABEL, 40, True, now), "Evaluated in the last")
    other = offer.criteria(evaluation, "other variant", 40, True, now)
    assert not outcome(other, "This variant was evaluated")


def test_a_fresh_scan_with_a_current_list_has_no_problem(tmp_path: Path) -> None:
    assert offered(refreshed(tmp_path)).problems == []


def test_old_data_an_old_session_and_a_failed_refresh_are_each_a_problem(tmp_path: Path) -> None:
    scanner = refreshed(tmp_path)
    found = scanner.market()
    assert found is not None
    session = found.analysis.found.close.index[-1].date()
    later = datetime.now(UTC) + timedelta(days=10)
    assert offer.problems(found, scanner.state.last, later) == [
        "The data was last refreshed more than 36 hours ago",
        f"The last session in the data is {(later.date() - session).days} days old",
    ]
    failed = Refresh(started=datetime.now(UTC), seconds=1, status="error", detail="boom")
    unfinished = ["The last data refresh did not finish"]
    assert offer.problems(found, failed, datetime.now(UTC)) == unfinished
    assert offer.problems(found, None, datetime.now(UTC)) == unfinished


def test_without_a_current_list_the_offer_holds_nothing_and_says_why(tmp_path: Path) -> None:
    old = (datetime.now(UTC) - timedelta(days=20)).strftime("%m/%d/%Y")
    scanner = listed_scanner(tmp_path, noisy_market(), as_of=old)
    scanner.refresh()
    result = offered(scanner)
    assert result.holdings == []
    assert result.problems == ["No current Shariah list", "The SPUS list is more than 7 days old"]


def test_the_route_serves_the_offer_for_the_variant_asked_for(
    client: TestClient, tmp_path: Path
) -> None:
    scanner = refreshed(tmp_path)
    app.dependency_overrides[get_scanner] = lambda: scanner
    body = client.get("/scanner/sleeve").json()
    assert body["variant"] == LABEL and body["passed"] is False
    assert [item["symbol"] for item in body["holdings"]] == [
        item.symbol for item in offered(scanner).holdings
    ]
    other = client.get("/scanner/sleeve", params={"variant": "mom_12_1 top 10"}).json()
    assert other["variant"] == "mom_12_1 top 10"
    assert client.get("/scanner/sleeve", params={"variant": "nope"}).json() is None
