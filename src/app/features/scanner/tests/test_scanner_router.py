from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.features.scanner import rank
from app.features.scanner.lists import Source
from app.features.scanner.news import NewsFeed, NewsStore
from app.features.scanner.router import get_scanner
from app.features.scanner.sentiment import MODEL, Batch, Scorer
from app.features.scanner.service import Scanner
from app.features.scanner.tests.support import (
    Market,
    article,
    asset,
    client_with,
    last_business_day,
    listed_scanner,
    noisy_market,
    spus_csv,
    trading_with,
    website,
    yesterday,
)
from app.main import app

URL = "https://lists.test/SPUS.csv"


@pytest.fixture
def scanner(client: TestClient, tmp_path: Path) -> Iterator[Scanner]:
    market = Market()
    market.add("AAA")
    market.add("BBB", volume=1_000)
    built = Scanner(
        trading_with([asset("AAA"), asset("BBB")]),
        market.client(),
        root=tmp_path / "scanner",
        sources=[Source(name="SPUS", url=URL)],
        http=website({URL: spus_csv(["BBB"])}),
    )
    app.dependency_overrides[get_scanner] = lambda: built
    yield built


def test_the_status_is_empty_before_the_first_refresh(client: TestClient, scanner: Scanner) -> None:
    assert client.get("/scanner/status").json() == {
        "running": None,
        "last": None,
        "next_run": None,
        "news": {
            "enabled": False,
            "polled": None,
            "articles": 0,
            "following": 0,
            "error": None,
            "sec": False,
            "filings_at": None,
            "filings_error": None,
            "sentiment": {
                "enabled": False,
                "model": "ProsusAI/finbert@0bc2ac1",
                "scored": 0,
                "backlog": 0,
                "below": None,
                "last": None,
                "error": None,
            },
        },
        "review": None,
        "crypto": {"enabled": False, "coins": 0, "as_of": None, "refreshed": None, "error": None},
    }


def test_the_status_shows_the_funnel_and_the_lists_after_a_refresh(
    client: TestClient, scanner: Scanner
) -> None:
    scanner.refresh()
    last = client.get("/scanner/status").json()["last"]
    assert last["status"] == "ok"
    assert last["funnel"] == {
        "assets": 2,
        "tradable": 2,
        "liquid": 1,
        "listed": 1,
        "listed_liquid": 0,
        "panel": 2,
    }
    assert [(item["name"], item["symbols"], item["stale"]) for item in last["lists"]] == [
        ("SPUS", 1, False)
    ]


def test_a_refresh_can_be_started_from_the_api(client: TestClient, scanner: Scanner) -> None:
    response = client.post("/scanner/run")
    assert response.status_code == 202
    assert len(scanner.scheduler.get_jobs()) == 1


def test_a_second_start_while_one_runs_is_a_conflict(client: TestClient, scanner: Scanner) -> None:
    with scanner.lock:
        response = client.post("/scanner/run")
    assert response.status_code == 409
    assert response.json()["detail"] == "A refresh is already running"


def test_an_evaluation_needs_a_refreshed_panel(client: TestClient, scanner: Scanner) -> None:
    response = client.post("/scanner/evaluate")
    assert response.status_code == 422
    assert "price history" in response.json()["detail"]
    assert client.get("/scanner/evaluation").json() is None


def test_an_evaluation_is_run_and_read_back(client: TestClient, tmp_path: Path) -> None:
    market = Market()
    names = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    for index, name in enumerate(names):
        market.add(name, drift=0.0004 + 0.0001 * index, noise=0.012)
    market.through = last_business_day(yesterday())
    built = Scanner(
        trading_with([asset(name) for name in names]),
        market.client(),
        root=tmp_path / "scanner",
        sources=[Source(name="SPUS", url=URL)],
        http=website({URL: spus_csv(names)}),
    )
    app.dependency_overrides[get_scanner] = lambda: built
    built.refresh()
    posted = client.post("/scanner/evaluate").json()
    assert [variant["label"] for variant in posted["variants"]][:2] == [
        "mom_12_1 top 10",
        "mom_12_1 top 20",
    ]
    assert client.get("/scanner/evaluation").json() == posted


def test_the_latest_scan_the_reasons_and_the_shadow_report_are_served(
    client: TestClient, tmp_path: Path
) -> None:
    built = listed_scanner(tmp_path, noisy_market())
    app.dependency_overrides[get_scanner] = lambda: built
    assert client.get("/scanner/latest").json() is None
    assert client.get("/scanner/symbol/AAA").status_code == 404
    built.refresh()
    latest = client.get("/scanner/latest").json()
    assert latest["variant"] == "rank_avg top 20"
    assert latest["funnel"]["listed"] == 6
    assert latest["candidates"][0]["lists"] == ["SPUS"]
    reasons = client.get("/scanner/symbol/aaa").json()
    assert reasons["symbol"] == "AAA" and reasons["verdict"] in {"picked", "eligible"}
    shadow = client.get("/scanner/shadow").json()
    assert shadow["horizons"] == [1, 5, 21]
    assert len(shadow["summary"]) == len(rank.VARIANTS)


def test_a_headline_comes_with_its_sentiment_record_in_the_format_asked_for(
    client: TestClient, tmp_path: Path
) -> None:
    store = NewsStore(tmp_path / "news.sqlite")
    store.add([article(1, "Acme Lowers Full-Year Outlook", ["AAA"], timedelta(hours=1))])
    found = listed_scanner(
        tmp_path,
        noisy_market(),
        news=NewsFeed(client_with([]), store),
        sentiment=Scorer(
            store, lambda texts: Batch(MODEL, [(0.1, 0.8, 0.1)] * len(texts)), lambda: True
        ),
    )
    assert found.watch.sentiment is not None
    found.watch.sentiment.score_new()
    app.dependency_overrides[get_scanner] = lambda: found
    try:
        [item] = client.get("/scanner/news/AAA").json()["headlines"]
    finally:
        app.dependency_overrides.pop(get_scanner, None)
    record = item["sentiment"]
    assert set(record) == {
        "text_hash",
        "model",
        "score",
        "risk",
        "risk_basis",
        "confidence",
        "positive",
        "negative",
        "neutral",
        "fallback",
        "processed",
        "uncertainty",
    }
    assert record["score"] == pytest.approx(-0.7) and record["confidence"] == 0.8
    assert (record["risk"], record["fallback"], record["model"]) == (5, False, MODEL)
    assert len(record["text_hash"]) == 64


def test_the_veto_study_needs_news_and_reads_back_as_none_before_a_run(
    client: TestClient, scanner: Scanner
) -> None:
    assert client.get("/scanner/vetoes").json() is None
    response = client.post("/scanner/vetoes")
    assert response.status_code == 422 and "News is not set up" in response.json()["detail"]


def test_the_veto_study_runs_on_the_stored_panel_and_is_kept_in_the_audit_log(
    client: TestClient, tmp_path: Path
) -> None:
    store = NewsStore(tmp_path / "news.sqlite")
    store.add([article(1, "Acme Lowers Full-Year Guidance", ["AAA"], timedelta(days=100))])
    built = listed_scanner(tmp_path, noisy_market(), news=NewsFeed(client_with([]), store))
    built.refresh()
    app.dependency_overrides[get_scanner] = lambda: built
    try:
        posted = client.post("/scanner/vetoes").json()
        read = client.get("/scanner/vetoes").json()
    finally:
        app.dependency_overrides.pop(get_scanner, None)
    assert posted["horizons"] is not None and posted["notes"]
    assert read == posted
    assert [e.kind for e in built.audit.entries()].count("veto-study") == 1
