import re
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.core import sidecar
from app.features.scanner import headlines, sentiment
from app.features.scanner.headlines import Direction, Event, Severity
from app.features.scanner.news import NewsStore
from app.features.scanner.sentiment import Batch, Scorer, SentimentError
from app.features.scanner.tests.support import NOW, article

CLOCK = datetime(2026, 11, 4, 9, 30, tzinfo=UTC)
POSITIVE, NEGATIVE, NEUTRAL = (0.9, 0.05, 0.05), (0.02, 0.9, 0.08), (0.1, 0.1, 0.8)


class Classifier:
    """A stand-in for FinBERT: it answers as told for each text and keeps what it was asked."""

    def __init__(self, **answers: tuple[float, float, float]) -> None:
        self.answers = answers
        self.asked: list[list[str]] = []
        self.spreads: dict[str, float] | None = None

    def __call__(self, texts: list[str]) -> Batch:
        self.asked.append(list(texts))
        spreads = None if self.spreads is None else [self.spreads[text] for text in texts]
        return Batch(sentiment.MODEL, [self.answers.get(text, NEUTRAL) for text in texts], spreads)


@pytest.fixture
def news(tmp_path: Path) -> NewsStore:
    return NewsStore(tmp_path / "news.sqlite")


def scorer(news: NewsStore, classify: Classifier, usable: bool = True) -> Scorer:
    return Scorer(news, classify, lambda: usable, lambda: CLOCK)


def add(news: NewsStore, *items: tuple[int, str, list[str]]) -> None:
    news.add([article(id_, text, symbols, timedelta(hours=id_)) for id_, text, symbols in items])


def event(direction: Direction, severity: Severity, label: str = "x") -> Event:
    return Event(kind="other", direction=direction, severity=severity, label=label)


def test_the_risk_follows_how_serious_a_setback_the_rule_typing_reads() -> None:
    assert sentiment.risk_of(event("negative", "high")) == 5
    assert sentiment.risk_of(event("neutral", "high", headlines.PENDING)) == 5
    assert sentiment.risk_of(event("negative", "medium")) == 4
    assert sentiment.risk_of(event("negative", "low")) == 3
    assert sentiment.risk_of(event("neutral", "medium")) == 2
    assert sentiment.risk_of(event("positive", "medium")) == 1
    assert sentiment.risk_of(event("positive", "low")) == 1
    assert sentiment.risk_of(event("neutral", "low")) == 1


@pytest.mark.parametrize(
    ("text", "risk"),
    [
        ("First Solar Lowers Full-Year Guidance", 5),
        ("Morgan Stanley Downgrades Hasbro", 4),
        ("Deutsche Bank Lowers Price Target on Nike", 3),
        ("Acme Announces Acquisition of Beta", 2),
        ("Acme Beats Earnings Estimates", 1),
    ],
)
def test_the_real_rules_give_the_risk_of_a_headline(text: str, risk: int) -> None:
    now = sentiment.record(text, POSITIVE, sentiment.MODEL, CLOCK)
    assert now.risk == risk and now.risk_basis == "rules"


def test_a_record_has_the_score_confidence_model_time_and_hash_the_format_asks_for() -> None:
    found = sentiment.record("  Acme Wins Contract  ", (0.7, 0.1, 0.2), "m@1", CLOCK)
    assert found.score == pytest.approx(0.6) and found.confidence == 0.7
    assert (found.positive, found.negative, found.neutral) == (0.7, 0.1, 0.2)
    assert found.model == "m@1" and found.processed == CLOCK
    assert re.fullmatch(r"[0-9a-f]{64}", found.text_hash)
    assert found.text_hash == sentiment.digest("Acme Wins Contract")
    assert -1.0 <= found.score <= 1.0 and 1 <= found.risk <= 5


def test_the_score_runs_from_minus_one_to_plus_one() -> None:
    assert sentiment.record("a", (1.0, 0.0, 0.0), "m", CLOCK).score == 1.0
    assert sentiment.record("a", (0.0, 1.0, 0.0), "m", CLOCK).score == -1.0
    assert sentiment.record("a", (0.0, 0.0, 1.0), "m", CLOCK).score == 0.0


def test_a_confidence_under_a_fifth_short_of_certainty_is_marked_for_the_next_stage() -> None:
    low = sentiment.record("a", (0.5, 0.3, 0.2), "m", CLOCK)
    edge = sentiment.record("a", (0.8, 0.1, 0.1), "m", CLOCK)
    high = sentiment.record("a", (0.81, 0.1, 0.09), "m", CLOCK)
    assert (low.fallback, edge.fallback, high.fallback) == (True, False, False)


def test_the_same_text_has_the_same_hash_whatever_surrounds_it() -> None:
    assert sentiment.digest("Acme Wins") == sentiment.digest("  Acme Wins\n")
    assert sentiment.digest("Acme Wins") != sentiment.digest("Acme Wins.")


def test_every_new_headline_is_scored_and_stored_with_its_record(news: NewsStore) -> None:
    add(news, (1, "Acme Wins Contract", ["ACME"]), (2, "Beta Cuts Guidance", ["BETA"]))
    classify = Classifier(**{"Acme Wins Contract": POSITIVE, "Beta Cuts Guidance": NEGATIVE})
    assert scorer(news, classify).score_new() == 2
    items = news.of(["ACME", "BETA"], NOW - timedelta(days=1))
    found = {name: item[0].headline for name, item in items.items()}
    assert found == {"ACME": "Acme Wins Contract", "BETA": "Beta Cuts Guidance"}
    cached = scorer(news, classify).store.cached(
        {sentiment.digest(text) for text in found.values()}, sentiment.MODEL
    )
    assert sorted(round(record.score, 2) for record in cached.values()) == [-0.88, 0.85]


def test_a_text_that_was_scored_is_not_asked_for_again(news: NewsStore) -> None:
    add(news, (1, "Acme Wins Contract", ["ACME"]))
    classify = Classifier()
    both = scorer(news, classify)
    both.score_new()
    add(news, (2, "Beta Wins Contract", ["BETA"]))
    assert both.score_new() == 1 and classify.asked == [
        ["Acme Wins Contract"],
        ["Beta Wins Contract"],
    ]
    assert both.score_new() == 0 and len(classify.asked) == 2


def test_the_same_text_on_two_articles_is_scored_once_and_both_are_linked(news: NewsStore) -> None:
    add(news, (1, "Market Wrap", ["AAA"]), (2, "Market Wrap", ["BBB"]), (3, "Other", ["CCC"]))
    classify = Classifier()
    assert scorer(news, classify).score_new() == 3
    asked = sorted(text for batch in classify.asked for text in batch)
    assert asked == ["Market Wrap", "Other"]
    counts = scorer(news, classify).store.counts(sentiment.MODEL)
    assert counts[0] == 2 and counts[1] == 0


def test_a_later_article_with_a_text_already_scored_costs_no_inference(news: NewsStore) -> None:
    add(news, (1, "Market Wrap", ["AAA"]))
    classify = Classifier(**{"Market Wrap": NEGATIVE})
    found = scorer(news, classify)
    found.score_new()
    add(news, (2, "Market Wrap", ["BBB"]))
    assert found.score_new() == 1 and len(classify.asked) == 1
    [shown] = found.attach(news.of(["BBB"], NOW - timedelta(days=1))["BBB"])
    assert shown.sentiment is not None and shown.sentiment.score < -0.8


def test_a_blank_headline_is_never_sent_to_the_model(news: NewsStore) -> None:
    add(news, (1, "   ", ["ACME"]), (2, "Acme Wins", ["ACME"]))
    classify = Classifier()
    assert scorer(news, classify).score_new() == 1
    assert classify.asked == [["Acme Wins"]]


def test_the_backlog_is_scored_newest_first_in_chunks(
    news: NewsStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sentiment, "CHUNK", 2)
    add(news, *[(n, f"Story {n}", ["ACME"]) for n in range(1, 6)])
    classify = Classifier()
    assert scorer(news, classify).score_new() == 5
    assert classify.asked == [["Story 1", "Story 2"], ["Story 3", "Story 4"], ["Story 5"]]


def test_a_model_that_cannot_run_leaves_the_headlines_for_the_next_try(
    news: NewsStore, caplog: pytest.LogCaptureFixture
) -> None:
    add(news, (1, "Acme Wins", ["ACME"]))

    def broken(texts: list[str]) -> Batch:
        raise SentimentError("the sidecar is not set up")

    failing = Scorer(news, broken, lambda: True, lambda: CLOCK)
    assert failing.score_new() == 0
    assert failing.error == "SentimentError: the sidecar is not set up"
    assert "Sentiment scoring failed" in caplog.text
    assert failing.status().backlog == 1
    assert scorer(news, Classifier()).score_new() == 1


def test_without_the_model_nothing_is_scored_and_nothing_is_asked(news: NewsStore) -> None:
    add(news, (1, "Acme Wins", ["ACME"]))
    classify = Classifier()
    idle = scorer(news, classify, usable=False)
    assert idle.score_new() == 0 and classify.asked == [] and not idle.status().enabled


def test_a_run_already_going_is_left_alone(news: NewsStore) -> None:
    add(news, (1, "Acme Wins", ["ACME"]))
    classify = Classifier()
    busy = scorer(news, classify)
    busy.lock.acquire()
    assert busy.score_new() == 0 and classify.asked == []


def test_a_headline_comes_back_with_its_record_when_it_has_one(news: NewsStore) -> None:
    add(news, (1, "Acme Wins Contract", ["ACME"]), (2, "Beta Cuts Guidance", ["BETA"]))
    classify = Classifier(**{"Acme Wins Contract": POSITIVE})
    first = scorer(news, classify)
    first.score_new()
    shown = first.attach(news.of(["ACME"], NOW - timedelta(days=1))["ACME"])
    assert shown[0].sentiment is not None and shown[0].sentiment.score == pytest.approx(0.85)
    unknown = news.of(["BETA"], NOW - timedelta(days=1))["BETA"]
    fresh = Scorer(NewsStore(news.path.with_name("other.sqlite")), classify, lambda: True)
    assert fresh.attach(unknown)[0].sentiment is None


def test_the_status_counts_records_the_backlog_and_the_share_below_the_threshold(
    news: NewsStore,
) -> None:
    add(news, (1, "A story", ["AAA"]), (2, "B story", ["BBB"]), (3, "C story", ["CCC"]))
    classify = Classifier(**{"A story": POSITIVE, "B story": (0.5, 0.3, 0.2)})
    found = scorer(news, classify)
    assert (
        found.status().backlog == 3 and found.status().scored == 0 and found.status().below is None
    )
    found.score_new()
    status = found.status()
    assert (status.scored, status.backlog, status.enabled) == (3, 0, True)
    assert (
        status.below == pytest.approx(1 / 3)
        and status.last == CLOCK
        and status.model == sentiment.MODEL
    )


def test_a_record_of_another_model_is_not_used(news: NewsStore) -> None:
    add(news, (1, "Acme Wins", ["ACME"]))
    first = scorer(news, Classifier())
    first.score_new()
    with closing(first.store.connect()) as connection, connection:
        connection.execute("UPDATE sentiment_cache SET model = 'someone/else@1'")
    item = news.of(["ACME"], NOW - timedelta(days=1))["ACME"]
    assert first.attach(item)[0].sentiment is None


def test_the_records_and_their_links_survive_a_restart(news: NewsStore) -> None:
    add(news, (1, "Acme Wins", ["ACME"]))
    scorer(news, Classifier()).score_new()
    again = Classifier()
    assert scorer(NewsStore(news.path), again).score_new() == 0 and again.asked == []


class Runner:
    """A stand-in for the sidecar that answers every job with one result and keeps the jobs."""

    def __init__(self, result: sidecar.JobResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(
        self, task: str, params: dict[str, Any], files: Any, timeout: float = 0
    ) -> sidecar.JobResult:
        self.calls.append((task, params))
        return self.result


def reply(**fields: Any) -> sidecar.JobResult:
    body = {
        "labels": ["positive", "negative", "neutral"],
        "probabilities": [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]],
        "device": "mps",
        "revision": sentiment.REVISION,
        **fields,
    }
    return sidecar.JobResult(task="sentiment", status="ok", seconds=1.0, data=body)


def test_the_sidecars_answer_is_read_by_label_not_by_position() -> None:
    swapped = reply(labels=["neutral", "positive", "negative"], probabilities=[[0.1, 0.7, 0.2]])
    batch = sentiment.finbert(["x"], Runner(swapped))
    assert batch == Batch(sentiment.MODEL, [(0.7, 0.2, 0.1)])


def test_the_job_asks_for_the_task_with_the_texts_and_an_absolute_model_folder() -> None:
    run = Runner(reply())
    sentiment.finbert(["a", "b"], run)
    [(task, params)] = run.calls
    assert task == "sentiment" and Path(params["model_dir"]).is_absolute()
    assert params["batch"] == sentiment.BATCH


def test_weights_other_than_the_pinned_ones_are_refused() -> None:
    with pytest.raises(SentimentError, match="0000000 ran; 0bc2ac1 is pinned"):
        sentiment.finbert(["x"], Runner(reply(revision="0000000" + "0" * 33)))


def test_a_failed_job_is_an_error_with_its_reason() -> None:
    failed = sidecar.JobResult(task="sentiment", status="error", seconds=1.0, error="boom")
    with pytest.raises(SentimentError, match="boom"):
        sentiment.finbert(["x"], Runner(failed))
    nothing = sidecar.JobResult(task="sentiment", status="ok", seconds=1.0, data=None)
    with pytest.raises(SentimentError, match="returned nothing"):
        sentiment.finbert(["x"], Runner(nothing))


def test_the_pinned_revision_is_the_one_the_fetch_script_downloads() -> None:
    fetch = Path("finrl_sidecar/fetch_finbert.py").read_text()
    assert f'REVISION = "{sentiment.REVISION}"' in fetch and f'REPO = "{sentiment.REPO}"' in fetch


def test_the_model_counts_as_ready_only_with_the_sidecar_and_the_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sentiment, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(sidecar, "available", lambda: True)
    assert not sentiment.ready()
    (tmp_path / "model.safetensors").write_text("x")
    assert sentiment.ready()
    monkeypatch.setattr(sidecar, "available", lambda: False)
    assert not sentiment.ready()


def test_the_daily_mood_reads_scored_headlines_of_the_named_stocks_once_each(
    news: NewsStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date

    from app.features.scanner import mood

    first = datetime(2026, 11, 2, 14, 0, tzinfo=UTC)
    news.add(
        [
            {
                **article(1, "Acme Wins Contract", ["ACME", "BETA"], timedelta(0)),
                "created_at": first.isoformat(),
            },
            {
                **article(2, "Gamma Cuts Guidance", ["GAMMA"], timedelta(0)),
                "created_at": first.isoformat(),
            },
        ]
    )
    classify = Classifier(**{"Acme Wins Contract": POSITIVE, "Gamma Cuts Guidance": NEGATIVE})
    classify.spreads = {"Acme Wins Contract": 0.01, "Gamma Cuts Guidance": 0.05}
    monkeypatch.setattr(sentiment, "MIN_GATE_SAMPLE", 2)
    scorer(news, classify).score_new()
    found = mood.scored(
        news, ["ACME", "BETA"], first - timedelta(days=1), first + timedelta(days=1)
    )
    assert found["id"].tolist() == [1]
    daily = mood.daily(found, [date(2026, 11, 2)])
    assert daily.loc["2026-11-02", "sentiment"] == pytest.approx(0.85, abs=0.01)


def test_the_gate_lets_through_the_surest_third_weighted_by_how_sure(
    news: NewsStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.features.scanner import mood

    spreads = pd.Series([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, None])
    weights = mood.gate_weights(spreads, 0.04)
    assert weights.round(3).tolist() == [0.75, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert mood.gate_weights(spreads, None).tolist() == [0.0] * 7
    headlines = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "session": [date(2026, 11, 2)] * 3,
            "score": [0.8, -0.6, -0.9],
            "risk": [1, 3, 5],
            "weight": [0.75, 0.25, 0.0],
        }
    )
    day = mood.daily(headlines, [date(2026, 11, 2)]).iloc[0]
    assert day["sentiment"] == pytest.approx((0.8 * 0.75 - 0.6 * 0.25) / 1.0)
    assert (day["news_risk"], day["headlines"]) == (3.0, 3)
    closed = mood.daily(headlines.assign(weight=0.0), [date(2026, 11, 2)]).iloc[0]
    assert np.isnan(closed["sentiment"]) and closed["news_risk"] == 3.0


def test_old_headlines_are_measured_and_the_gate_calibrated_on_all_of_them(
    news: NewsStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sentiment, "MIN_GATE_SAMPLE", 3)
    news.add([article(i, f"Headline {i}", ["ACME"], timedelta(0)) for i in range(1, 4)])
    classify = Classifier()
    run = scorer(news, classify)
    run.score_new()
    assert run.store.gate() is None
    classify.spreads = {"Headline 1": 0.01, "Headline 2": 0.05, "Headline 3": 0.09}
    assert run.measure() == 3 and run.measure() == 0
    tau = run.store.gate()
    assert tau is not None and 0.01 < tau < 0.05
    found = run.store.cached({sentiment.digest("Headline 1")}, sentiment.MODEL)
    assert next(iter(found.values())).uncertainty == 0.01
