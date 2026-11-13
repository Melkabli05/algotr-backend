from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest

from app.features.scanner import headlines, news
from app.features.scanner.news import NewsFeed, NewsStore
from app.features.scanner.tests.support import NOW, article, client_with


@pytest.fixture
def store(tmp_path: Path) -> NewsStore:
    return NewsStore(tmp_path / "news.sqlite")


def test_articles_are_stored_with_the_event_typed_from_the_headline(store: NewsStore) -> None:
    added = store.add(
        [
            article(1, "First Solar Lowers Full-Year Guidance", ["FSLR"], timedelta(hours=2)),
            article(2, "Morgan Stanley Downgrades Hasbro", ["HAS", "MAT"], timedelta(hours=3)),
        ]
    )
    found = store.of(["FSLR", "HAS", "MAT"], NOW - timedelta(days=1))
    assert added == 2 and store.count() == 2
    assert found["FSLR"][0].event.label == "Guidance cut" and found["FSLR"][0].event.veto
    assert [item.id for item in found["HAS"]] == [2] == [item.id for item in found["MAT"]]
    assert found["HAS"][0].tags == 2 and found["HAS"][0].url == "https://example.test/2"


def test_an_update_replaces_the_headline_and_is_not_counted_as_new(store: NewsStore) -> None:
    store.add([article(1, "Acme Faces Class Action", ["ACME"], timedelta(hours=2))])
    again = store.add([article(1, "Acme Wins Contract", ["ACME"], timedelta(hours=2))])
    [only] = store.of(["ACME"], NOW - timedelta(days=1))["ACME"]
    assert again == 0 and store.count() == 1
    assert (only.headline, only.event.kind) == ("Acme Wins Contract", "contract")


def test_a_stored_headline_is_typed_again_with_the_current_rules(store: NewsStore) -> None:
    store.add([article(1, "First Solar Lowers Full-Year Guidance", ["FSLR"], timedelta(hours=2))])
    with closing(store.connect()) as connection, connection:
        connection.execute("UPDATE articles SET kind = 'other', severity = 'low', label = 'Old'")
    store.retype()
    [only] = store.of(["FSLR"], NOW - timedelta(days=1))["FSLR"]
    assert (only.event.kind, only.event.label) == ("guidance", "Guidance cut")


def test_a_headline_is_returned_newest_first_and_only_since_the_cutoff(store: NewsStore) -> None:
    store.add(
        [
            article(1, "Old", ["ACME"], timedelta(days=9)),
            article(2, "Middle", ["ACME"], timedelta(days=2)),
            article(3, "New", ["ACME"], timedelta(hours=1)),
        ]
    )
    found = store.of(["ACME", "NONE"], NOW - timedelta(days=5))
    assert [item.headline for item in found["ACME"]] == ["New", "Middle"]
    assert "NONE" not in found and store.of([], NOW) == {}


def test_the_store_is_still_there_in_a_new_instance(store: NewsStore) -> None:
    store.add([article(1, "Acme Declares Dividend", ["ACME"], timedelta(hours=1))])
    again = NewsStore(store.path)
    assert again.count() == 1 and again.latest() == NOW - timedelta(hours=1)


def test_the_first_poll_reaches_back_sixty_days_and_the_next_only_a_day_before_the_last(
    store: NewsStore,
) -> None:
    client = client_with(
        [
            article(1, "Ancient", ["ACME"], timedelta(days=50)),
            article(2, "Recent", ["ACME"], timedelta(days=1)),
        ]
    )
    feed = NewsFeed(client, store)
    assert feed.poll(["ACME"], NOW) == 2
    first = client.get_news.call_args.args[0]
    assert first.start == NOW - news.BACKFILL
    client.get_news.reset_mock()
    assert feed.poll(["ACME"], NOW + timedelta(hours=1)) == 0
    second = client.get_news.call_args.args[0]
    latest = store.latest()
    assert latest is not None and second.start == latest - news.OVERLAP


def test_a_name_followed_later_is_filled_back_on_its_own(store: NewsStore) -> None:
    client = client_with(
        [
            article(1, "Acme Raises Dividend", ["ACME"], timedelta(days=1)),
            article(2, "Beta Wins Contract", ["BETA"], timedelta(days=40)),
        ]
    )
    feed = NewsFeed(client, store)
    feed.poll(["ACME"], NOW)
    assert feed.poll(["ACME", "BETA"], NOW) == 1
    assert [item.id for item in store.of(["BETA"], NOW - timedelta(days=60))["BETA"]] == [2]


def test_symbols_are_asked_for_in_groups(store: NewsStore) -> None:
    client = client_with([])
    names = [f"S{index:02d}" for index in range(45)]
    NewsFeed(client, store).poll(names, NOW)
    sizes = [len(call.args[0].symbols.split(",")) for call in client.get_news.call_args_list]
    assert sizes == [20, 20, 5]
    assert NewsFeed(client_with([]), store).poll([], NOW) == 0


def test_no_limit_is_sent_because_the_sdk_pages_only_without_one(store: NewsStore) -> None:
    client = client_with([])
    NewsFeed(client, store).poll(["ACME"], NOW)
    assert client.get_news.call_args.args[0].limit is None


def story(number: int, headline: str, age: timedelta, tags: int = 1) -> news.Headline:
    return news.Headline(
        id=number,
        time=NOW - age,
        source="benzinga",
        headline=headline,
        url=None,
        tags=tags,
        event=headlines.classify(headline, tags),
    )


def test_a_serious_setback_in_the_last_five_days_blocks_a_buy_and_names_the_event() -> None:
    result = news.badge([story(1, "Acme Lowers Full-Year Outlook", timedelta(days=4))], NOW)
    assert result.veto is not None and result.veto.label == "Guidance cut"
    assert result.count == 0 and result.worst is None


def test_a_setback_older_than_five_days_or_buried_in_commentary_does_not() -> None:
    old = story(1, "Acme Lowers Full-Year Outlook", timedelta(days=6))
    shared = story(2, "Acme Lowers Full-Year Outlook", timedelta(hours=2), tags=4)
    assert news.badge([old, shared], NOW).veto is None


def test_the_worst_recent_negative_event_and_the_count_describe_the_last_three_days() -> None:
    items = [
        story(1, "Morgan Stanley Downgrades Acme", timedelta(hours=5)),
        story(2, "Acme Faces Class Action Over Claims", timedelta(hours=9)),
        story(3, "Acme Declares Quarterly Dividend", timedelta(hours=30)),
        story(4, "Acme Wins Contract", timedelta(days=10)),
    ]
    result = news.badge(items, NOW)
    assert result.count == 3
    assert result.worst is not None and result.worst.label in {"Downgrade", "Legal action"}
    assert result.veto is None


TOPICS = [
    "Navy Contract",
    "Ohio Plant",
    "Cloud Partnership",
    "Battery Supply Deal",
    "Retail Launch",
    "Chip Licence",
    "Solar Order",
    "Rail Agreement",
    "Drug Collaboration",
    "Data Centre Lease",
    "Satellite Award",
    "Software Rollout",
    "Water Project",
    "Pipeline Contract",
    "Bank Alliance",
    "Gene Therapy Licence",
    "Port Upgrade",
    "Fleet Order",
    "Mine Agreement",
    "Studio Partnership",
]


def test_notes_from_different_brokers_with_the_same_wording_are_not_merged() -> None:
    items = [
        story(1, "Stifel Maintains Buy on Acme, Raises Price Target to $650", timedelta(hours=1)),
        story(2, "TD Cowen Maintains Buy on Acme, Raises Price Target to $650", timedelta(hours=2)),
    ]
    assert news.badge(items, NOW).count == 2


def test_attention_compares_the_last_three_days_with_the_weeks_before() -> None:
    quiet = [
        story(n, f"{topic} Is Announced By Acme", timedelta(days=4 + 2 * n))
        for n, topic in enumerate(TOPICS)
    ]
    busy = [
        story(100 + n, f"Acme Signs {topic}", timedelta(hours=n + 1))
        for n, topic in enumerate(TOPICS[:3])
    ]
    result = news.badge([*quiet, *busy], NOW)
    assert result.count == 3
    assert result.attention is not None and result.attention > 1
    assert news.badge(busy, NOW).attention is None
