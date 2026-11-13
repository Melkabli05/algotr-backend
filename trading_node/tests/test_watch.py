import json

from trading_node.actors.watch import Leases, WatchListener


def test_a_lease_lapses_unless_it_is_asked_for_again() -> None:
    now = [0.0]
    leases = Leases([], ttl=90.0, clock=lambda: now[0])
    leases.renew(["AAPL", "TSLA"])
    now[0] = 60.0
    leases.renew(["AAPL"])
    now[0] = 100.0
    assert leases.active() == ["AAPL"]


def test_the_fixed_symbols_are_never_leased() -> None:
    leases = Leases(["SPY"])
    leases.renew(["SPY", "AAPL"])
    assert leases.active() == ["AAPL"]


def test_only_the_most_recently_asked_for_symbols_fit_under_the_limit() -> None:
    now = [0.0]
    leases = Leases([], limit=2, clock=lambda: now[0])
    for symbol in ("A", "B", "C"):
        now[0] += 1
        leases.renew([symbol])
    assert leases.active() == ["C", "B"]


def test_the_listener_renews_leases_from_announcements_and_ignores_the_rest() -> None:
    leases = Leases([])
    listener = WatchListener("postgresql+psycopg://localhost/none", leases)
    listener.take(json.dumps({"symbols": ["SPUS", "SOL/USD"]}))
    listener.take("not json")
    listener.take(json.dumps({"other": 1}))
    assert sorted(leases.active()) == ["SOL/USD", "SPUS"]
