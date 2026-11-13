from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app import watch
from app.system_health import Probe


def state_with(error: str | None) -> SimpleNamespace:
    probe = Probe()
    probe.results.append((datetime.now(UTC), None if error else 5.0, error))
    stream = SimpleNamespace(name="account", connects=1)
    health = MagicMock()
    health.report.return_value = [stream]
    return SimpleNamespace(probe=probe, stream_health=health, streams=MagicMock())


def test_refused_keys_are_the_one_connection_problem(trading: MagicMock) -> None:
    state = state_with('APIError: {"message": "unauthorized."}')

    assert watch.connection_problems(state, datetime.now(UTC)) == [watch.KEYS_REFUSED]


def test_with_good_keys_each_down_stream_is_its_own_problem(
    trading: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = state_with(None)
    state.streams.data_paused.return_value = True
    monkeypatch.setattr(watch, "connected", lambda item: False)

    assert watch.connection_problems(state, datetime.now(UTC)) == [
        "The account stream is disconnected"
    ]


def test_held_coins_are_named_as_the_platform_names_them() -> None:
    assert watch.platform_symbol("BTCUSD", "crypto") == "BTC/USD"
    assert watch.platform_symbol("BTC/USD", "crypto") == "BTC/USD"
    assert watch.platform_symbol("SPUS", "us_equity") == "SPUS"


def test_the_market_clock_is_read_whether_alpaca_returns_a_dict_or_a_model() -> None:
    from app.freshness import is_open

    assert is_open({"is_open": True}) and not is_open({"is_open": False})
    assert is_open(SimpleNamespace(is_open=True)) and not is_open({})
