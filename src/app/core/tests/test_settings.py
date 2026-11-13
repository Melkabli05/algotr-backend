import pytest
from alpaca.data.enums import DataFeed
from pydantic import ValidationError

from app.core.settings import Settings


def settings() -> Settings:
    return Settings(alpaca_api_key="key", alpaca_api_secret="secret", _env_file=None)


def test_the_data_feed_is_iex_unless_chosen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_FEED", raising=False)
    assert settings().data_feed is DataFeed.IEX


def test_the_data_feed_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_FEED", "sip")
    assert settings().data_feed is DataFeed.SIP


@pytest.mark.parametrize("feed", ["otc", "delayed_sip", "SIP", ""])
def test_a_feed_the_stream_cannot_use_is_refused(
    monkeypatch: pytest.MonkeyPatch, feed: str
) -> None:
    monkeypatch.setenv("DATA_FEED", feed)
    with pytest.raises(ValidationError, match="data_feed"):
        settings()
