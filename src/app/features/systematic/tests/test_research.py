from unittest.mock import create_autospec

import pytest
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

from app.features.systematic import research
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.tests.support import serve, uptrend

BTC = STRATEGIES["btc-trend-200"]
STOCKS = create_autospec(StockHistoricalDataClient, instance=True)
CRYPTO = create_autospec(CryptoHistoricalDataClient, instance=True)


def test_recent_keeps_the_evaluation_window_plus_the_warmup() -> None:
    frame = uptrend(BTC.universe, 2_000)
    assert len(research.recent(frame, BTC, 2)) == 2 * 365 + 220
    assert research.recent(frame, BTC, 2).index[-1] == frame.index[-1]


def test_recent_returns_everything_when_history_is_shorter_than_asked() -> None:
    frame = uptrend(BTC.universe, 300)
    assert len(research.recent(frame, BTC, 5)) == 300


def test_equity_strategies_count_252_days_to_the_year() -> None:
    etf = STRATEGIES["halal-etf-trend"]
    frame = uptrend(etf.universe, 2_000)
    assert len(research.recent(frame, etf, 3)) == 3 * 252 + 220


def test_results_are_cached_per_strategy_years_and_fee(monkeypatch: pytest.MonkeyPatch) -> None:

    fetched: list[object] = []

    def fake(_s: object, _c: object, symbols: list[str], _y: float) -> object:
        fetched.append(symbols)
        return uptrend(symbols, 700)

    serve(monkeypatch, fake)
    first = research.research(None, None, BTC, 2)  # type: ignore[arg-type]
    assert research.research(None, None, BTC, 2) is first  # type: ignore[arg-type]
    assert research.research(None, None, BTC, 1) is not first  # type: ignore[arg-type]
    assert research.research(None, None, BTC, 2, 0).fee_bps == 0  # type: ignore[arg-type]
    assert len(fetched) == 1


def test_clearing_forgets_everything() -> None:
    research.RESULTS[("x", 1, 1.0, 4, None)] = None  # type: ignore[assignment]
    research.clear()
    assert len(research.RESULTS) == 0


def test_every_strategys_returns_are_offered_to_the_drift_monitor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.systematic.tests.support import fake_history

    fake_history(monkeypatch)
    found = research.StrategyReturns(STOCKS, CRYPTO).streams()
    assert [stream.name for stream in found] == list(STRATEGIES)
    assert {stream.kind for stream in found} == {"strategy"}
    assert all(len(stream.returns) > 300 for stream in found)


def test_a_strategy_whose_prices_cannot_be_read_is_left_out_and_the_rest_still_come(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from app.quant.history import NotEnoughHistoryError

    def daily_closes(_s: object, _c: object, symbols: list[str], _y: float) -> object:
        if "/" in symbols[0]:
            raise NotEnoughHistoryError("Alpaca has no daily bars")
        return uptrend(symbols, 700)

    serve(monkeypatch, daily_closes)
    found = research.StrategyReturns(STOCKS, CRYPTO).streams()
    assert [stream.name for stream in found] == [
        "halal-etf-trend",
        "halal-multi-trend",
        "halal-etf-rotation",
    ]
    assert "No returns of btc-trend-200 to watch" in caplog.text
