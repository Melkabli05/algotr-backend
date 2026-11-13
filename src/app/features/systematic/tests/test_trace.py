import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.features.systematic import backtest, signals, trace
from app.features.systematic.catalog import STRATEGIES
from app.quant import history

BTC = STRATEGIES["btc-trend-200"]
NEXT_OPEN = BTC.model_copy(update={"fills": "next_open", "fee_bps": 0})
DONCHIAN = STRATEGIES["crypto-donchian"]
ROTATION = STRATEGIES["halal-etf-rotation"]


def swings(symbols: list[str], days: int = 900, rise: bool = False) -> history.Prices:
    """Prices that rise, fall and rise again, so a trend rule trades more than once."""
    rng = np.random.default_rng(4)
    index = pd.date_range("2022-01-01", periods=days)
    wave = np.sin(np.linspace(0, 6 * np.pi, days)) * 0.02
    if rise:
        wave = wave + np.where(np.arange(days) < 260, 0.02, 0.0)
    closes = {
        symbol: 100 * np.exp(np.cumsum(wave * 0.3 + rng.normal(0.0004, 0.004, days)))
        for symbol in symbols
    }
    close = pd.DataFrame(closes, index=index)
    opens = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, close.shape))
    return history.Prices(close, opens)


def candles(prices: history.Prices, symbol: str) -> pd.DataFrame:
    close, opened = prices.closes[symbol], prices.opens[symbol]
    return pd.DataFrame(
        {
            "open": opened,
            "high": np.maximum(opened, close) * 1.001,
            "low": np.minimum(opened, close) * 0.999,
            "close": close,
            "volume": 1000.0,
        }
    )


@pytest.mark.parametrize("spec", [BTC.model_copy(update={"fee_bps": 0}), NEXT_OPEN])
@pytest.mark.parametrize("start", ["swings", "rising"])
def test_the_trades_compound_to_the_backtests_total_return(spec, start) -> None:  # type: ignore[no-untyped-def]
    prices = swings(spec.universe, rise=start == "rising")
    found = trace.trace(spec, "BTC/USD", prices, candles(prices, "BTC/USD"))
    result = backtest.run(prices.closes, spec, fee_bps=0, opens=prices.opens)
    assert len(found.trades) >= 2
    compounded = np.prod([1 + item.gain for item in found.trades]) - 1
    assert compounded == pytest.approx(result.metrics.total_return, rel=1e-9)


def test_a_trade_enters_where_the_backtest_fills_it() -> None:
    prices = swings(BTC.universe)
    held = signals.weights(prices.closes, NEXT_OPEN).shift(1).fillna(0.0)["BTC/USD"]
    held.iloc[: signals.warmup_days(NEXT_OPEN)] = 0.0
    entered = held[(held > 0) & (held.shift(1, fill_value=0.0) <= 0)].index[1]
    second = trace.trace(NEXT_OPEN, "BTC/USD", prices, candles(prices, "BTC/USD")).trades[1]
    assert second.entry == entered.date()
    assert second.entry_price == pytest.approx(prices.opens.loc[entered, "BTC/USD"])
    at_close = trace.trace(BTC, "BTC/USD", prices, candles(prices, "BTC/USD")).trades[1]
    before = entered - pd.Timedelta(days=1)
    assert at_close.entry == before.date()
    assert at_close.entry_price == pytest.approx(prices.closes.loc[before, "BTC/USD"])


def test_the_last_stretch_stays_open_and_is_valued_at_the_last_close() -> None:
    prices = swings(BTC.universe, 700)
    found = trace.trace(NEXT_OPEN, "BTC/USD", prices, candles(prices, "BTC/USD"))
    last = found.trades[-1]
    if last.open:
        assert last.exit is None and last.exit_price is None
        assert last.gain == pytest.approx(prices.closes["BTC/USD"].iloc[-1] / last.entry_price - 1)
    assert all(not item.open for item in found.trades[:-1])


def test_the_trend_line_is_empty_during_the_warmup_and_is_the_average_after() -> None:
    prices = swings(BTC.universe)
    found = trace.trace(BTC, "BTC/USD", prices, candles(prices, "BTC/USD"))
    assert found.lines == ["SMA 200"]
    assert found.bars[10].lines["SMA 200"] is None
    expected = prices.closes["BTC/USD"].rolling(200).mean().iloc[300]
    assert found.bars[300].lines["SMA 200"] == pytest.approx(expected)
    assert found.bars[300].held in (0.0, 1.0)


def test_each_rule_reports_its_own_readings() -> None:
    donchian = trace.trace(
        DONCHIAN,
        "ETH/USD",
        swings(DONCHIAN.universe),
        candles(swings(DONCHIAN.universe), "ETH/USD"),
    )
    assert donchian.lines == ["High 60", "Midpoint 60"]
    assert donchian.values == ["Votes", "Volatility scale"]
    rotation_prices = swings(ROTATION.universe, 600)
    rotation = trace.trace(ROTATION, "SPUS", rotation_prices, candles(rotation_prices, "SPUS"))
    assert rotation.lines == [] and rotation.values == ["Gain 126d"]


def test_it_reports_no_gaps_as_numbers() -> None:
    prices = swings(BTC.universe, 300)
    found = trace.trace(BTC, "BTC/USD", prices, candles(prices, "BTC/USD"))
    assert found.model_dump_json().count("NaN") == 0


def test_the_endpoint_traces_a_symbol_the_rule_trades(
    client: TestClient,
    stock_data: object,
    crypto_data: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = swings(BTC.universe, 600)
    monkeypatch.setattr(history, "daily_prices", lambda *_: prices)
    monkeypatch.setattr(trace, "bars_of", lambda *_: candles(prices, "BTC/USD"))
    response = client.get(
        "/systematic/strategies/btc-trend-200/trace", params={"symbol": "BTC/USD", "years": 1}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "BTC/USD" and body["fills"] == "close"
    assert len(body["bars"]) == 365 + 220
    assert body["trades"]


def test_the_endpoint_refuses_a_symbol_the_rule_does_not_trade(client: TestClient) -> None:
    response = client.get("/systematic/strategies/btc-trend-200/trace", params={"symbol": "SPUS"})
    assert response.status_code == 404


def test_the_catalog_lists_every_rule_with_its_universe(client: TestClient) -> None:
    found = client.get("/systematic/strategies").json()
    assert {item["name"] for item in found} == set(STRATEGIES)
    assert next(item for item in found if item["name"] == "btc-trend-200")["universe"] == [
        "BTC/USD"
    ]


def test_each_trade_says_what_the_rule_read_when_it_moved() -> None:
    prices = swings(BTC.universe)
    found = trace.trace(BTC, "BTC/USD", prices, candles(prices, "BTC/USD"))
    later = found.trades[1]
    assert (
        later.why_entry.startswith("Closed at ") and "above its 200-day average" in later.why_entry
    )
    assert later.why_exit is None or "at or below its 200-day average" in later.why_exit
    assert found.trades[0].why_entry in (
        "Already holding when the test begins",
        found.trades[0].why_entry,
    )
    donchian_prices = swings(DONCHIAN.universe)
    donchian = trace.trace(
        DONCHIAN, "ETH/USD", donchian_prices, candles(donchian_prices, "ETH/USD")
    )
    assert any("Breakout vote rose to" in item.why_entry for item in donchian.trades)
    rotation_prices = swings(ROTATION.universe, 600)
    rotation = trace.trace(ROTATION, "SPUS", rotation_prices, candles(rotation_prices, "SPUS"))
    assert all(
        "over 126 days" in item.why_entry or "begins" in item.why_entry for item in rotation.trades
    )


def test_a_trade_already_open_when_the_window_starts_says_so() -> None:
    prices = swings(BTC.universe, rise=True)
    found = trace.trace(NEXT_OPEN, "BTC/USD", prices, candles(prices, "BTC/USD"))
    assert found.trades[0].why_entry == "Already holding when the test begins"
    assert found.trades[0].entry == found.evaluated_from


def test_a_day_without_a_bar_is_left_out_instead_of_breaking_the_json() -> None:
    prices = swings(BTC.universe, 600)
    bars = candles(prices, "BTC/USD")
    bars.iloc[300] = np.nan
    found = trace.trace(BTC, "BTC/USD", prices, bars)
    assert len(found.bars) == 599
    assert "NaN" not in found.model_dump_json()
