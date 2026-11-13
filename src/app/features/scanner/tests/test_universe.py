import pandas as pd

from app.features.scanner import bars, universe
from app.features.scanner.tests.support import (
    asset,
    bar,
    last_business_day,
    sessions,
    trading_with,
    yesterday,
)

END = last_business_day(yesterday())


def recent(rows: dict[str, tuple[float, float, int]]) -> pd.DataFrame:
    """Per symbol: last close, daily volume and the number of sessions traded."""
    return bars.to_frame(
        {
            symbol: [bar(day, price, volume) for day in sessions(END, count)]
            for symbol, (price, volume, count) in rows.items()
        }
    )


def test_only_tradable_stocks_on_the_listed_exchanges_with_plain_tickers_remain() -> None:
    found = [
        asset("AAPL"),
        asset("IBM", exchange="NYSE"),
        asset("SPY", exchange="ARCA"),
        asset("BRK.B", exchange="NYSE"),
        asset("OTCX", exchange="OTC"),
        asset("HALT", tradable=False),
        asset("892ESC0A3"),
        asset("ABC.WS", exchange="NYSE"),
        asset("TOOLONGX"),
    ]
    assert universe.tradable(found) == {"AAPL", "IBM", "SPY", "BRK.B"}


def test_the_assets_are_asked_for_active_us_equities_only() -> None:
    trading = trading_with([asset("AAPL")])
    assert universe.assets(trading) == [asset("AAPL")]
    request = trading.get_all_assets.call_args.args[0]
    assert (request.asset_class.value, request.status.value) == ("us_equity", "active")


def test_liquid_needs_price_dollar_volume_and_enough_sessions() -> None:
    frame = recent(
        {
            "GOOD": (50.0, 1_000_000, 30),
            "CHEAP": (9.99, 10_000_000, 30),
            "THIN": (50.0, 100_000, 30),
            "NEW": (50.0, 1_000_000, 14),
            "EDGE": (50.0, 400_000, 15),
        }
    )
    assert universe.liquid(frame) == {"GOOD", "EDGE"}


def test_liquidity_is_measured_on_the_last_twenty_sessions_only() -> None:
    quiet = [bar(day, 50.0, 1_000.0) for day in sessions(END, 40)[:20]]
    busy = [bar(day, 50.0, 1_000_000.0) for day in sessions(END, 20)]
    frame = bars.to_frame({"TURNED": quiet + busy})
    assert universe.liquid(frame) == {"TURNED"}
