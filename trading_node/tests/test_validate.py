import numpy as np
import pandas as pd

from trading_node import validate


def prices(symbols: dict[str, float], days: int = 400, drift: float = 0.0006) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=days)
    rng = np.random.default_rng(3)
    rows = []
    for symbol, base in symbols.items():
        closes = base * np.exp(np.cumsum(rng.normal(drift, 0.01, days)))
        opens = np.r_[closes[0], closes[:-1]] * (1 + rng.normal(0, 0.002, days))
        rows += [
            {"date": day.strftime("%Y-%m-%d"), "symbol": symbol, "open": o, "close": c}
            for day, o, c in zip(dates, opens, closes, strict=True)
        ]
    return pd.DataFrame(rows)


def test_nothing_trades_until_the_average_exists_then_the_rule_holds_and_pays_fees() -> None:
    found = validate.run(prices({"SPUS": 40.0, "HLAL": 50.0}), 200, 5)
    equity = [value for _, value in found["equity"]]
    assert len(equity) == 400
    assert all(value == validate.CASH for value in equity[:199])
    assert found["orders"] == found["fills"] > 0
    assert found["commissions"] > 0


def test_a_coin_trades_in_fractions() -> None:
    found = validate.run(prices({"BTC/USD": 20_000.0}, days=300), 200, 25)
    assert found["fills"] > 0
    assert found["equity"][-1][1] != validate.CASH


def test_a_commission_cell_is_read_in_either_shape() -> None:
    assert validate.paid(["49.99 USD"]) == 49.99
    assert validate.paid("[1.50 USD]") == 1.5
