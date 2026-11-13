import pytest
from pydantic import TypeAdapter, ValidationError

from app.core.types import Symbol

adapter = TypeAdapter(Symbol)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("aapl", "AAPL"),
        (" btc/usd ", "BTC/USD"),
        ("BTCUSD", "BTCUSD"),
        ("brk.b", "BRK.B"),
    ],
)
def test_valid_symbols_are_stripped_and_uppercased(raw: str, expected: str) -> None:
    assert adapter.validate_python(raw) == expected


@pytest.mark.parametrize(
    "raw", ["", "..", "../orders", "AAPL/../x", "A B", "A" * 11, "BTC/USD/X", "a/b", "/USD"]
)
def test_invalid_symbols_are_rejected(raw: str) -> None:
    with pytest.raises(ValidationError):
        adapter.validate_python(raw)
