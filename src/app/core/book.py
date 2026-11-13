from typing import Protocol


class BookGuard(Protocol):
    """What order entry may ask of the order book: a reason to refuse, and a size limit."""

    def refusal(self, symbol: str, side: str, qty: float) -> str | None: ...

    def max_size(self, symbol: str, side: str) -> float | None: ...
