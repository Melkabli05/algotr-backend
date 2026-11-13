from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from app.core.tables import shadow_orders
from trading_node.arbiter import StrategyTarget, net_orders
from trading_node.state_writer import StateWriter

Book = Callable[[str], Any]


def walk(levels: list[tuple[float, float]], qty: float) -> float | None:
    """The average price of taking `qty` from the book's levels, best first; None if too thin."""
    left, cost = qty, 0.0
    for price, size in levels:
        taken = min(left, size)
        cost += taken * price
        left -= taken
        if left <= 0:
            return cost / qty
    return None


class ShadowExecutionClient:
    """Takes a shadow strategy's intended orders, fills them on paper and records them.

    Nothing reaches the venue: each order is filled against the live order book where there is
    one, else at the quote's far side, else at the last price, and kept with the strategy's
    version so its record can be compared with what the live strategies were filled at.
    """

    def __init__(self, writer: StateWriter, book: Book, quote: Book) -> None:
        self.writer = writer
        self.book = book
        self.quote = quote
        self.held: dict[str, dict[str, float]] = {}

    def fill(self, symbol: str, side: str, qty: float, last: float) -> tuple[float, str]:
        found = self.book(symbol)
        if found is not None:
            levels = found.asks() if side == "buy" else found.bids()
            pairs = [(float(level.price), float(level.size())) for level in levels]
            price = walk(pairs, qty)
            if price is not None:
                return price, "book"
        tick = self.quote(symbol)
        if tick is not None:
            far = tick.ask_price if side == "buy" else tick.bid_price
            return float(far), "quote"
        return last, "last"

    def submit(
        self, target: StrategyTarget, equity: float, prices: Mapping[str, float]
    ) -> list[dict[str, Any]]:
        """Bring the strategy's paper book to its target; returns the simulated fills."""
        held = self.held.setdefault(target.strategy, {})
        notional = target.allocation * equity
        fills = []
        for order in net_orders(target.weights, notional, held, prices):
            symbol, side, qty = str(order["symbol"]), str(order["side"]), float(order["qty"])  # type: ignore[arg-type]
            price, source = self.fill(symbol, side, qty, prices[symbol])
            held[symbol] = held.get(symbol, 0.0) + (qty if side == "buy" else -qty)
            fills.append(
                {
                    "time": datetime.now(UTC),
                    "strategy": target.strategy,
                    "version_id": target.version_id,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "source": source,
                }
            )
        if fills:
            rows = list(fills)

            def work(connection: Connection) -> None:
                connection.execute(sa.insert(shadow_orders), rows)

            self.writer.put(work)
        return fills
