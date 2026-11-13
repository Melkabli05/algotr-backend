# Order book

The crypto order book from Alpaca, kept as it arrives, sampled every second and stored. Why only coins, and what this venue's book is like: [orderbook-research.md](orderbook-research.md).

| Endpoint | What it does |
| --- | --- |
| `GET /book/status` | the books followed, whether each is ready, and how long the stream has been quiet |
| `GET /book/view?symbol=&levels=` | the best levels of a coin's book (1 to 25, default 10) with the running total of size, the mid, the spread against its usual, the depth by distance, the imbalance and the order flow |
| `GET /book/impact?symbol=&side=&qty=` | what a market order of that size would find in the book: the fill, its cost against the mid, whether the book holds it, the largest size under the slice budget, and market or limit |
| `GET /book/flow?symbol=&minutes=` | the order flow imbalance in windows of about 5 seconds over the last minutes (up to 1,440) |
| `GET /book/history?symbol=&minutes=` | the mid, the spread in basis points and the imbalance by levels each second over the last minutes (up to 1,440) |

**Source.** The `orderbooks` channel of the app's own crypto stream, for the coins in `STREAM_SYMBOLS`. A second crypto connection is refused (`connection limit exceeded`), so the subscription shares the stream. The location is `us`, Alpaca's own exchange. Stocks have no book at Alpaca: `/book/view` answers 404 for them and the Book tab shows the best quote instead.

**Keeping a book.** A message with `r` true is a whole book and replaces what is held; the others are changes, one level each, and a size of 0 removes the level. A book is ready after its first snapshot, and an update that arrives before one is dropped. Checked against Alpaca's own `latest/orderbooks` taken at the same moment, the 25 best levels of BTC/USD and ETH/USD were identical in price and size. Every second a check runs:

- a book with no snapshot ten seconds after it was wanted is renewed (unsubscribe, subscribe again a second later; the stream answers with a snapshot), at most once in 30 seconds;
- a crossed book (best bid at or above the best ask) is dropped and renewed;
- if no message at all has arrived for 5 seconds the feed is stale and no book is ready, so nothing serves old numbers.

The stream library resubscribes by itself after a reconnect, which also brings a new snapshot, and the autopilot's node pausing the data connection leaves the books not ready until the stream returns.

**Storage.** One SQLite file for each UTC day in `data/book/` (`2026-11-03.sqlite`). Each second a row for each ready coin holds the mid, the spread and the 25 best levels of each side, with exact prices and single-precision sizes, compressed: about 550 bytes of levels a sample and some 105 MB a day for two coins. A second table holds the order flow of each window (coin, end, length, value). Files older than three days are deleted. A second is stored once.

**Imbalance.** (bid volume − ask volume) / (bid volume + ask volume), from +1 (all volume on the bid) to −1 (all on the ask), over the 5, 10 and 25 best levels of each side as the feature list asks, and over the volume within 10, 25, 50 and 100 bps of the mid. The second form compares the same price range on both sides, which the first does not: on this venue the 25th level is 157 bps from the mid on the bid side and 966 bps on the ask side. A reading above +0.3 is labelled buying and one below −0.3 selling; the 0.3 is the list's convention, not a derived threshold. It is computed from the live book each time it is asked for, and a stale feed or a book without a snapshot gives none, so a consumer never gets an old number (the list's one-second expiry). The books of BTC/USD and ETH/USD read close to zero most of the time, because the venue's book is mirrored ladders of the kind a market maker quotes ([orderbook-research.md](orderbook-research.md)). On three hours of both coins no imbalance predicted the price of the next 5 to 60 seconds, the 25-level reading of BTC/USD stayed below −0.3 throughout, and the other labels never fired.

**Order flow.** The Cont, Kukanov and Stoikov definition at the best bid and ask. For each change of the best bid or ask, e = I{P^B_n ≥ P^B_{n−1}} q^B_n − I{P^B_n ≤ P^B_{n−1}} q^B_{n−1} − I{P^A_n ≤ P^A_{n−1}} q^A_n + I{P^A_n ≥ P^A_{n−1}} q^A_{n−1}: a best bid that rises or holds adds its new size, one that falls or holds takes off its old size, and the ask side counts the other way, so a buy that lifts the ask is positive and a sell that hits the bid negative. The values of every book message are summed over a window of about 5 seconds (it closes at the first once-a-second sample within half a second of 5 seconds, and its length is stored), in coins. A snapshot is not flow, and a change below the top of the book is none. The view carries the last window and the last minute (12 windows), and every window is stored in the day's file. It is not weighted by price, because the definition is not. Against the change of price in its own window it correlates at 0.27 to 0.67, depending on the coin and the price used; against the next 5 to 60 seconds it showed nothing ([orderbook-research.md](orderbook-research.md)). The trade stream's taker side would measure aggression directly, but alpaca-py's trade model drops it, so it is not used.

**Spread and depth.** The view gives the spread in dollars and in basis points of the mid, and the size and the dollars resting within 10, 25, 50 and 100 bps of the mid on each side. A spread is wide only against the coin's own habit (the coins the scanner follows ranged from 3 to 67 bps on one snapshot), so the view also holds the usual spread, the median of the stored seconds of the last hour, and today's as a multiple of it. It needs a minute of stored seconds to say anything and is read from the day's file at most once a minute.

`/book/impact` walks the side an order would take, best level first: the quantity filled, the average price, the cost in basis points against the mid, the levels used and whether the book held the whole order. It also gives `advice`, `market` or `limit`, with the `reason` when it is `limit`:

- the spread is more than `MAX_SPREAD_MULTIPLE` (3) times the usual one;
- the walk costs more than `MAX_SLIPPAGE_BPS` (100) over the mid, or the book holds less than the order.

`max_qty` is the largest size whose average fill stays within `SLICE_SLIPPAGE_BPS` (20) of the best price on that side. It is measured from the best price and not from the mid, because the half spread is paid whatever the size and a coin whose half spread alone is over the budget would otherwise get no slice at all. Each of the three is off at 0. None has a source; they are starting points to tune, and a thin top of the book makes them bite: on one snapshot \$50,000 of DOT would have paid about 59% over the mid ([orderbook-research.md](orderbook-research.md)).

**Guards.** A market order on a coin that `advice` would send as a limit order is refused with 422 and the reason, and the refusal is in the audit log. This applies to `POST /orders` for a coin with type `market`, buy or sell. Limit and stop-limit orders, stocks and the parent of an algorithm order are not judged, and a coin whose book is not ready (no snapshot yet, a stale feed, not followed) is let through, because there is nothing to judge it by. A market slice of a coin in a TWAP or VWAP order is cut to `max_qty` before it is sent; if the spread is then too wide for it, it is held back, the order's note says so, and the amount carries to the next interval as it does for the \$10 minimum. The plan from `POST /orders/algo/preview` names `book_cap` while the book is as it was. A paper account fills from the quote and ignores depth, so none of this shows a saving there.

**Dashboard.** The Book tab of the blotter on the Terminal shows the selected coin's mid, spread (in basis points and in dollars) with the usual spread and the multiple, when it was last updated, the imbalance by levels and by distance (green past +0.3, red past −0.3), the 10 best levels of both sides with running totals, the size and dollars resting within each distance of the mid, a chart of the imbalance over the last 30 minutes, the order flow of the last window and the last minute (green when positive) and a bar chart of the windows. For a stock it says there is no book and shows the best bid and ask. The confirm dialog of a market order on a coin adds the expected fill from `/book/impact` and has no Confirm while the book advises a limit order.

## Where the code is

`src/app/features/book/`

| Module | What it holds |
| --- | --- |
| `book.py` | `Book` (the size at each price) and `BookKeeper` (the books, written by the stream's thread) |
| `store.py` | the daily SQLite files |
| `metrics.py` | the imbalance by levels and by distance, its labels, the order flow of one change, the depth within a distance and the walk of an order through the book |
| `service.py` | `BookDesk`: the views, the cost of an order, the guards, the once-a-second sampler and the snapshot watchdog |
| `router.py`, `schemas.py` | the HTTP routes and models |

The market feature's `MarketStreams` subscribes the crypto stream to the books and renews one on request; `main.py` connects it to the desk, because features do not import each other. The order entry reaches the desk through `BookGuard` in `core/book.py` (`refusal` and `max_size`), which `BookDesk` satisfies: `orders/risk.py` asks it about a market order and `orders/algo.py` about each market slice of a coin.
