# Order book intelligence: research

The fourth feature list (Level 2 ingestion, order book imbalance, order flow imbalance, spread and depth) read against what Alpaca offers and what the app has. Facts come from Alpaca's documentation, from probes on 3 November 2026 (read-only snapshots, and two short captures of the crypto stream with the app stopped) and from the market microstructure papers the list leans on. Earlier work: [execution-research.md](execution-research.md), [lifecycle-research.md](lifecycle-research.md).

## Claims in the list that the evidence does not support

| Claim | What the evidence says |
| --- | --- |
| Alpaca offers Level 2 market-by-price data | For crypto only. The crypto stream has an `orderbooks` channel and there is a `latest/orderbooks` endpoint ([stream](https://docs.alpaca.markets/us/docs/real-time-crypto-pricing-data.md), [latest](https://docs.alpaca.markets/us/reference/cryptolatestorderbooks-1.md)). Stocks have none: alpaca-py 0.44.0 has no book method for them, the stock stream carries trades, quotes, bars, price bands, trading status and imbalances, and a stock quote is one best bid and one best ask, which on the free plan is IEX's own book ([stream](https://docs.alpaca.markets/us/docs/real-time-stock-pricing-data.md)) |
| The top N levels can be captured at sub-second intervals | The stream sends one level per message, 1 to 3.5 messages a second for BTC/USD, ETH/USD and SOL/USD, so a snapshot every second already holds everything there is. The book arrives whole once and the rest is changes |
| Level counts (5, 10, 25) are comparable depths | Not on this venue. The book is sparse: BTC/USD had 78 bid levels from \$48.8k to \$77.5k. The 5th level was 10 bps from the mid and the 25th was 157 bps away on the bid side and 966 bps on the ask side, so "25 levels" covers a different price range on each side |
| Order book imbalance above 0.3 signals buying pressure | The 0.3 is a convention with no source. Here the book is mirrored ladders of the kind a market maker quotes (ETH/USD asks 0.0333, 0.0665, 14.8, 29.6, 44.4 against bids 0.0335, 0.0669, 15.0, 30.1, 45.1), so imbalance sits near zero, and what it does predict is measured below |
| Order flow imbalance is a stronger short-term predictor than order book imbalance | The paper it comes from ([Cont, Kukanov and Stoikov](https://arxiv.org/abs/1011.6402)) explains the price change in the same interval (a linear relation, average R² 65% over 10-second intervals, slope inversely proportional to depth) and "does not test whether OFI predicts future price changes". The multi-level version ([Xu, Gould and Howison](https://arxiv.org/abs/1907.06230)) is also contemporaneous. Prediction is measured below |
| Order flow imbalance is weighted by price | It is not in the definition, which uses the sizes at the best bid and ask only (below) |
| The book is stored in PostgreSQL, partitioned daily, and streamed through Redis | Neither is in the stack (SQLite, parquet and JSON files). Daily SQLite files give the partitions, the stream hub passes events inside the process, and a cache with a one-second life gives the one-second freshness |
| Wide spreads can be rejected | Against what? On one snapshot the spreads of the ten coins the scanner follows ranged from 3 bps (BTC) to 67 bps (BCH), so no single threshold fits; a spread has to be compared with the coin's own usual one |
| Depth-aware sizing improves fills | Paper orders are simulated from the real-time quotes, and "your order quantity is not checked against the NBBO quantities" ([paper trading](https://docs.alpaca.markets/us/docs/paper-trading.md)); Alpaca staff have said on the forum that liquidity is not considered. It can protect a live order; it cannot show a gain here |

## Probes

| Case | Result |
| --- | --- |
| the latest book of ten coins | the whole book, 11 to 91 levels a side, no zero sizes, 0.14 to 1.9 seconds a call |
| the stream, 150 and 100 seconds | BTC/USD 3.3 and 3.4 messages a second, ETH/USD 2.0 and 1.9, SOL/USD 1.0. Each update message holds one level and about half are removals (size 0). Every message has the keys `S`, `T`, `t`, `b`, `a` and `r` |
| the first message after subscribing | a full snapshot with `r` true (78 bids and 88 asks for BTC/USD) at 0.8 seconds. In the first capture BTC and ETH sent none, and in the second unsubscribing and subscribing again returned a fresh snapshot within 3 seconds, so a missing snapshot has to be noticed and the subscription renewed |
| a book kept from a snapshot and the updates after it | the same 27 bids and 47 asks as the REST book a second or two later, with the best prices a tick apart |
| the depth and the cost of an order, against Alpaca's own book | the mid, the dollars within 10, 25, 50 and 100 bps and the average fill of buys and sells of three sizes, for BTC/USD and ETH/USD, were identical to a separate walk of `latest/orderbooks` taken between two reads of the app |
| market orders larger than the book, or costing over 100 bps, posted to the API | refused with 422 and the book's reason, each recorded as refused in the audit log, nothing reached Alpaca |
| a second crypto connection while the app's own runs | refused, `connection limit exceeded`; the app's stream is unaffected. The book subscription has to share the app's crypto stream, and an experiment needs the app stopped |

## The book on this venue

Crypto orders execute on Alpaca's own exchange, and the book is that venue's. One snapshot on 3 November 2026: the bid side's depth within a distance of the mid in dollars, and the cost of a market buy against the ask side, in basis points from the mid.

| | Spread (bps) | Depth within 10, 25, 50, 100 bps | Market buy of \$1,000, \$10,000, \$50,000 |
| --- | --- | --- | --- |
| BTC/USD | 3.2 | 257k, 697k, 1.07M, 1.07M | 5.7, 6.5, 6.6 |
| ETH/USD | 2.9 | 102k, 205k, 496k, 496k | 5.1, 5.8, 6.6 |
| SOL/USD | 6.3 | 267, 1.5k, 4.1k, 7.9k | 12, 50, 378 |
| XRP/USD | 39 | 0, 5.3k, 16k, 42k | 20, 28, 87 |
| LINK/USD | 17 | 12.8k, 77k, 102k, 154k | 8.7, 8.7, 14.5 |
| AVAX/USD | 60 | 0, 0, 2.0k, 5.8k | 30, 80, 519 |
| DOT/USD | 18 | 615, 3.7k, 4.9k, 7.4k | 11, 70, 5,947 |
| LTC/USD | 59 | 0, 0, 3.7k, 10.9k | 30, 47, 273 |
| BCH/USD | 67 | 0, 0, 8.9k, 26k | 33, 36, 82 |
| UNI/USD | 25 | 0, 31k, 41k, 62k | 13, 15, 35 |

The top of the book is thin even for BTC/USD (0.001 BTC at the touch, 1.1 BTC at the fourth level), which is why a \$1,000 buy costs more than half the spread, and for the thin coins a large market order is dangerous: \$50,000 of DOT would pay about 59% over the mid.

## What the imbalance and the order flow predict

Measured on what the app stored on 3 November 2026 from 17:10 to 20:07 UTC: 893 windows of about 5 seconds for each of BTC/USD and ETH/USD (1,786 in all; the stack was restarted several times while it was built, so more than half of that span is missing). The order flow of each window and the book at its end are set against the change of price after it. A coin's quoted mid is a poor reference here, because the touch holds dust (0.001 BTC) and the mid moved in every one of the windows. The second price is the mid of the average fill of a \$1,000 market buy and a \$1,000 market sell. The slopes are standardised, so each is a correlation; the two coins are pooled after standardising each; the intervals are 95%, with errors that allow for the overlap of the windows (Newey-West, one lag for each window the horizon spans).

**The same window.** The order flow of a window against the change of price in the same window, the relation the paper reports:

| Price | BTC/USD | ETH/USD | Pooled | R² pooled |
| --- | --- | --- | --- | --- |
| mid | 0.27 [0.18, 0.36] | 0.45 [0.38, 0.52] | 0.36 [0.30, 0.42] | 0.13 |
| $1,000 impact mid | 0.46 [0.40, 0.53] | 0.67 [0.59, 0.76] | 0.57 [0.51, 0.62] | 0.32 |

So the order flow does move with the price in the window, which is what the paper shows, and more with the depth-based price than with the mid. It is well short of the paper's average R² of 65% (large US stocks, 10-second intervals).

**The next 5 to 60 seconds.** What was measured at the end of a window against the change of price after it, with the \$1,000 impact mid (pooled r, and its interval; \* marks an interval that excludes 0):

| | next 5 s | next 10 s | next 30 s | next 60 s |
| --- | --- | --- | --- | --- |
| order flow | +0.02 [-0.01, +0.06] | +0.02 [-0.02, +0.06] | +0.04 [-0.01, +0.09] | +0.01 [-0.04, +0.07] |
| imbalance, 5 levels | -0.00 [-0.05, +0.04] | -0.02 [-0.08, +0.04] | -0.02 [-0.12, +0.08] | -0.03 [-0.19, +0.12] |
| imbalance, 10 levels | +0.05 [-0.02, +0.11] | +0.03 [-0.04, +0.09] | +0.00 [-0.08, +0.08] | +0.01 [-0.10, +0.11] |
| imbalance, 25 levels | -0.04 [-0.10, +0.02] | -0.05 [-0.12, +0.02] | -0.08 [-0.18, +0.02] | -0.11 [-0.25, +0.02] |
| imbalance within 25 bps | -0.05 [-0.10, -0.01]\* | -0.07 [-0.13, -0.01]\* | -0.06 [-0.15, +0.03] | -0.05 [-0.17, +0.07] |
| imbalance within 100 bps | -0.10 [-0.16, -0.04]\* | -0.13 [-0.19, -0.07]\* | -0.17 [-0.27, -0.07]\* | -0.22 [-0.35, -0.09]\* |

Nothing here supports the claim that the order flow is a stronger short-term predictor than the imbalance: it predicts nothing at any horizon, and neither do the imbalances by level. Of the 48 cells (six variables, four horizons, the mid and the impact mid), 10 exclude 0 where 2 or 3 would by chance, and all 10 are the imbalances within a distance of the mid, with a negative sign: more volume on the bid, a lower price next. That is the opposite of the claim, and it is not established. The imbalance within a distance moves slowly (the correlation of a window with the one a minute earlier is 0.46 to 0.73 for the 25 and 100 bps imbalances of both coins), so three hours hold far fewer independent observations than 1,786 windows, and the intervals above are too narrow. With a five-minute window for the errors only the 100 bps imbalance at 60 seconds still excludes 0 (-0.22 [-0.37, -0.07]); in five-minute blocks (13 for each coin) no variable shows a clear effect in both coins. Several days of stored seconds could settle it, and the store keeps three.

**The 0.3 labels.** The list's threshold for buying and selling pressure, on the same windows:

| | BTC/USD | ETH/USD |
| --- | --- | --- |
| 5 levels | 32 buying, 488 selling, 373 balanced | 0 buying, 39 selling, 854 balanced |
| 10 levels | always balanced | always balanced |
| 25 levels | selling in all 893 windows | always balanced |
| within 25 or 100 bps | always balanced | always balanced |

A label that is on all the time says nothing, and one that never fires does not either: the 25-level imbalance of BTC/USD is below -0.3 for three hours because the ask side's ladder reaches further with larger sizes, which is the unequal range of the level counts. Where a label did fire often enough to compare (5 levels, BTC/USD) the mean change after "selling" was +0.12 bps in 10 seconds and +0.63 in 60, against -0.02 and -0.06 when balanced, which is the wrong way round for a selling signal and small against a spread of 3 bps; the labels change slowly, so those windows are not independent either.

## Decisions

**Crypto only.** The book is kept for the coins in `STREAM_SYMBOLS`, on the app's own crypto stream. For a stock only the best quote exists, so the Book tab says so and shows the quoted spread and sizes without calling them depth.

**Keeping the book.** Each symbol is a dictionary of price to size on each side, replaced by every `r` message and changed by the rest (a size of 0 removes the level). A book is ready only after a snapshot, and a crossed book, or no snapshot ten seconds after subscribing, renews the subscription.

**Storage.** One SQLite file for each UTC day, with a row for each symbol each second holding the metrics and the 25 best levels of each side packed and compressed; files older than a retention (three days) are deleted. The latest metrics of each symbol live in a cache with a one-second life, so a consumer that asks for a number gets nothing, not a stale one, when the stream stops.

**Imbalance.** (bid volume − ask volume) / (bid volume + ask volume) over the 5, 10 and 25 best levels as listed, and over the volume within 10, 25, 50 and 100 bps of the mid, which compares like with like. The 0.3 labels are the list's convention and are shown as such.

**Order flow.** The Cont, Kukanov and Stoikov definition at the best bid and ask, e_n = I{P^B_n ≥ P^B_{n−1}} q^B_n − I{P^B_n ≤ P^B_{n−1}} q^B_{n−1} − I{P^A_n ≤ P^A_{n−1}} q^A_n + I{P^A_n ≥ P^A_{n−1}} q^A_{n−1}, summed over each 5-second window. The trade stream also names the taker side of every trade, so the signed taker volume over the same window is measured directly and kept beside it.

**Spread and depth.** The spread in dollars and basis points, the depth by price band, a walk of the book for the cost of an order of a given size and the largest size that stays under a cost. The "wide spread" test is against the coin's own median spread of the last hour. The slice budget is measured from the best price on the side and not from the mid: the half spread is paid whatever the size, and AVAX, LTC and BCH on the snapshot above (60, 59 and 67 bps wide) would never get a slice under a 20 bps budget measured from the mid. Orders use them only for coins: a market order is refused when the spread is several times its usual or the walk costs more than a limit in basis points, and a slice of a coin's TWAP or VWAP is cut to what the book can take under a budget, the rest carrying on.

## Plan

1. The book: stream, upkeep, storage, the API and a first Book tab.
2. Order book imbalance.
3. Order flow imbalance, with the measurement of what either predicts.
4. Spread and depth: the cost of an order, the ticket's estimate, the refusals and the slice size.

Not built: a book for stocks (Alpaca has none), PostgreSQL and Redis, and an order book strategy, because none was asked for.
