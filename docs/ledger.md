# Ledger

Lots, cost basis and realized P&L rebuilt from Alpaca's fills, and a check that they match the broker's positions. Alpaca has no lot or realized-P&L endpoint, and no library fits; the evidence is in [trading-research.md](trading-research.md#positions).

| Endpoint | What it does |
| --- | --- |
| `GET /ledger/status` | activities stored, last sync, the last error and the last reconciliation |
| `POST /ledger/sync` | fetch new activity, rebuild, compare with the positions |
| `GET /ledger/reconciliation` | the last comparison |
| `POST /ledger/adopt/{symbol}` | make the ledger hold what Alpaca holds in that symbol |
| `GET /ledger/positions?method=` | open positions with cost basis, unrealized and realized P&L |
| `GET /ledger/lots?method=&symbol=` | open lots, each at Alpaca's current price |
| `GET /ledger/realized?method=&symbol=&days=` | closed lots, newest first, with totals |
| `GET /ledger/fills?symbol=&days=&limit=` | the stored fills, newest first: time, quantity, price, whether partial, and the coin fee of a crypto buy |
| `GET /ledger/picks/{order}`, `PUT`, `DELETE` | the lots a sell order closes (specific lot) |

`method` is `fifo` (the default), `average` or `specific`.

**Source.** `GET /v2/account/activities`. The first sync pages the whole history, 100 rows a page. Later syncs ask for each day from three days back (or the day before the last stored one, if that is older) to tomorrow, by `date`, because Alpaca lists a crypto fee a day after its trade and its ids sort before the day's fills, so paging on from the last id would miss it. Rows are stored by id, so a repeat adds nothing. A job syncs every 5 minutes, five seconds after start, and an offline attempt is retried on the next tick; a fill starts one sooner, below. `data/ledger.sqlite` holds the rows as Alpaca sent them, the lots chosen for sales, and the adjustment lots.

**On a fill.** The trade-update stream carries each `fill` and `partial_fill` with its `execution_id`, price, quantity and the position size. `main.py` gives the stream hub a listener that calls `Ledger.nudge()` for those two events and no others. That schedules one sync three seconds out and moves it back with each further fill, so a burst (a bracket's legs, an iceberg's slices, an order filled in parts) costs one sync, and a burst that keeps going syncs after thirty seconds at most. The three seconds are a debounce, not a measured delay: a fill that Alpaca has not listed in its activities yet is found by the next sync. The timer stays as the backstop, because the stream has no documented replay and a fill missed while the API was down is found only by a sync. The stream message has no fee, so the coin fee comes in with the sync, a day late.

**Lots.** A lot is one buy order, its fills at the volume-weighted price; a sale is one sell order. The coin fee Alpaca takes from a crypto buy is deducted from the lot, which keeps the money paid, so the cost per coin includes the fee. The USD fee on a crypto sale and the daily regulatory fees are left out of the P&L. A sell with no lot to close opens a short lot, and a later buy covers it first; this account has that history from before orders were restricted to long only. Quantity is signed wherever lots are shown, so a short is negative.

**Methods.** The same executions are replayed three ways. FIFO closes the oldest lot first. Average cost pools a symbol into one lot at the average cost and keeps it when part is sold. Specific lot closes the lots chosen for that sale and the oldest for the rest; with no choice recorded it equals FIFO. The choice is made in the sell ticket ("Lots to sell") and saved with `PUT /ledger/picks/{order}` once the order is placed. Alpaca takes no lot instruction with an order, so the choice is internal bookkeeping and does not change what the broker reports.

**Realized and unrealized.** Realized P&L is computed for each closed part of a lot: what was received less what was paid, for a short the other way round. Unrealized P&L is each open lot at the current price Alpaca reports for the position. Totals by symbol are in `/ledger/positions`.

**Reconciliation.** Alpaca is the source of truth for quantity. After each sync the average-cost view is compared with every position:

| Result | Meaning |
| --- | --- |
| quantity | the ledger and Alpaca hold different amounts (beyond 1e-9). A split, spin-off, merger or name change on record for the symbol is named as the likely cause and is not applied |
| pending | the ledger holds slightly more of a coin than Alpaca. Within half a percent of the last three days' coin buys this is the fee that Alpaca has not posted yet, and it clears by itself |
| cost | the average cost differs by more than 0.5%. This is a note only, because Alpaca's method is not documented and its average entry excludes the coin fee; positions under \$1 are skipped |

`ok` means no quantity difference. `POST /ledger/adopt/{symbol}` closes one: if Alpaca holds more it adds a lot of the missing quantity, priced so that the pooled cost equals Alpaca's total; if it holds less it writes the difference off at cost with no gain. Either way an entry goes to the audit log.

**On this account** (3 November 2026): 1,378 activities become 581 orders. AAPL (10 at 340.342), HLAL (349 at 71.864699) and SPUS (427 at 58.64726) match Alpaca's quantity and average cost, BTC matches at 0.000000009, and SPY, which was sold short on the first day and covered, leaves no open lot. Realized P&L is -\$18.37 before fees, the same by FIFO and by average cost, of which -\$23.04 came from longs and \$4.67 from covered shorts.

**Dashboard.** The positions table adds the cost, P&L percent and the day's P&L, with their totals above it, and under it one line on the check ("Lots match Alpaca", or the positions that differ, with a button to use Alpaca's figure) and "Check now". The Lots tab shows open lots, closed lots or the fills for one symbol or all, the first two by method. A fill is listed with its price and quantity, whether it was partial, and the coin fee of a crypto buy. The view loads the newest 500 fills, and the line above the table says "newest 500 only" when there are more.

## Where the code is

`src/app/features/ledger/`

| Module | What it holds |
| --- | --- |
| `activities.py` | reading the rows, grouping fills into executions, attaching coin fees, paging |
| `lots.py` | the replay: lots, closes, the three methods, shorts |
| `reconcile.py` | the comparison with the positions |
| `store.py` | the SQLite store |
| `service.py` | `Ledger`: sync, the views, the fills, adopting, the jobs |
| `router.py`, `schemas.py` | the HTTP routes and models |

Not covered: splits and other corporate actions are reported, not applied; fees are not allocated to lots; there is no tax-year report.

## Wash sales

The lots, holdings and closed lots apply the wash-sale rule (IRS Publication 550) to everything but coins, which the rule does not cover. A sale at a loss is matched, share for share and each replacement share once, with shares of the same stock bought in the 30 days before it and still held, then with shares bought in the 30 days after it. The matched part of the loss is disallowed (`disallowed` on the closed lot, `reported` is the gain less it) and added to the basis of the shares bought back (`wash` on the open lot). The economic `gain` is unchanged beside it, and the reconciliation with Alpaca uses unadjusted cost, since Alpaca's average cost does not apply the rule. Not modelled: the holding period carried over to the replacement shares, buys in other accounts (an IRA counts for the IRS), and options or other substantially identical securities. Treat the figures as a guide for the tax return, not the return.

## Purification and zakat

`GET /ledger/purification?year=` (the Terminal's **Purify** tab) reads the stored activity for the year. A dividend (`DIV`, `DIVTXEX`, the amount received before tax withheld) owes its holding's non-compliant income share; a return of capital or a tax line is not income. The share is entered per symbol with `PUT /ledger/purification/ratios` (`{"SPUS": 0.012}` for 1.2%; `null` removes one) or in the tab, from the ETF issuer's yearly purification figure or a screening service; a holding without one is listed as missing, never guessed. Interest received (`INT` above zero) is owed in full. Interest paid (`INT` below zero) cannot be purified and is shown as a warning so its cause can be stopped. The zakat line is an estimate by the trading-asset method, 2.5% of the account's value today, due once a lunar year has passed on it; scholars differ on the method for long-term holdings, so treat it as a starting point. The ratios are in `data/ledger.sqlite` (`purification`).
