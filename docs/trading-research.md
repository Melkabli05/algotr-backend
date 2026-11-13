# Order types and position accounting: research

What the app does today, what Alpaca and the libraries offer for each requested item, and what was decided. Alpaca's rules were read from its documentation and then probed on the paper account on 3 November 2026 (Ford, one SPUS exit order and BTC/USD; every order canceled at once, open orders 0 before and after). Plan and build order: [Plan](#plan).

## Today

`POST /orders` takes market and limit orders with `day`, `gtc` and `ioc`. `GET /positions` returns Alpaca's quantity, average entry price, market value and unrealized P&L. There is no cost basis, no lots, no realized P&L and no check against the broker. The autopilot's Nautilus path supports market and limit only ([adapter README](../trading_node/alpaca_adapter/README.md)), so every new type goes through `alpaca-py` directly.

## Order types

| Item | Alpaca | `alpaca-py` 0.44 | Decision |
| --- | --- | --- | --- |
| Market, limit | native | `MarketOrderRequest`, `LimitOrderRequest` | exist |
| Stop-loss | `stop`; elects on the consolidated last print, then becomes a market order. A buy stop must sit above the price (`stop price must be greater than current price`) | `StopOrderRequest` | add |
| Stop-limit | `stop_limit`; becomes a limit order at the limit price. A limit below the stop was accepted | `StopLimitOrderRequest` | add |
| Trailing stop | `trailing_stop` with `trail_percent` 0.1 to 25, or `trail_price` between 0.1% and 25% of the price. Regular hours only. Single orders: not a bracket leg | `TrailingStopOrderRequest` | add |
| Bracket | `bracket`: entry, take-profit limit and stop-loss stop (an optional stop-loss limit makes it a stop-limit). Legs sit `held` until the entry fills in full; one leg filling cancels the other | `order_class` with `TakeProfitRequest`, `StopLossRequest` | add; buy side only |
| OTO | `oto`: entry plus one exit | same | add; buy side only |
| OCO | `oco`: exit-only pair for a position already held, same side, type `limit` | same | add; sell side only |
| Iceberg | none. Alpaca has no display quantity or reserve order. Nautilus's `display_qty` works only where the venue adapter supports it, and ours does not | none | slice client-side |
| Time in force | below | `TimeInForce` | add `fok`, check per type |

Alpaca refuses with these messages, which the API already passes on as a 422:

| Case | Message |
| --- | --- |
| `ioc` while the market is closed | `ioc orders are only accepted during market hours` |
| `fok` on stop or stop-limit | `stop orders only support DAY or GTC time_in_force` |
| `ioc` or `fok` on a trailing stop | `trailing stop orders support only "day" or "gtc" time_in_force` |
| `ioc` or `fok` on a bracket | `bracket orders support only "day" or "gtc" time_in_force` |
| fractional quantity with `gtc` | `fractional orders must be DAY orders` |
| fractional trailing stop | `fractional orders must be market, limit, stop, or stop_limit orders` |
| fractional bracket | `fractional orders must have order_class=simple` |
| crypto `day` or `fok` | `invalid crypto time_in_force` |
| crypto stop or trailing stop | `invalid order type for crypto order` |
| crypto bracket | `crypto orders not allowed for advanced order_class: otoco` |
| crypto below \$10 | `cost basis must be >= minimal amount of order 10` |
| price below a cent | `sub-penny increment does not fulfill minimum pricing criteria` |

## Time in force

| | market | limit | stop | stop-limit | trailing stop | bracket, OTO, OCO |
| --- | --- | --- | --- | --- | --- | --- |
| Whole shares | day, gtc, ioc, fok | day, gtc, ioc, fok | day, gtc | day, gtc | day, gtc | day, gtc |
| Fractional shares | day | day | day | day | none | none |
| Crypto | gtc, ioc | gtc, ioc | none | gtc, ioc | none | none |

Sources: [Alpaca orders](https://docs.alpaca.markets/docs/orders-at-alpaca), [crypto orders](https://docs.alpaca.markets/docs/crypto-orders), [fractional trading](https://docs.alpaca.markets/docs/fractional-trading), and the probes above. Alpaca's table marks `ioc` and `fok` on market and limit orders with an asterisk ("contact sales"); a limit `fok` was accepted on the paper account and a limit `ioc` was refused only because the market was closed. Market-order `ioc` and `fok` were not probed and need a market-hours check. `gtc` orders are canceled by Alpaca after 90 days. Extended hours take limit orders only and are not offered here.

## Long only

The paper account is a margin account (multiplier 4, shorting enabled) and accepts a sell with nothing held: the order comes back with `position_intent: sell_to_open`. The ticket therefore could open a short today. `position_intent` is an `alpaca-py` field, and Alpaca checks it: a `sell_to_close` with no position is refused (`position intent mismatch, inferred: sell_to_open, specified: sell_to_close`), and one above the held quantity is refused too (`insufficient qty available for order`). Sending `sell_to_close` on every stock sell and `buy_to_open` on every stock buy lets the broker enforce long only, with no position lookup of our own. Crypto cannot be shorted at Alpaca. Bracket and OTO therefore take a buy entry only, and OCO is a sell exit only.

Stop-loss and take-profit orders are treated as permissible in the one fatwa found ([islamqa.org](https://islamqa.org/hanafi/askimam/126108/what-is-the-ruling-of-investing-on-the-stock-market-trading-with-forex-investing-in-crypto-currency-and-the-concept-of-stop-loss-and-take-profit/)); no AAOIFI ruling turned up. Margin and short selling are the items scholars broadly rule out ([HBKU study](https://elmi.hbku.edu.qa/en/studentTheses/shariah-appraisal-of-short-selling-and-margin-trading/)). Manual buys are limited to cash (see [orders.md](orders.md)); the account's margin multiplier is not used.

## Iceberg

Orders are public at Alpaca, so nothing can be hidden. Slicing is the only way to show less: one visible limit order of the display size at a time, the next placed when it fills, the rest held by the app. Nautilus execution algorithms do this kind of work, but the Nautilus node here is short-lived by design and Alpaca's free plan allows one data connection, so a node kept running for an iceberg would collide with the API's own stream ([adapter README](../trading_node/alpaca_adapter/README.md)). A small executor in the API process, polled by APScheduler and persisted in `data/icebergs.json`, is the custom part. If the app is down, the held remainder is not working; the one visible child still is.

## Positions

**Data.** `GET /v2/positions` carries `cost_basis`, `unrealized_plpc`, `unrealized_intraday_pl`, `change_today` and `qty_available`; the app keeps only six fields. Alpaca has no lot or realized-P&L endpoint. Fills come from `GET /v2/account/activities`, paged by `page_token` at up to 100 rows ([docs](https://docs.alpaca.markets/us/docs/account-activities)). `TradingClient` has no method for it, so the call is `TradingClient.get`; `alpaca-py`'s `TradeActivity` model requires an `account_id` the Trading API does not send, so rows are parsed with a small model of our own.

**What this account's history shows.** 1,378 rows: 896 `FILL` (581 orders, 139 of them filled in pieces), 478 `CFEE`, 3 `FEE`, 1 `JNLC`. Replaying the fills gives the broker's quantity for AAPL (10), HLAL (349) and SPUS (427), and the broker's average entry price and cost basis to the cent (HLAL 71.864699, SPUS 58.64726, AAPL 340.342). The FIFO and average-cost replays agree there because the remaining lots coincide, so Alpaca's own convention (undocumented) cannot be told apart from this data; the check compares against the average-cost view. BTC does not match from fills alone: bought 0.427833, sold 0.426763345, short by 0.001069646 against the broker's 0.000000009. The difference is the `CFEE` rows: a coin fee with negative quantity on each buy (154 rows for 154 buys) and a USD fee on each sell (324 rows, -\$84.77). Counting them gives exactly 9E-9. The `FEE` rows are daily REG and CAT fees (-\$3.62, -\$0.01) and stay outside lots.

**Libraries for lots.**

| Candidate | Fit |
| --- | --- |
| `coinbasis` 0.1.0 | Pure Python, typed, MIT or Apache, FIFO, LIFO, HIFO, average and specific-ID. One release, June 2026, one author; built for crypto tax (wallets, gifts, brackets). `holdings()` returns one aggregated row per asset and raises for specific-ID, so open lots cannot be listed or picked. No split handling |
| `rp2` 1.7.2 | Crypto tax calculator fed by spreadsheets, six dependencies |
| `irs-asset-fifo-calculator` 1.0.0 | GPL, FIFO only |
| `beancount` 3.2.3 | GPL-2.0, ledger-file model |
| Nautilus `Position` | Average price only; no lots ([positions](https://nautilustrader.io/docs/latest/concepts/positions/)) |

None lists open lots with an id, takes a lot choice per sale, and handles fractional stock and coin fees. A ledger of about 150 lines does, and it is checked against the broker's own quantities on every sync. The library to revisit is `coinbasis` once it has releases and open lots.

**Cost basis methods.** FIFO, average cost and specific lot, computed from the same executions. Alpaca takes no lot instruction on an order, so specific lot is internal accounting: the dashboard records which lots a sale closes, and any sale without a record closes FIFO. A lot is one buy order (its fills at the volume-weighted price); a sale is one sell order.

**Reconciliation.** The broker is the source of truth for quantity. Each sync pulls new activities, rebuilds the lots and compares per symbol: quantity must match within 1e-9, the average-cost price within 0.01%. Differences are breaks and are listed, not hidden. Activities that change a holding without a fill (`SPLIT`, `SPIN`, `MA`, `NC`, `SC`, `REORG`) are reported as the likely cause and are not applied; none have occurred on this account. A break is closed by adopting the broker's figures, which writes an adjustment lot and an audit entry. Practice from reconciliation guides (vendor and community sources, not a standard): exact match for units, corporate actions as their own category, timing differences recorded rather than alerted ([overview](https://www.solvexia.com/glossary/investment-reconciliation)).

## Plan

1. Orders: the types, time in force per asset class (served as `GET /orders/rules` so the ticket and the API share one table), bracket, OTO and OCO, the sell-to-close rule, ticket and open-orders table.
2. Iceberg executor.
3. Ledger: activities store, lots, three methods, realized and unrealized per lot, reconciliation job, Lots and Realized tabs, the extra position columns.
4. Specific-lot choice in the sell ticket.

Each ends with tests, docs and a commit. Not built: order replace, extended hours, tax-year reports, split adjustment, fee allocation to lots.

## As built

Orders: [orders.md](orders.md). Ledger: [ledger.md](ledger.md). Where the build differed from the plan:

- The ledger models shorts. The account sold SPY short on its first day, before orders were restricted, and a long-only ledger left a phantom 49-share lot and a permanent break.
- Fills are synced by `date`, not by the last id: Alpaca lists a crypto fee a day after its trade, under an id that sorts before the day's fills.
- A coin buy's missing fee is a "pending" difference, not a break, for up to three days.
- The average-cost check allows 0.5% (the coin fee is in the ledger's cost and not in Alpaca's entry price), and a cost difference is a note, not a break.
- Adopting Alpaca's quantity prices the added lot so the pooled cost equals Alpaca's total.
- The Realized tab became the closed-lots view of the Lots tab.
- The specific-lot choice is made at the sell ticket and saved against the order after it is placed, so it applies to plain sells only.
