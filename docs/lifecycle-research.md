# Order lifecycle and broker integration: research

The second feature list (order types, bracket orders, position tracking, the order lifecycle, the broker and its data) read against what the app already does and what Alpaca does. Facts come from Alpaca's documentation and from probes on the paper account on 3 November 2026: Ford orders far from the market, a one-share SPUS exit, all replaced or canceled, and no fill. Earlier work: [trading-research.md](trading-research.md), [orders.md](orders.md), [ledger.md](ledger.md).

## Claims in the list that the evidence does not support

| Claim | What Alpaca says or does |
| --- | --- |
| Alpaca fills orders against the IEX book on the free tier and the consolidated feed on the paid tier | The data plan decides which prices you can read, not how an order fills. Paper orders are matched against the best available price (NBBO) from real-time quotes and are not routed to an exchange ([paper trading](https://docs.alpaca.markets/us/docs/paper-trading.md)); Alpaca staff say fills depend on the latest quote whatever the plan (forum, informal) |
| A market order executes immediately | Only while the market is open. Outside hours it queues for the open, and `ioc` is refused (`ioc orders are only accepted during market hours`) |
| A stop-limit protects against a gap-down fill | It protects the price, not the position: after a gap past the limit it "stays active until executable or canceled", so the position can stay open. A plain stop gets out, at whatever price |
| When one bracket child fills, the other is canceled | Normally. Alpaca adds that "both may fill in fast markets" |
| Limit orders have full time-in-force options | See the matrix in [orders.md](orders.md): fractional shares day only, crypto gtc and ioc, `ioc` never outside market hours |
| Cancellation of a filled order returns a clear status | Documented as a 422 (`The order status is not cancelable`). The paper account answers 2xx and changes nothing, so the status has to be read back |
| Tick-level data needs the paid plan | Historical trades and quotes older than 15 minutes are available on the free plan, on both feeds (5 rows read from each). What the paid plan adds is recent data: the last minutes of SIP trades and the latest SIP quote are refused with `subscription does not permit querying recent SIP data` |

## Item by item

| Item | Alpaca | In the app | Decision |
| --- | --- | --- | --- |
| Market, limit, stop-loss, stop-limit | native | built ([orders.md](orders.md)) | nothing to add |
| Bracket (OCO) | `order_class=bracket`; legs `held` until the entry fills in full | built, with OTO and OCO | nothing to add |
| Position tracking: quantity, cost, value, realized and unrealized P&L | positions endpoint has quantity, average entry, cost basis, value, unrealized P&L; no realized P&L | positions table and the lot ledger ([ledger.md](ledger.md)) | nothing to add |
| Reconcile on every fill and on a timer | the trade-update stream has no documented delivery guarantee or replay | the ledger reconciles every 5 minutes | add the fill trigger |
| Order submission, logged for audit | `POST /v2/orders` returns the order with status `pending_new`, `new` or `accepted`; legs come back `held` | validated, then submitted; only the autopilot's orders reach the audit log | log every manual attempt, accepted or refused; add the cash check |
| Order modification | `PATCH /v2/orders/{id}`; see below | none | build |
| Order cancellation: one, all, idempotent | `DELETE /v2/orders/{id}`, `DELETE /v2/orders`; see below | one order, no bulk, errors pass through | build |
| Fill tracking in real time with price, quantity, fees | `trade_updates` stream; see below | events kept in memory for the Activity tab; the ledger stores fills and fees from the activities endpoint | trigger the ledger on a fill; list the stored fills |
| Paper trading mode | `paper-api.alpaca.markets`; fills at the NBBO, a random partial fill 10% of the time, no slippage or queue position, no dividends | `ALPACA_PAPER`, shown as a badge; the autopilot refuses live accounts | nothing to add |
| Market data feed | below | the stream and the chart are fixed to IEX | make the feed a setting |

## Replace and cancel, as probed

| Case | Result |
| --- | --- |
| replace a limit order's price | a new order id (`replaces` points back), status `new`; the old order becomes `replaced` and carries `replaced_by` |
| replace quantity and time in force together | accepted; the other fields keep their values |
| replace again with the old id | 422 `order already replaced` |
| a field the type does not have | 422 `limit orders must not have stop_price` |
| stop price; both prices of a stop-limit; a trailing stop's `trail` | accepted (`trail` is the new percent or dollar value, whichever the order has) |
| a bracket's parent price, and each leg by its own id | accepted; the parent's legs then show the new prices |
| an OTO parent | accepted on paper although the documentation lists OTO as not replaceable |
| quantity of a fractional order | 422 `cannot replace qty in fractional order`; its price can change |
| a filled order, an unknown id | 404 `order not found` |
| a canceled order | 422 `order is not open` |
| an already replaced order | 422 `order already replaced` |
| cancel a filled, canceled or replaced order | success, nothing changes |
| cancel an unknown id | 404 `order not found for <id>` |
| cancel all open orders | a list of `{id, status: 200}`, one per order; the stream shows a `canceled` event for each |

The documentation adds that "a success return code from a replaced order does NOT guarantee the existing open order has been replaced": if the original fills first, the replacement is rejected and `order_replace_rejected` arrives on the stream. An order cannot be replaced while it is `accepted`, `pending_new`, `pending_cancel` or `pending_replace`, notional orders cannot be replaced, and while a replace is pending buying power is held for the larger of the two orders ([replace](https://docs.alpaca.markets/us/reference/patchorderbyorderid-1.md), [cancel one](https://docs.alpaca.markets/us/reference/deleteorderbyorderid-1.md), [cancel all](https://docs.alpaca.markets/us/reference/deleteallorders-1.md)).

## Trade updates

The stream is `wss://paper-api.alpaca.markets/stream` (live without `paper-`), listening to `trade_updates`; the paper endpoint sends binary frames. Events: `new`, `fill`, `partial_fill`, `canceled`, `expired`, `done_for_day`, `replaced`, and the rarer `accepted`, `rejected`, `pending_new`, `stopped`, `pending_cancel`, `pending_replace`, `calculated`, `suspended`, `order_replace_rejected`, `order_cancel_rejected`. A fill message carries `execution_id`, `timestamp`, `price`, `qty` (this fill), `position_qty` and the order. It carries no fee: crypto fees arrive as activity rows a day later, and the daily regulatory fees are not per fill ([stream](https://docs.alpaca.markets/us/docs/websocket-streaming.md)). Observed: a new order gives `pending_new` then `new`; a replace gives `replaced` for the old order only, whose `replaced_by` names the new one; held bracket legs give `held`; canceling all gives one `canceled` per order within a tenth of a second.

## Market data

| | Basic (free) | Algo Trader Plus (\$99 a month) |
| --- | --- | --- |
| Real-time equities | IEX only | all US exchanges (SIP) |
| WebSocket symbols | 30 | unlimited |
| Historical data | all of it except the latest 15 minutes | no restriction |
| Historical calls | 200 a minute | 10,000 a minute |

Feeds: `iex`, `sip`, `delayed_sip` (15 minutes late), `boats`, `overnight`; a feed outside the subscription fails at authentication ([plans](https://docs.alpaca.markets/us/docs/about-market-data-api.md), [stream](https://docs.alpaca.markets/us/docs/real-time-stock-pricing-data.md)). The app reads the stream, the chart and the price of a market buy from the feed in `DATA_FEED` (IEX unless set to SIP) and old daily bars from SIP. The `delayed_sip` feed could serve a free account's SIP reads but is not offered, because the stream library supports only IEX and SIP and a late price is a poor basis for an order. On this free-plan account SIP bars with an end of now, the SIP latest trade and the SIP latest quote are refused with `subscription does not permit querying recent SIP data`, and SIP bars with no end time stop 15 minutes ago.

## Decisions

Replace and cancel follow Alpaca's own behavior and add only what it leaves to the client: a clear message for each refusal, a status read back after a cancel, bulk cancel that also stops iceberg orders first (or the next slice would be placed after "cancel all"), and an iceberg that follows a replaced slice to its new id. Every manual submission, change and cancellation is written to the hash-chained audit log, including the ones the app refuses before sending. One risk check is added because the platform's rule is cash and no leverage: a buy may not cost more than the cash left after open buy orders. Other limits (largest order, position size, daily loss) need numbers that were not given.

## Built

1. Order modification and cancellation (single, all), with Modify and Cancel all in the open-orders tab ([orders.md](orders.md)).
2. The audit log of submissions, changes and cancellations, and the cash check ([orders.md](orders.md)).
3. The ledger syncs when a fill arrives; the stored fills are listed ([ledger.md](ledger.md)).
4. The data feed as a setting, `DATA_FEED`, named in the header ([architecture.md](architecture.md#market-data)).

Not built: a kill switch that also closes positions and halts the autopilot, extended hours, and modifying iceberg orders as a whole.
