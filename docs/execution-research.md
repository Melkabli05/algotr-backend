# Execution algorithms: research

The third feature list (TWAP, VWAP, implementation shortfall, smart order routing) read against what Alpaca offers and what the app has. Facts come from Alpaca's documentation, from probes on the paper account on 3 November 2026 (Ford limit buys at \$1.00 against a market near \$12.50, all canceled, none filled) and from a measurement on Alpaca's historical bars. Earlier work: [lifecycle-research.md](lifecycle-research.md), [orders.md](orders.md).

## Claims in the list that the evidence does not support

| Claim | What the evidence says |
| --- | --- |
| Alpaca routes each order across IEX, NASDAQ, NYSE and others and the platform can choose between them | Not on this account. The order API has no venue field. The only routing control is `destination` (NYSE, NASDAQ or ARCA) inside `advanced_instructions`, for accounts on the Elite Smart Router; a paper account accepts it and does not run it ([docs](https://docs.alpaca.markets/us/docs/alpaca-elite-smart-router)). Without it, Alpaca's own router decides, and retail orders may go to market makers rather than an exchange ([orders](https://docs.alpaca.markets/us/docs/orders-at-alpaca.md)) |
| Routing can weigh depth at the target price, recent fill rates per venue and fee structures | The data does not exist. A stock quote carries one best bid and one best ask with an exchange code and a size; there is no depth, and the free feed is IEX's book alone ([stream](https://docs.alpaca.markets/us/docs/real-time-stock-pricing-data.md)). A fill or trade update names no venue (`TradeUpdate` has the event, execution id, time, price, quantity, position and the order, and the order has no venue field). Stocks pay no commission, only regulatory fees on sells whatever the venue ([fees](https://alpaca.markets/support/commission-clearing-fees)); coins pay a maker or taker rate by volume tier, 0.15% and 0.25% at the lowest ([fees](https://docs.alpaca.markets/us/docs/crypto-fees)) |
| TWAP slice sizes can follow order book imbalance to improve the average fill price | There is no order book for stocks, only the top-of-book sizes. The research on queue imbalance ([Gould and Bonart](https://arxiv.org/abs/1512.03492), ten Nasdaq stocks) predicts the direction of the next mid-price move, not a move minutes away. And a paper order is filled at the best price whatever its size ("your order quantity is not checked against the NBBO quantities"), so the paper account cannot show a gain either way |
| Implementation shortfall: a risk aversion of 0.5 is aggressive and front-loaded, 2.0 patient and back-loaded | Reversed. In Almgren and Chriss the more risk-averse trader front-loads, a risk-neutral one (λ = 0) trades the straight line (TWAP), and λ → ∞ sells at once ([summary](https://en.wikipedia.org/wiki/Almgren%E2%80%93Chriss_model)). The model cannot back-load without an expected price drift in the trader's favour |
| The algorithm solves a quadratic optimisation at each time step | The trajectory is the closed-form solution of that quadratic, x_k = X·sinh(κ(T − t_k)) / sinh(κT); no solver is needed. Evaluating it again at each step from the quantity and time left gives the same slices while everything fills as planned, and adapts when a slice falls short |
| The 20-day volume profile is stored in PostgreSQL | The stack has no PostgreSQL (SQLite, parquet and JSON files). Reading 20 sessions of 5-minute bars takes about a second, so the profile is read when an order is planned and kept in memory for the day |
| VWAP lands close to the day's VWAP | Better than a flat schedule for large caps, no better for thin or noisy names; see the measurement below |
| Splitting an order lowers its market impact | Paper trading simulates none ("does not account for market impact... price slippage... queue position"). No algorithm here can show a price gain on the paper account; what can be checked is that the schedule is right and that the fills add up |

## Does it already exist

| Option | Finding | Decision |
| --- | --- | --- |
| Alpaca's own TWAP and VWAP | `advanced_instructions` with `algorithm` (`TWAP` or `VWAP`), `start_time`, `end_time` and `max_percentage`, on the Elite Smart Router only (reported minimum deposit \$30,000, from secondary sources); an account outside it is refused with `account not allowed to use advanced_instructions`. Paper accepts it and does not simulate it ([article](https://alpaca.markets/learn/optimize-your-orders-with-vwap-and-twap-on-alpaca)). Probed: TWAP with a window, VWAP, and DMA to NASDAQ were all accepted, the instruction was echoed on the order, and each rested as a plain limit order until canceled | Not usable or testable on this account, so the schedule runs in the app. Revisit if the account goes live on the Elite router: pass `advanced_instructions` and drop the slicing. alpaca-py 0.44.0 has no field for it |
| NautilusTrader | 2.0.0rc7 has `nautilus_trader.trading.ExecutionAlgorithm` (`on_order`, `spawn_market`, `spawn_limit`, a clock for timers) and `LiveNode.add_exec_algorithm`, so a TWAP could run inside a node. It would save only the child-order plumbing, which the iceberg already has, because the schedules themselves (equal, profile, trajectory) are custom either way. It would also need a node alive for the whole horizon, where today's node is a subprocess that reconciles the account and lives 20 to 60 seconds per run ([trading-engine.md](trading-engine.md)), with the schedule inside a process the dashboard can neither show, cancel nor resume after a restart. No end-to-end test of it was made | Not used. Revisit if the autopilot's own orders should go through algorithms: its node is already running then |
| `almgren-chriss` 1.1.0 on PyPI | Three releases in one week of May 2023, one author, GPL, and it requires `numpy<2` where this project pins `numpy>=2.5.2` | Not used. The closed form is a few lines; the package's published example (the paper's numbers) is the test vector, and it matches to 4e-9 shares |
| A library for TWAP or VWAP schedules | None found. Equal slices and a rolling profile are a few lines | Custom, with `pandas` for the profile |

## Probes on the paper account

| Case | Result |
| --- | --- |
| a limit buy with `advanced_instructions` TWAP, VWAP or DMA | accepted, instruction echoed, the order rests as a plain limit; none ran |
| an `ioc` limit buy of a coin far below the market | `pending_new`, then `canceled` with nothing filled in under half a second |
| an `ioc` limit buy of a stock at the open, far below the market | still `pending_new` after 1.5 seconds, so a child order's outcome can take seconds |
| a cancel while the order is `pending_cancel` | 422 `order pending cancel`; the orders became `canceled` after several seconds |
| a limit buy with `exchange`, `destination`, `route` or `venue` set to NASDAQ | accepted and ignored: none is echoed back and the order is a plain limit, so a venue chosen this way would change nothing |
| a coin order under \$10 | 403 `cost basis must be >= minimal amount of order 10`, counted on the limit price for a limit order |

Alpaca's Trading API allows 200 calls a minute per key, separate from market data (staff answer on the [forum](https://forum.alpaca.markets/t/429-rate-limit-exceeded-when-creating-orders/14120) and the [old API page](https://alpaca.markets/deprecated/docs/api-documentation/api-v2)). A desk that reads one order every few seconds stays well inside it.

## The volume profile, measured

Five-minute SIP bars of the regular session, which the free plan allows once they are 15 minutes old: the last 77 full sessions to 3 November 2026. The plan for a day is the mean of the previous 20 sessions' bucket shares, judged on the 57 days after the first 20. Execution is idealised at each bucket's VWAP, so there is no spread or impact. "Misallocated" is half the summed gap between planned and actual bucket shares. The closing auction print falls after 16:00 and is not in these bars.

| | First and last hour of volume | Misallocated, profile against flat | Tracking error vs the day's VWAP, median (90th percentile) in bps, profile against flat |
| --- | --- | --- | --- |
| SPUS | 36%, 15% | 31% vs 40% | 2.8 (12.9) vs 5.8 (15.7) |
| HLAL | 31%, 15% | 53% vs 61% | 3.9 (12.8) vs 3.5 (15.8) |
| AAPL | 27%, 21% | 15% vs 26% | 3.3 (9.7) vs 4.7 (15.4) |
| MSFT | 28%, 21% | 15% vs 27% | 3.8 (10.9) vs 5.8 (20.5) |
| F | 18%, 21% | 18% vs 23% | 4.9 (18.6) vs 4.3 (20.7) |

The profile follows volume far better than a flat schedule for AAPL and MSFT, and the tail of the tracking error shrinks for all five. The typical gain is a few basis points, and for HLAL and F the median is no better than flat.

## The shortfall inputs, measured

The trajectory needs a volatility, a daily volume and a spread. The volatility is the standard deviation of a five-minute price change over the last 20 sessions, scaled to a trading day; the volume is their average over regular hours. For the spread the free plan's IEX quote does not do: it is IEX's book alone, and read at 11:00 New York on 2 November 2026 it was 24 cents for AAPL against a median of 5 cents in the SIP quotes of the same five minutes (MSFT 1.70 against 0.15, SPY 0.08 against 0.02; Ford 0.01 against 0.01). SIP quotes at least 16 minutes old are allowed on the free plan, so the spread is the median over the latest of those. Previews on 3 November 2026, 30 minutes in 6 slices and then two hours, as the share of the shares that trade in the first half of the time:

| | Volatility (\$ a day) | Daily volume | Spread | 30 min, risk 0 | 30 min, risk 1 | 30 min, risk 5 | 2 hours, risk 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | 3.55 | 28.9 M | \$0.04 | 50% | 53% | 63% | 78% |
| MSFT | 6.68 | 15.9 M | \$0.12 | 50% | 52% | 59% | 72% |
| SPY | 3.84 | 32.1 M | \$0.03 | 50% | 55% | 69% | 85% |
| F | 0.18 | 32.4 M | \$0.01 | 50% | 50% | 50% | 51% |
| SPUS | 0.39 | 0.34 M | \$0.01 | 50% | 50% | 50% | 50% |
| HLAL | 0.53 | 0.05 M | \$0.05 | 50% | 50% | 50% | 50% |

The shape follows the theory. A liquid, volatile stock is worked down sooner the longer the horizon and the higher the risk aversion (AAPL to the close at risk 1 sends half of the order in the first slice). Ford barely moves, and SPUS and HLAL trade so little that their impact outweighs any price risk, so all three stay flat. Because the unit of risk aversion has no universal scale, the ticket shows the share in the first half so the setting can be judged by what it does.

## Decisions

**One engine, three schedules.** A parent order is a list of intervals with a weight each. At interval k the slice is what is left times w_k over the weights from k on, so a short fill, a late start or a missed interval spreads the rest over the intervals that remain, and the last one takes everything. TWAP weights are equal, VWAP weights are the profile, and shortfall weights are the steps of the Almgren and Chriss trajectory (the same rule gives the same slices as solving it again at each step).

**Children are `ioc` orders.** A market order, or a limit at the parent's price (a cap on a buy, a floor on a sell). Nothing rests, so a crash or a cancel-all leaves no order behind and a slice that does not fill is carried into the rest. Stocks trade whole shares in regular hours and the parent must end before the close; coins trade fractionally around the clock with at least \$10 a child. Each child has a client order id from the parent's id, so a retry never doubles it.

**Timing and state.** The desk checks every 3 seconds and keeps its state in `data/algos.json`, so a restart or a laptop waking up carries on from where it was. After the end time the parent is `done` or `ended` with the quantity left; nothing is swept late. Every creation and cancel goes to the audit log, cancel-all stops the algorithms before the orders, and a buy passes the cash check for the whole parent.

**Profile and market inputs.** The 20-session profile, the average daily volume and the volatility come from one read of 5-minute bars (SIP for stocks, the crypto bars for coins), kept in memory for the day. The arrival price is the mid of the latest quote.

**Shortfall.** Risk aversion is in units of 10⁻⁶ per dollar of variance, the paper's order of magnitude (its example uses 2 × 10⁻⁶), and a larger number front-loads. Impact follows the paper's example from the spread and the average daily volume: ε is half the spread, η is the spread over one percent of daily volume and γ the spread over ten percent. The volatility is in dollars per root trading day. The spread is read from SIP quotes, not IEX's. It is for stocks only, because the inputs were checked against stock data and not against a coin's volume and quotes. At risk aversion 1 a 30-minute plan is front-loaded only slightly for a liquid stock and a longer horizon or more risk aversion front-loads it firmly; a thin stock stays flat (see the table above).

**Imbalance.** TWAP can tilt the slice by the best bid and ask sizes, up to a chosen percentage and off by default: more when the quote leans towards the side the order is on. It is bounded, the rest is re-spread, and it is labelled unproven because paper fills cannot show a gain.

## Built

1. The engine and TWAP: schedule, desk, API, audit, cancel-all, the ticket and the open-orders table ([orders.md](orders.md)).
2. VWAP, with a 20-session profile.
3. Implementation shortfall, with the closed form and its inputs read from Alpaca.
4. Smart order routing is not built; see the first table.

Not built: routing by venue, Alpaca's native algorithms (Elite accounts only), passive children that earn a maker fee, sending the autopilot's orders through the algorithms, a start time in the future, a participation cap, auction participation, and a cost report beyond the average fill against the arrival price.
