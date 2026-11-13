# Compliance and pre-trade checks

Every buy placed by hand or through the API (`POST /orders`, `/orders/iceberg`, `/orders/algo`) passes these checks in order before it reaches Alpaca; a sell is never stopped by them, so a position can always be closed. A check that cannot read what it needs refuses the buy, so nothing passes for want of data. Each refusal is a 422 with the reason and is written to the audit log like every order attempt. The autopilot's own orders are governed by its allocation, its books and the kill switch below, and by Nautilus' risk engine.

| Check | Refuses a buy when | Setting |
| --- | --- | --- |
| Shariah screen | the stock is not held by a Shariah-screened fund whose holdings list is no more than 7 days old (SPUS, and HLAL when its file is supplied); the screened funds SPUS, HLAL, UMMA and SPSK pass themselves; a coin passes only when it is in `CRYPTO_COINS` (scholarly opinion on crypto is divided) | the lists in `data/scanner/lists/` ([scanner.md](scanner.md)) |
| Kill switch | the autopilot's books are at `defensive` or `flatten` | `max_drawdown`, `reduce_at` |
| Daily loss | the account is down `DAILY_LOSS_LIMIT` (3%) or more against yesterday's close | `DAILY_LOSS_LIMIT` |
| Position size | the position after the buy would be more than `MAX_POSITION_SHARE` (25%) of account value | `MAX_POSITION_SHARE` |
| Sector (stocks) | the stock's GICS sector (S&P 500 constituents, refreshed monthly into `data/risk/sectors.csv`) would be more than `MAX_SECTOR_SHARE` (40%) of account value; funds and coins are not capped | `MAX_SECTOR_SHARE` |
| Liquidity (stocks) | the buy is more than `MAX_ADV_SHARE` (1%) of the stock's average dollar volume over 20 sessions, from consolidated (SIP) bars: IEX's own volume is a few percent of the market's | `MAX_ADV_SHARE` |
| Spread (stock market orders) | the quoted spread is wider than `MAX_STOCK_SPREAD_BPS` (50 bps); a limit order sets its own price and is not checked | `MAX_STOCK_SPREAD_BPS` |
| Order book (coin market orders) | the spread is more than 3 times usual, the walk through the book costs more than 100 bps, or the best ten levels lean more than `MAX_BOOK_IMBALANCE` (60%) against the order while the last minute's order flow imbalance (Cont, Kukanov and Stoikov) points the same way | `MAX_SPREAD_MULTIPLE`, `MAX_SLIPPAGE_BPS`, `MAX_BOOK_IMBALANCE` ([book.md](book.md)) |
| Cash | the buy costs more than the cash left after open buy orders | none: the account never borrows |

Stocks have no order book on Alpaca (one best bid and ask only), so the depth and flow checks are for coins.

**Monitoring what is held.** `GET /compliance/holdings` screens every position against today's lists, and the status strip warns (`Shariah: Not on the lists now: …`) when a held name has left them, so it can be sold. `GET /compliance/screen/{symbol}` gives the verdict and its authority for one symbol. The lists are the authority: the app does not compute screening ratios itself.

**Kill switch.** Five steps, set by the autopilot's book drawdown against `max_drawdown` ([systematic.md](systematic.md)): `normal`; `watch` from a quarter of the limit (shown, nothing changes); `reduce` from `reduce_at` of it (each strategy at half size); `defensive` from three quarters (the autopilot only sells and no book grows, and order entry refuses every buy); `flatten` at the limit or while halted (everything sold, trading stopped until Resume). The step is in `GET /systematic/automation` (`kill_state`) and in the Safeguards badges.

**Pattern day trading.** Not enforced: FINRA's amendments to Rule 4210, approved by the SEC on 14 April 2026 and effective 4 June 2026, removed the pattern day trader rule and its $25,000 minimum in favour of an intraday margin standard, and Alpaca deprecated its PDT fields. The account trades without borrowing, so the margin standard does not apply to it.

**Wash sales, purification and zakat** are in [ledger.md](ledger.md#wash-sales).
