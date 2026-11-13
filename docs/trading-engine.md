# Trading engine

Python 3.13+ (NautilusTrader 2 is a release candidate; pinned to `2.0.0rc7`). There is nothing to start by hand: orders are placed by the daily job described here.

**Automatic order placement.** Autopilot places its orders through Nautilus by default (switch it off on the Research page, Advanced section, to send them straight to Alpaca). Each daily run starts `python -m trading_node.run_plan`, a node that exists only for that run: it reconciles with the account, cancels orders still queued from earlier runs for the symbols in the plan (so a closed market never stacks duplicates), places sells then buys through Nautilus' risk engine, writes each order's outcome to the run record and exits (about 20 to 60 seconds; logs in `logs/plan.log`). Stocks trade whole shares only, and a buy the free cash cannot cover is denied rather than leveraged. Stock orders placed at night all wait for the open, so selling first cannot fund the buys there: what keeps the account from borrowing is that each stock buy is a limit order at most 2% above its decision price, and that the plan counts buys at that limit and stock sells 2% under their price. There is nothing to start or stop by hand, and it needs no all-day data connection.

The Alpaca adapter that Nautilus uses is documented beside its code: [trading_node/alpaca_adapter/README.md](../trading_node/alpaca_adapter/README.md).

**Event-driven backtest.** `python -m trading_node.validate` runs a trend rule (`trading_node/strategies/trend.py`) through Nautilus' `BacktestEngine` on daily bars, filling at the next bar's open; the app runs it as a subprocess and compares it with the vectorised backtest ([systematic.md](systematic.md)).
