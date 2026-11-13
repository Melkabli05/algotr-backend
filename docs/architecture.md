# Architecture

```
src/app/         backend (FastAPI)
  core/          settings, Alpaca clients, errors, the audit log
  quant/         price history, performance metrics and significance tests
  features/      one folder per feature
    portfolio/   account, positions, equity history
    orders/      place (market, limit, stop, stop-limit, trailing, bracket, OTO, OCO, iceberg, TWAP, VWAP, implementation shortfall), list and cancel orders
    ledger/      lots, cost basis and realized and unrealized P&L from the fills, checked against the broker
    market/      historical bars, the market clock, the live WebSocket stream
    book/        the order book of each followed coin, kept from the stream, sampled and stored
    drift/       the drift monitor: ADWIN and distance checks over the returns of every strategy and agent
    systematic/  daily trend strategies: research, planning, the autopilot job
    scanner/     nightly market scan, ranking, evaluation, a shadow log, news, SEC filings and a crypto table
trading_node/    NautilusTrader trading node: the Alpaca adapter, strategies and the plan job
                 (the React terminal lives in the algotr-frontend repo now,
                  served at /ui by its own nginx image)
finrl_sidecar/   FinRL, FinBERT and SHAP in their own environment
tests/           app-level tests: the launcher and the application lifecycle
docs/            this documentation
```

## Backend

`main.py` creates the app, installs the error handlers, includes each feature's router and starts the background services in its lifespan: the market streams and the `Automation` and `Scanner` schedulers.

A feature folder has the same parts: `router.py` (HTTP), `service.py` or a few named modules (logic), `schemas.py` (models) and `tests/`, which sit beside the code. Tests build services on temporary folders and fake Alpaca clients, so none touches the network, the real audit log or the real Nautilus job (guards in `conftest.py`).

**Dependency direction.** Features depend on `core/` and `quant/`, never on each other's internals; `systematic` calls the `orders` and `portfolio` services because it plans and places orders, and takes the scanner's picks through the `SleeveProvider` protocol in `core/sleeve.py`, which `main.py` connects. `main.py` also connects the stream to the ledger (the hub passes every trade update to a listener, and a fill makes the ledger sync) and to the drift monitor (it reads the return streams of the strategies and the agents through `ReturnSource`, and the autopilot reads its size cuts through `SizingGuard`) and to the order book (the crypto stream hands each book message to the book desk, which can ask the stream to renew a book; order entry asks the desk about market orders on coins through the `BookGuard` protocol in `core/book.py`). Order entry also asks the scanner for the Shariah verdict and the autopilot for the kill switch's step through `ShariahScreen` and `KillSwitch` in `core/compliance.py`; `compliance` reads the same screen for what is held. `quant/` imports nothing else from the app. `trading_node/` and `frontend/` do not import features: the terminal talks HTTP, and Nautilus takes only `app.core.settings`.

| Package | What it holds |
| --- | --- |
| `core/` | `settings.py` (environment), `broker.py` (Alpaca clients; market-data clients get a timeout and retries), `errors.py` (broker errors as HTTP responses), `audit.py` (the hash-chained log), `sleeve.py` (what the scanner offers the autopilot: `Sleeve` and the `SleeveProvider` protocol), `drift.py` (what the drift monitor reads and gives: the `ReturnSource` and `SizingGuard` protocols), `sidecar.py` (runs a task in the FinRL sidecar's own environment: the data goes in as files, the result comes back), `book.py` (what order entry asks of the order book: the `BookGuard` protocol), `types.py` |
| `quant/` | `history.py` (adjusted daily bars with a shared cache), `performance.py` (strategy returns, metrics and the five checks), `validation.py` (Sharpe significance, Deflated Sharpe, probability of backtest overfitting), `volume.py` (the average share of a day's volume in each five-minute bucket over the last 20 sessions, with their daily volume and volatility, and a stock's quoted spread) |

## Market data

`DATA_FEED` in `.env` is `iex` (the default) or `sip`. It sets the feed of the stock stream, of the chart's bars, of the last trade that prices a market buy in the cash check, and of the Nautilus data client; `GET /market/feed` reports it and the header names it ("Stream · IEX"). Crypto has one feed. The feed decides which prices are read here; Alpaca matches paper orders against the best quote it holds ([lifecycle-research.md](lifecycle-research.md)).

| | Basic (free) | Algo Trader Plus (\$99 a month) |
| --- | --- | --- |
| `DATA_FEED` | `iex` | `sip` or `iex` |
| Streamed stock symbols | 30 | unlimited |
| Recent data | IEX only | all exchanges |
| Older data (15 minutes and more) | both feeds | both feeds |

`sip` on the free plan fails in three ways, all measured on this account on 3 November 2026. Alpaca ends the stream with `insufficient subscription` (the header says "Stream off" and the log says why). A chart request with no end time returns bars that stop 15 minutes ago, so the last candle is late. The last-trade lookup is refused with `subscription does not permit querying recent SIP data`, which is logged, and the price check on a market buy is then skipped. The scanner's and the strategies' daily history is read from SIP whatever `DATA_FEED` says, ending 16 minutes ago, which the free plan allows. Bars come at one minute at the finest; the stream carries every quote and trade, and old quotes and trades can be read from the free plan too, but the app does not store or use them.

## Background work

| Job | When | State on disk |
| --- | --- | --- |
| Scanner refresh | 00:01 UTC, and at startup and every 10 minutes after until it has succeeded that day | `data/scanner/` |
| News poll | every 15 minutes, and once at startup | `data/scanner/news.sqlite`, `filings.sqlite` |
| Sentiment scoring | after each news poll; sends the sidecar only headlines without a record, and only when the FinBERT weights are installed | `data/scanner/news.sqlite` |
| Pre-open review | checked every 5 minutes, runs once in the 45 minutes before the open | `data/scanner/watch.json` |
| Crypto check | every hour, and once at startup; fetches only when yesterday's bar is missing | `data/scanner/crypto.parquet` |
| Autopilot rebalance | 00:05 UTC, and at startup and every 10 minutes after until a run has not failed that day | `data/automation.json` |
| Iceberg slices | every 3 seconds; calls Alpaca only while an iceberg is working | `data/icebergs.json` |
| Algorithm slices | every 3 seconds; calls Alpaca only while an algorithm order is working | `data/algos.json` |
| Ledger sync | every 5 minutes, first run 5 seconds after start; an offline attempt is retried on the next tick; also 3 seconds after a fill on the trade-update stream (30 seconds at most while fills keep coming) | `data/ledger.sqlite` |
| Book samples | every second; also asks the stream to renew a book that has no snapshot | `data/book/` |
| Drift check | 00:01 UTC, and whoever asks first on a day it was missed (the autopilot's plan does) | `data/drift.json` |
| AI agents | every 30 minutes, first check 5 seconds after start; works only when a model is due, a day is new or a decision is waiting to be explained, and only with `AI_AGENTS=true` | `data/ai/`, `data/audit.jsonl` |
| Market streams | while the API runs | none |

The jobs use APScheduler with a catch-up run at startup. Every decision of the autopilot, the scanner and the review, and every manual order action, is appended to `data/audit.jsonl`, one hash-chained log shared by the features (`GET /systematic/audit` shows it and whether the chain is intact).

## Health

`GET /status` (`app/health.py`, wired in `main.py` because no feature may import another) reads one check from each of the autopilot (a failed last run or a halt is an error), the audit log (a broken chain), the ledger (a failed sync or a difference from Alpaca), the order book (a quiet stream), the scanner (no scan, a failed or day-and-a-half-old refresh, a Shariah list without a file, a news or sentiment error), the drift monitor (a stream sized down) and the agents (the sidecar missing, decisions waiting to be explained). The Terminal shows the problems as badges under the account strip, or one green badge when there are none. `GET /health` stays the bare liveness check the launcher waits for.

## Data on disk

`data/` and `logs/` are runtime state and are not committed.

| Path | What it holds |
| --- | --- |
| `data/audit.jsonl` | the audit log: autopilot runs, lab and FinRL runs, scans and evaluations, manual order actions, the agents' training and the SHAP explanation of each of their decisions ([xai.md](xai.md)) |
| `data/automation.json` | autopilot settings, the last 30 runs, the drawdown peak and the symbols it manages |
| `data/ledger.sqlite` | the account's activity as Alpaca sent it, the lots chosen for sales, the adjustment lots |
| `data/book/` | the order book of each followed coin once a second, one SQLite file a day, three days kept ([book.md](book.md)) |
| `data/drift.json` | when the drift monitor started and the events it recorded |
| `data/ai/` | the trained agents and their scaling, and each agent's record and holdings ([ai.md](ai.md)) |
| `data/icebergs.json` | iceberg orders: the slices placed, their fills, and the last 50 finished |
| `data/algos.json` | algorithm orders: the schedule, the slices placed, their fills, and the last 50 finished |
| `data/scanner/bars.parquet` | eight years of adjusted daily bars for the liquid and the listed stocks |
| `data/scanner/state.json`, `names.json` | the last refresh and the ticker names |
| `data/scanner/news.sqlite`, `filings.sqlite` | headlines with their typed events and sentiment records ([sentiment.md](sentiment.md)); 8-K items per company (only with `SEC_CONTACT`) |
| `data/models/finbert/` | the pinned FinBERT weights, from `uv run poe finbert-setup` |
| `data/scanner/watch.json` | when news was last polled, filings last refreshed, and the last pre-open review |
| `data/scanner/crypto.parquet` | four years of daily bars for the coins in `CRYPTO_COINS` |
| `data/scanner/lists/` | dated Shariah list copies; drop `HLAL.csv` here |
| `logs/plan.log` | output of the Nautilus plan job |

## Running checks

`uv run poe check` runs ruff (lint and format), mypy in strict mode, deptry (declared against imported dependencies) and the tests. Each is also a task of its own: `lint`, `format-check`, `types`, `deps`, `test`.

## Adding a feature

Create `src/app/features/<name>/` with a router, a service, schemas and tests, and include the router in `main.py`. Take shared logic from `core/` and `quant/`; if a second feature needs something the first one owns, move it there instead of importing across features.
