# algotr

Shariah-compliant algorithmic trading platform on Alpaca: long only, cash, no leverage, no derivatives, and automation trades paper accounts only.

> **Split layout.** This repo is the backend only. The React terminal lives in a separate repository, **algotr-frontend**. The backend expects it checked out as `./frontend/` — `uv run poe start` builds it (`pnpm --dir frontend build`) and the API serves it at `/ui`, and the Docker build copies it the same way. To run the full platform locally:
>
> ```bash
> git clone https://github.com/Melkabli05/algotr-frontend.git frontend
> cp .env.example .env   # set ALPACA_API_KEY and ALPACA_API_SECRET (paper keys)
> uv sync
> uv run poe start       # builds the terminal, then runs the API; open http://localhost:8000/ui/
> ```
>
> `./frontend/` is gitignored here, so a local checkout is never committed to this repo.

- **Autopilot.** Daily-bar trend strategies with fixed rules, each judged by five statistical checks. Only what passes them runs, and it is rebalanced once a day ([docs/systematic.md](docs/systematic.md)).
- **Scanner.** A nightly scan of the liquid US market that ranks what is trending, shows which Shariah-listed stocks the rules would hold and keeps a log of those picks ([docs/scanner.md](docs/scanner.md)).
- **Terminal.** A React trading terminal served by the API at `/ui`: markets, positions, orders, the coin order book, strategies, signals with their explanations, backtests, research, risk, compliance, the audit log and settings ([docs/terminal.md](docs/terminal.md)).
- **Trading engine.** [NautilusTrader](https://nautilustrader.io/) places the orders through a pure-Python Alpaca adapter ([docs/trading-engine.md](docs/trading-engine.md)).
- **Compliance and risk.** Shariah screening before every buy and on what is held, position, daily-loss, liquidity, spread and order-flow limits, and a five-step kill switch ([docs/compliance.md](docs/compliance.md)).
- **Purification.** Dividends and interest per year with what each owes to charity, and a zakat estimate ([docs/ledger.md](docs/ledger.md#purification-and-zakat)).
- **FinRL.** Reinforcement-learning agents run in an isolated environment and are judged by the same gates ([docs/systematic.md](docs/systematic.md)).

## Quick start

```bash
cp .env.example .env   # set ALPACA_API_KEY and ALPACA_API_SECRET (paper keys)
uv sync
pnpm --dir frontend install
uv run poe start       # builds the terminal, then runs the API; open http://localhost:8000/ui/
uv run poe check       # lint, formatting, types, dependencies and tests
```

`poe start` ([Poe the Poet](https://poethepoet.natn.io), tasks in `pyproject.toml`) builds the terminal with Vite, then starts the API, which serves it. `STREAM_SYMBOLS` in `.env` picks what the watchlist streams (stocks stream only while the market is open; crypto is 24/7). `DATA_FEED` picks the stock feed: `iex` (the free plan) or `sip` (needs Algo Trader Plus; see [architecture.md](docs/architecture.md#market-data)). FinRL needs its own environment: `uv run poe finrl-setup`.

## Layout

| Path | What it holds |
| --- | --- |
| `src/app/core/`, `src/app/quant/` | shared code: settings, Alpaca clients, the audit log; price history, performance metrics, significance tests |
| `src/app/features/` | `portfolio`, `orders`, `market`, `systematic` and `scanner`, each with its router, logic, schemas and tests |
| `trading_node/` | the NautilusTrader trading node: the Alpaca adapter, strategies and the plan job |
| `frontend/` | the React trading terminal, with its tests |
| `finrl_sidecar/` | FinRL, installed and run apart from the project's dependencies |
| `tests/` | app-level tests; shared fixtures are in `conftest.py` |
| `docs/` | documentation |

## Documentation

| Document | What it covers |
| --- | --- |
| [docs/description.md](docs/description.md) | what the platform is and does, for people meeting it for the first time |
| [docs/architecture.md](docs/architecture.md) | layout, dependency rules, background jobs, data on disk, checks |
| [docs/systematic.md](docs/systematic.md) | strategies, the five checks, the rule lab, FinRL, the audit log, autopilot |
| [docs/scanner.md](docs/scanner.md) | the scanner as built; [plan](docs/scanner-plan.md) and [research](docs/scanner-research.md) behind it |
| [docs/terminal.md](docs/terminal.md) | the React trading terminal at `/ui`: stack, shell, views, commands |
| [docs/trading-engine.md](docs/trading-engine.md) | how orders are placed through Nautilus |
| [docs/quant-research.md](docs/quant-research.md) | the research behind the strategies |
