# AI-Native Trading Platform

## Application Description

---

### What It Is

The **AI-Native Trading Platform** is a single-broker algorithmic trading system for Shariah-compliant investing. It combines systematic rules, reinforcement learning and financial language models with careful execution and risk management. It is built for quantitative traders and small teams who want to research, validate and run strategies on US equities and spot crypto through Alpaca, without assembling a dozen disconnected tools and without trusting a black-box model with capital.

It is not a signal service and not a charting tool. It is the whole pipeline: **research → validation → paper trading → execution → risk control → explainability → compliance**, running as one system. Automation trades paper accounts only; orders placed by hand go to whichever Alpaca account the keys belong to.

---

### What It Does

**It researches strategies with discipline.** Strategies are fixed, published rules (trend, dual momentum, Donchian breakouts) and a cross-sectional momentum scanner over the Shariah-listed stocks. Every rule family is screened across all of its variants in a fast vectorised backtest, and the trend rules are then checked again in NautilusTrader's event-driven backtest, with whole shares, real fees and next-open fills, so the two engines can be compared. Backtests fill trades where live orders fill: stocks at the next open, coins at the close. Each strategy must pass five gates before it may trade: a Sharpe lead over buy-and-hold that a bootstrap test finds unlikely to be luck, a smaller drawdown, a profit in the most recent 30%, a profit at 50 bps per trade, and a Deflated Sharpe Ratio of at least 95% counted over every rule and variant ever tried. For the ETF trend rule, whose funds have only five years of history, the two significance tests run on the US stock market since 1926. Look-ahead is tested by cutting the history at several dates. Survivorship bias cannot be removed from Alpaca's data (delisted stocks are missing, the Shariah lists are today's), so every result that carries it says so. The reinforcement learning agents are evaluated by purged, embargoed walk-forward retraining.

**It generates signals with a hybrid AI layer.** Reinforcement learning agents (PPO, SAC, A2C) see technical indicators and the market's daily headline sentiment and risk, scored by FinBERT, and choose weights over two Shariah ETFs and cash. A UCB1 bandit blends them and adapts as their records change. The ensemble trades only once it passes the same gates as the rules, admitted after ten passing days in a row. A drift monitor (River's ADWIN, confirmed by a Wasserstein distance and a Kolmogorov-Smirnov test) watches every strategy and agent and halves the size of one whose behaviour changes for the worse.

**It executes with care.** Orders support market, limit, stop, stop-limit, trailing stop, bracket, OTO, OCO and iceberg types. Large orders can be worked through TWAP, VWAP or Implementation Shortfall algorithms, a TWAP tilting its slices with the quote imbalance. Every buy passes pre-trade checks before it reaches the broker: the Shariah screen, the kill switch, a daily loss limit, a position limit, its share of the stock's daily dollar volume, the quoted spread for a market order, and the cash it would need, so the account never borrows. The autopilot's stock buys are limit orders capped 2% above the decision price, so an overnight gap cannot overspend, and every order's fill is costed against the price the backtest assumed.

**It protects capital automatically.** A five-step kill switch follows the autopilot's own holdings, not the account: normal, watch, half size, sells only, then everything sold and trading halted until a person resumes it. Deposits, withdrawals and trades by hand do not move it, and it is checked every 10 minutes at live prices. The ledger is reconciled against Alpaca's positions on every fill. For coins, where Alpaca provides an order book, a market order is refused when the book cannot absorb it, when the spread is far wider than usual, or when the book and the last minute's order flow both lean hard against it.

**It explains every AI decision.** Each agent decision carries an exact SHAP explanation attributing it to its inputs: the indicators, the headline sentiment and risk, and the weights already held. Explanations, runs, evaluations and order attempts are stored in a hash-chained audit log, with a manifest of the prices seen and the code version, so every result can be traced and repeated.

**It stays compliant.** Every buy is screened against the holdings of Shariah-screened funds whose lists are current, and every holding is monitored for names that leave them. Wash sales are applied to the cost basis for tax reporting. Dividend purification is calculated from per-holding ratios, interest received is counted in full, interest paid is flagged, and zakat is estimated. Pattern day trading is not enforced: FINRA retired that rule on 4 June 2026.

---

### Who It Is For

- **Quantitative researchers** who want a rigorous research pipeline without building multiple-testing control, cost models and bias checks from scratch.
- **Individual investors and small teams** who need Shariah compliance built into every order, not checked afterwards.
- **Portfolio managers** running systematic strategies who need risk controls, a kill switch and explainable AI for oversight.
- **Developers and data scientists** moving from research notebooks to orders who want a clear path from a backtest to the broker.

It is not designed for high-frequency trading, and not for options, futures, margin or international markets. It is built for US equities and spot crypto through Alpaca.

---

### How It Is Built

The platform is a **modular monolith** on **FastAPI**, organised by feature (portfolio, orders, market data, order book, ledger, systematic strategies and AI, scanner, drift, compliance). Features share a core of protocols and never import one another, so each stays independently testable. A **React and TypeScript** terminal, served by the API, is the operator's screen.

Orders from the autopilot go through **NautilusTrader 2.0** with a pure-Python Alpaca adapter, and NautilusTrader's event-driven backtest is the second stage of strategy validation.

Data lives in files: **SQLite** for the ledger, news and sentiment, **Parquet** for eight years of daily bars across the liquid market, and an append-only hash-chained **JSON lines** audit log. Background work (the daily rebalance, the nightly market scan, news polling, sentiment scoring, agent training, drift checks) runs as **APScheduler** jobs inside the process, each catching up after the machine has slept.

The AI stack runs **FinRL** and **Stable-Baselines3** in an isolated sidecar environment that never sees broker keys, with **FinBERT** for headline sentiment and **SHAP** for explanations.

---

### What Makes It Different

**Framework-first, not framework-building.** The platform composes NautilusTrader, FinRL, Stable-Baselines3, FinBERT, SHAP, arch, empyrical and River, and builds custom code only where nothing fits: the Shariah and risk controls, the trading books and the bridge from agents to orders.

**Validation decides what trades.** No strategy, rule or agent trades because it was built. It trades because it passed gates that account for every variant ever tried, and it stops when it no longer passes.

**Explainability is structural.** Every AI decision is attributed, stored and retrievable through the API, beside the hash-chained record of every run.

**Compliance is enforced at the order.** The Shariah screen, the no-borrowing rule and the kill switch sit in front of every buy.

**One broker, done properly.** Orders, positions, trade updates, market data, news and paper trading all flow through Alpaca, which keeps complexity manageable.

---

### What It Is Not

- **Not a high-frequency system.** It trades daily bars and works orders over minutes.
- **Not multi-broker or multi-asset beyond US stocks and spot crypto.**
- **Not a signal marketplace.** It runs your strategies.
- **Not a black box.** Every AI decision is attributed, stored and retrievable.
- **Not over-engineered.** One process with clear module boundaries and file-based storage, not a constellation of services.

---

### Known Limits

- Alpaca provides no order book for stocks, so depth and order-flow controls apply to coins only, and an event-driven backtest cannot model queue position for stocks.
- Alpaca routes orders itself; the platform does not choose venues.
- Backtests carry survivorship bias that Alpaca's data cannot remove.
- The proxy history for the trend rule's significance tests is the whole US market, not a Shariah index.
- FinBERT's weights are licensed for non-commercial use only.

---

### In One Sentence

A Shariah-compliant algorithmic trading platform that takes strategies from rigorous research through gated, explainable AI and rule-based signals to careful Alpaca execution, with risk controls, compliance and a complete audit trail at every step.
