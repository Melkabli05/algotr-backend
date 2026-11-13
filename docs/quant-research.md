# Quant strategy and tooling research

Researched 2026-10-28. Package versions and release dates come from PyPI on that date; everything else is from the sources linked at the end. Many sources are vendor or blog pages, and paper figures marked *unverified* could not be checked against the paper itself. Nothing here is investment or religious advice.

## 1. Constraints that shape everything

algotr is a Shariah-compliant platform, which rules out most of the quant playbook.

- **Compliant setups.** Sources describe a compliant setup as cash-only, with no leverage, short selling or derivatives. Scholars treat margin as riba and shorting as selling what you do not own.
- **Ruled out.** Pairs trading, stat-arb, market-neutral books, perpetuals and options. The long and short halves of strategies such as the SPY noise-area system split apart, and the long-only half is unproven.
- **Screening.** A secondary summary of AAOIFI Standard 21 gives three caps: interest-bearing debt under 30% of market cap, interest-bearing securities under 30%, and non-permissible income under 5% of revenue. AAOIFI, DJIM, S&P, FTSE and MSCI differ, so the same stock can classify differently. The standard text itself was not found.
- **Crypto is disputed.** Most sources say spot is permissible with conditions, and AAOIFI has no definitive ruling. A July 2026 fatwa by Mufti Usmani and others reportedly rules otherwise, but it was only found in a news item. That decision belongs to the account owner and a qualified scholar.
- **Alpaca free plan.** Real-time equities data is IEX only, with 30 websocket symbols and 200 historical calls per minute. SIP data older than 15 minutes is identical to the paid plan. Crypto fees at tier 1 are 15 bps maker and 25 bps taker. Staff say paper accounts enforce pattern-day-trader logic and margin cannot be turned off (forum answer, not official docs).

## 2. Strategies that fit, ranked

| # | Strategy | Evidence | Fit |
|---|---|---|---|
| 1 | Long/flat trend following on BTC/ETH: daily 200-day rule, or the Donchian ensemble (lookbacks 5-360 days, volatility-sized) | The Donchian paper reports a Sharpe above 1.5 on a top-20 coin rotation (*unverified*: Bitcoin-only and cost figures not read). A practitioner test found a plain 200-day long/flat matched the ensemble with a shallower drawdown | Best: long-only, spot, low turnover, so 15/25 bps fees barely matter |
| 2 | Absolute and dual momentum on halal ETFs: Faber's 10-month moving average, Antonacci's GEM, monthly rebalance | Secondary, sympathetic sources; no independent out-of-sample numbers found | Good, but verify the ETFs from issuer documents |
| 3 | Volatility-targeted sizing, de-risking into cash only | Moreira-Muir report large alphas; critics show they fail out of sample and after costs | Use as a sizing rule capped at 1x leverage, not as alpha |
| 4 | Long-only cross-sectional momentum on screened stocks | Time-series momentum evidence is strong but mostly long-short futures. A crypto cross-sectional test lost money out of sample at 138% daily turnover | Needs the halal universe first; turnover decides it |
| 5 | RSI(2) mean reversion with a 200-day filter | Blogs and vendors only; no evidence on decay or costs | Low priority |
| 6 | ML cross-section (Gu-Kelly-Xiu) | Paper is real; nothing found on costs; needs point-in-time, survivorship-free data | Defer |

Portfolio construction: start with equal weight or inverse volatility. An S&P 500 study found 1/N beat hierarchical risk parity on return, and HRP only cut volatility by about 1%.

## 3. Tools

| Need | Use | Notes |
|---|---|---|
| Event-driven backtest and live | Nautilus (in use) | Builtin indicators first. `ParquetDataCatalog` plus `BacktestNode` for large data. We run 2.0.0rc7 |
| Fast parameter sweeps | vectorbt 1.2.0 (2026-10-19) | One tutorial calls the open edition "in maintenance", but the fresh release contradicts that. Vectorized code invites look-ahead bias |
| Metrics and tear sheets | empyrical-reloaded 0.7.1, quantstats 0.0.86, pyfolio-reloaded, alphalens-reloaded | All released in the last two months |
| Portfolio and CV | skfolio 1.12.2, riskfolio-lib 7.4.0, PyPortfolioOpt, cvxpy | skfolio is scikit-learn based; believed to include purged and combinatorial CV (not confirmed) |
| Stats and ML | statsmodels, arch 8.0.0 (GARCH), scikit-learn, lightgbm, optuna 5.0.0 | Sweeps need the trial-count correction in section 4 |
| Data | Alpaca (daily SIP bars free once 15 minutes old), databento 0.89.0, edgartools 5.62.1 (SEC filings) | Yahoo-type sources have survivorship bias; CRSP or Sharadar-grade data is paid |
| Halal screening | Zoya API (free sandbox) or Musaffa API (about 10k calls a month on its entry plan) | Vendor-described; verify before depending on one |
| Ops | apscheduler 3.11.3, mlflow 3.17, prefect 3.9, prometheus-client | Add as needed |
| Skip | backtrader (barely developed), pandas-ta (beta), `ta` (last release 2023), mlfinlab (not on PyPI), QuantLib (derivatives pricing), Google ADK (standing scope boundary) | |

## 4. Validation protocol

- Walk-forward testing, purged and embargoed cross-validation, and combinatorial purged CV (many backtest paths).
- Deflated Sharpe Ratio (Bailey and Lopez de Prado): discounts for the number of trials and fat tails. No maintained library was found; it is about 20 lines from the paper.
- Log every trial, because the count feeds the deflation.
- Crypto data traps: 252 instead of 365 days, forward-filled gaps, a cost model that does not scale with traded value, survivorship in the coin universe, hidden look-ahead.
- Test costs at 10, 25 and 50 bps.
- Prefer fixed, published parameters over optimised ones: nothing fitted means nothing to overfit.

## 5. Going live safely

- Stages: backtest and walk-forward, paper, shadow mode, small live size, scale. Each stage passes criteria before the next.
- Hard limits: per-order notional cap, order-rate limits, daily loss limit. A kill switch cancels working orders and stops new signals.
- Reconcile own fills against the broker's and track slippage, fill quality and trade frequency, not only P&L.

## 6. What algotr implements (status)

| Area | Status |
| --- | --- |
| Four daily-bar strategies (ETF trend, ETF dual momentum, BTC trend, crypto Donchian) with fees, benchmark, fee sensitivity, first-70% / last-30% split and four pass/fail checks | Implemented: `src/app/features/systematic/` |
| Portfolio-level volatility target for the Donchian basket (capped at 1x) | Implemented |
| Daily autopilot (APScheduler, 00:05 UTC) with `off`, `dry-run` and `paper` modes, multi-strategy plans, drawdown stop and halt, catch-up on start, immediate deploy on any change | Implemented |
| Dashboard: one Strategies page driven by `GET /systematic/overview`, settings that save on change | Implemented |
| Halal universe service (screening data, purification) | Not started |
| Deflated Sharpe, purged cross-validation, trial log | Not started |
| Experiment tracking, alerting | Not started |

Measured on the last five years of Alpaca daily bars (to 2026-10-27, from the first date the longest rule is warm; crypto 25 bps per trade, ETFs 5 bps). The dashboard recomputes these:

| Strategy | CAGR | Sharpe | Worst drop | Buy-and-hold CAGR | Checks passed |
| --- | --- | --- | --- | --- | --- |
| `halal-etf-trend` | 11.5% | 0.91 | -21.6% | 14.7% | 4 of 4 |
| `halal-etf-rotation` | 11.1% | 0.81 | -16.3% | 14.6% | 2 of 4 (history only from 2022) |
| `btc-trend-200` | 17.2% | 0.63 | -36.7% | 5.3% | 4 of 4 |
| `crypto-donchian` | 5.3% | 0.47 | -18.3% | 4.7% | 3 of 4 (negative recent 30%); about 20% invested at a 25% volatility target |

Trend following gave up return against buy and hold on the ETFs in a strong equity market and traded it for a smaller worst drop and a better Sharpe.

These are backtests on a handful of ETFs and two coins; they do not predict future results.

## Sources

- Halal constraints: [halalwallet day trading](https://www.halalwallet.us/is-it-halal/day-trading), [halalscreener](https://halalscreener.app/en/answers/is-day-trading-halal), [arXiv 2608.12634](https://arxiv.org/pdf/2608.12634), [halalwallet AAOIFI on Bitcoin](https://www.halalwallet.us/blog/aaoifi-cryptocurrency-shariah-ruling-bitcoin-2026), [cryptotimes](https://www.cryptotimes.io/2026/07/12/pakistans-top-islamic-scholar-rules-bitcoin-ethereum-and-usdt-haram/)
- Alpaca: [market data](https://docs.alpaca.markets/docs/trading-on-alpaca/market-data/), [crypto fees](https://docs.alpaca.markets/docs/crypto-fees), [forum PDT](https://forum.alpaca.markets/t/paper-account-trade-denied-due-to-pattern-day-trading-protection/12048)
- Strategies: [Concretum Donchian](https://concretumgroup.com/catching-crypto-trends-a-tactical-approach-for-bitcoin-and-altcoins/), [CXO Advisory](https://www.cxoadvisory.com/technical-trading/crypto-asset-trend-following-strategies/), [Summit Ward](https://summitward.com/learn/crypto-trend-following), [Moskowitz-Ooi-Pedersen](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf), [crypto momentum backtest](https://github.com/prams2104/crypto-momentum-backtest), [Alpha Architect dual momentum](https://alphaarchitect.com/book-review-dual-momentum-investing/), [JF 2024 volatility-managed](https://econpapers.repec.org/RePEc:bla:jfinan:v:79:y:2024:i:6:p:3859-3891), [Gu-Kelly-Xiu](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf), [HRP study](https://arxiv.org/pdf/2509.03712), [Zarattini et al. SPY](https://alexandria.unisg.ch/server/api/core/bitstreams/a99aba00-f967-49b3-aceb-f544dc386e0b/content)
- Tools and data: [Nautilus catalog](https://nautilustrader.io/docs/latest/concepts/data/catalog/), [QuantRocket survivorship](https://www.quantrocket.com/blog/survivorship-bias/), [Musaffa](https://musaffa.com/screening-methodology/), [halalterminal API comparison](https://www.halalterminal.com/blog/posts/best-islamic-finance-apis-2026)
- Validation and ops: [CV and DSR overview](https://halleytech.substack.com/p/overfitting-purged-cv-deflated-sharpe-quant-research), [Bailey and Lopez de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), [NexusFi live deployment](https://nexusfi.com/a/automation/algo-trading-live-deployment)
