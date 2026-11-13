# Scanner and news: plan

Written 2026-11-02. Phases 1 to 6 were built the same day on the recommended defaults for section 12, whose questions are still unanswered; section 16 records what was built, what was measured and where the build departed from this plan. The evidence behind every number is in [scanner-research.md](scanner-research.md).

## 1. Goal

1. Scan the whole liquid US market (and a short crypto list) automatically every day.
2. Rank what is worth owning, show why, and feed an automatic trading sleeve.
3. Add news and event analysis around each candidate.

Non-goals: intraday or day trading, shorting, options, trading any stock that is not Shariah-listed, letting an LLM make trade decisions, scraping social media, agent frameworks (ADK stays out of scope), paid data in the first version, and any claim of alpha.

## 2. Constraints carried over

- Shariah compliant: long only, cash, no leverage, no derivatives. Automation trades paper accounts only.
- Daily bars. Alpaca Basic plan: 200 calls/min, SIP data older than 15 minutes, one data websocket (already used by the live stream), so the scanner uses REST only.
- Library first. Every decision is an audit-log entry, and every claim is checked by the five-check gate plus the Deflated Sharpe, as for the existing strategies.

## 3. Shape of the solution

Scan broadly, trade narrowly. Anything liquid can be scanned and shown. Only names on a Shariah list can ever be traded automatically.

```
Alpaca assets -> universe filters -> daily panel (Parquet, incremental)
                                          |
SPUS / HLAL lists -> Shariah gate --------+
                                          v
                       point-in-time features (vectorized)
                                          v
                eligibility filters -> score -> ranked candidates -> scan log (audit)
                                          ^                              |
Alpaca news -> dedupe, tags, events -> overlay (flags, veto) ------------+
SEC 8-K, SIC -> event dates ------------+                                v
                                                     mode: shadow | trade (gated)
                                                                         v
                                                sleeve targets -> build_plan -> orders
```

Three layers of value, each useful on its own: (1) a daily candidate list with reasons and news context, (2) shadow tracking of how the picks performed, (3) automatic trading only if the gates in section 9 pass. If the gates never pass, layers 1 and 2 still deliver the scanner you asked for.

## 4. Modules

New feature folder, following the repo's feature layout. Tests sit beside the code.

```
src/app/features/scanner/
  schemas.py     Candidate, ScanResult, Funnel, NewsItem, EventFlag
  lists.py       Shariah list adapters (SPUS CSV download, HLAL file), sanitizing, dated copies, staleness
  bars.py        batched Alpaca daily bars; a symbol Alpaca refuses is set aside, not fatal
  store.py       Parquet panel: atomic write, overlap check for adjusted prices, merge
  universe.py    assets -> tradable and liquid
  refresh.py     one refresh: lists, the census of the market, the missing history
  market.py      cached read model of the last stored session
  service.py     Scanner: state, the 00:01 UTC job, the startup catch-up
  features.py    momentum, trend, dollar volume, gaps, relative strength
  rank.py        eligibility, the eight variants, weights, the monthly and weekly schedule
  scan.py        today's ranking, the picks and the reasons for each stock
  evaluate.py    the eight-variant backtest and its judgement
  news.py        ingest, normalize, dedupe, symbol weights, rule-based event typing
  events.py      SEC submissions: 8-K item codes, SIC
  sentiment.py   scorer protocol; rules now, optional LLM scorer later
  shadow.py      nightly scan log, forward returns, tracking report
  sleeve.py      adapter into the planner
  router.py      API
```

## 5. Data and storage

- **Universe.** `get_all_assets` (1.8 s) -> tradable, exchange not OTC, alphabetic ticker -> liquidity filter from SIP daily bars: price at least $10 and 20-day average dollar volume at least $20M (about 2,000 stocks; thresholds are parameters). The Shariah gate then keeps names on at least one list, with the list name and date stored on every row.
- **Panel.** One Parquet file, long format (symbol, date, open, high, low, close, volume), rewritten atomically each night. About 3,000 symbols x 2,000 days is tens of MB. Cold build for 214 names took 10 seconds in the skeleton. Measured later (section 16): 6 minutes for the cold build with four requests at a time, and about 70 seconds a night, almost all of it the census.
- **Adjusted prices.** Bars are fetched with split and dividend adjustment, which rescales history retroactively. Each night refetch a 10-day overlap and compare it with the stored values; any symbol that differs beyond a tolerance gets a full refresh. A test with a simulated split guards this.
- **Shariah lists.** Download, sanitize (drop cash, CUSIP-style ids, CVRs), store a dated copy, keep the last good copy on failure. A list older than 7 days stops new buys. The same code reads the quarterly purification files so dividends to purify are tracked.
- **News.** Headline, summary, metadata only (not the HTML body), keyed by article id, `created_at` and `updated_at` both stored, appended idempotently. Only tracked symbols: Shariah-listed plus current candidates.
- **Scan log.** One audit entry per scan: universe version hash, list versions and dates, parameter hash, data fingerprint, ranked result.

## 6. Signals, version 1

Pre-registered. The total number of variants tried is fixed at 8, so the luck bar for 8 years of data stays near 0.5 Sharpe (research section 3.2). Adding variants later is allowed but every one is counted.

Features, all computed from data up to and including the decision date:

| Feature | Definition |
|---|---|
| trend | close above its 200-day average (eligibility, not a score) |
| mom_12_1 | close 21 days ago divided by close 252 days ago, minus 1 |
| mom_6_1 | close 21 days ago divided by close 126 days ago, minus 1 |
| clenow_90 | annualized exponential regression slope over 90 days times R-squared |
| adv20 | 20-day mean of close times volume |
| gap90 | largest single-day move in 90 days |

Variants: score in {mom_12_1, mom_6_1, clenow_90, rank-average of the first two} x top N in {10, 20} = 8. Eligibility for every variant: trend, adv20 and price thresholds, at least 252 days of history, gap90 at most 15%.

Benchmarks: SPUS buy and hold, and the equal-weight basket of the same Shariah-listed liquid universe. The second is the real test of stock-picking skill; beating cash or the cap-weighted ETF does not show it.

## 7. Selection, sizing, guardrails

- Hold the top N by score, weights inverse to 60-day volatility, capped at 10% per name. Weekly eligibility check, monthly re-rank, 5% no-trade band (the planner already has one).
- A name that loses the trend, leaves the Shariah lists, or triggers a veto is sold at the next run. Names the scanner bought are tracked by the planner's existing "managed" set, so turning the scanner off sells them.
- Hard guardrails, checked in code and recorded in the audit entry:
  1. Shariah list membership, list no older than 7 days.
  2. Liquidity and price thresholds.
  3. Sleeve cap: 25% of the managed allocation to start, hard maximum 30%.
  4. At most 3 new names a day.
  5. News veto: no new buy within 3 trading days of a high-severity negative event (M&A pending, regulatory action, lawsuit, guidance cut, halt).
  6. Scan dated for the last complete session, otherwise the sleeve does nothing.
  7. The existing drawdown halt, paper-only rule and audit chain.
- Execution constraint: the Nautilus route sizes in whole units, so a $600 stock against a $1,250 position is rounded badly (upstream issue about fractional equity sizes is still open). Default: route the scanner sleeve through the Alpaca engine, which supports fractional notional orders, and keep Nautilus for the ETF core. Fallback: require each name to allow at least 2 shares.

## 8. News and events

**Ingest.** Poll `GET /v1beta1/news` for tracked symbols every 15 minutes, 50 per page; the budget is a few dozen calls a day against 200 a minute. REST only.

**Normalize.**
1. Near-duplicate clustering within a symbol and day (stdlib `difflib`, similarity above 0.85; about 11% of articles).
2. Symbol weight 1 divided by the number of tagged symbols; articles tagged to six or more symbols are ignored for attribution.
3. Source recorded (benzinga, fiscal.ai); treated as separate feeds.

**Classify by rules first.** Analyst action and price-target change parsed from the headline pattern; earnings, M&A, legal, insider, halt keywords. Each item gets an event type and a severity. Types follow the finding that event type and tone both matter (research section 4.2).

**Events from the SEC.** 8-K Item 2.02 dates give "days since last earnings"; SIC gives a coarse industry for concentration limits. Upcoming earnings dates need a calendar source (open decision); until then the scanner shows "earnings likely within N days" from the quarterly cadence, labelled as an estimate. Requests use a descriptive User-Agent and stay well under the SEC rate limit.

**Features per name.** news count 3 days, attention z-score against 60 days, rule-based tone, event flags, days since earnings.

**Use.**
1. Display: each candidate shows recent headlines with event type and source.
2. Veto and flags (section 7).
3. Pre-open review: 45 minutes before the open (computed from the Alpaca calendar, not hard-coded, because the UTC open shifts with daylight saving), re-read news for symbols with queued buy orders and cancel those with a new high-severity negative event. This covers the overnight window where most news-driven variance occurs and the nightly run cannot see.
4. Not a score input until shadow data shows it helps: its information coefficient must be positive and stable at the same significance bar as everything else.

**Optional ML tier.** A scorer interface lets a model replace rules for tone. If used: Claude Haiku 5.5 on the shortlist only, structured output, every raw output stored with prompt, model id and request id, never re-run for history, evaluated forward only because outputs cannot be made deterministic and a model may have memorized past headlines. FinBERT is the local alternative: non-commercial license and a torch sidecar. Neither is needed for the first news release.

## 9. Evaluation and promotion

Backtest on the survivorship-biased panel is exploratory. Alpaca cannot supply delisted history, so results are optimistic by an unknown amount. The process is built so that bias can only make the system more cautious.

1. **Signal study.** `alphalens-reloaded` on each score: information coefficient and quantile spreads at 1, 5 and 21 days. Dev dependency only.
2. **Strategy test.** The existing lab and checks on the 8 variants with trial counting: Sharpe above the benchmark, smaller drawdown, profitable in the last 30%, profitable at 50 bps per trade, Deflated Sharpe at least 95%, probability of overfitting reported.
3. **Shadow.** From the first nightly scan, log the picks and compute their forward 1, 5 and 21 day returns against both benchmarks, using stored prices only. This is bias-free but slow: proving a Sharpe of 1.0 at t = 3 needs about 9 years, so shadow results are integrity checks and early warnings, not proof.

Promotion to automatic trading requires all of:

- the five checks pass against the equal-weight Shariah universe benchmark, with the trial count included;
- at least 8 weeks of shadow with every nightly scan logged and the audit chain intact;
- realized turnover within 25% of the backtest figure;
- no unresolved data-integrity alert in the last 4 weeks.

If any fails the scanner stays in shadow and shows candidates only. As with strategies today, you can still select it by hand; it is then labelled "Failed checks".

## 10. Integration with the existing pipeline

- `StrategyName` is a closed literal and `build_plan` looks each strategy up in a fixed catalog. Minimal change: add one name, `scanner`, and a sleeve adapter. A sleeve is the strategy's closes frame (the held and candidate names only) plus a target weight per symbol. Existing strategies are wrapped by an adapter that returns the last row of `signals.weights`. `risk_shares`, order sizing and execution stay as they are.
- Config gains `scanner: off | shadow | trade` (default off) and `scanner_share`. With the scanner off the planner must produce identical output; a golden test pins this.
- Schedule: scanning runs as the first step of the existing 00:05 UTC job (it takes seconds incrementally). Two small jobs are added: the news poll and the pre-open review.
- Audit kinds added: `scan`, `news`, `review`. The Overview's "Next orders" and strategy lines pick up scanner holdings without changes to those panels.

## 11. API and screen

Endpoints: `GET /scanner/status` (funnel, list health, last refresh, progress; built), `POST /scanner/run` (built), `GET /scanner/latest`, `GET /scanner/symbol/{symbol}` (features, why or why not, news), `GET /scanner/shadow`, `PUT /scanner/config`, `POST /scanner/run`.

A new **Scanner** page between Overview and Research, in the existing design system (terse copy, tables and badges, no emoji):

- A funnel strip of inline figures: liquid universe, Shariah-listed, above trend, candidates.
- A candidates table: rank, symbol, name, score, 12-1 return, relative strength, trend badge, dollar volume, news badge, earnings badge, Shariah list badge.
- Selecting a row opens a price chart with the 200-day line and a headlines table (time, source, event type, tone).
- A "Why not" box: type a symbol and see which filter removed it.
- A shadow table: scan date, picks, forward returns against the benchmarks.
- A mode control (Off, Shadow, Trade) and the sleeve share, mirrored in Settings.

The Research page gains a "Scanner evaluation" tab for the signal study and lab results.

## 12. Decisions needed

| # | Decision | Recommendation |
|---|---|---|
| 1 | Shariah authority for automatic trading | SPUS and HLAL holdings (free, official, about 250 large and mid caps). Add a paid API later only if breadth matters. Do not build our own screen. Tell me which indexes you accept. |
| 2 | Scope | Scan every liquid US stock, trade only listed names. The alternative is to scan only the listed names, which is simpler and about 250 stocks. |
| 3 | News analysis | Rules first. An LLM scorer is optional later (about $0.5 to $4 a month, forward-only evaluation). FinBERT only if you want everything local. |
| 4 | Crypto | A curated list of about 10 coins with at least a year of history, display only until the gates pass. Not "any coin". |
| 5 | Data plan | Stay on the free Basic plan. The $99/mo plan only helps real-time SIP, which a daily system does not need. |
| 6 | Sleeve policy | Start in shadow at 0%. 25% of the managed allocation once the gates pass. Manual override allowed with a visible label. |
| 7 | SEC contact | The SEC asks automated clients to identify themselves in the User-Agent. Should I use a contact address from your settings? I will not use your email unless you say so. |

## 13. Phases

| # | Phase | You get | Acceptance | Size |
|---|---|---|---|---|
| 0 | Decisions | Answers to section 12; check the unverified items (HLAL sheet from this network, S&P thresholds) | This document updated | S |
| 1 | Data foundation | Universe, lists, panel, nightly job, funnel endpoint | Funnel within 5% of the research figures; list parser drops the 3 non-securities; stale-list guard; overlap refresh handles a simulated split; 214-name cold build under 60 s, incremental under 5 s | M |
| 2 | Signals and evaluation | Features, ranking, signal study, lab run with 8 counted variants | Truncation test passes (features on truncated data equal features on the full panel at that date, so there is no look-ahead); study tables produced; lab reports Deflated Sharpe | M |
| 3 | Scanner page and shadow | Daily candidates, why and why not, shadow tracking | Page renders on real data; scan entries verify the chain; shadow returns match a hand calculation on 3 dates | M |
| 4 | News and events | Headlines per candidate, flags, pre-open review | Idempotent ingest; dedupe fixture passes; analyst-action parsing covers at least 95% of rating headlines in a 30-day sample; review cancels a seeded negative case on paper | L |
| 5 | Automatic sleeve | Sleeve, config, guardrails, promotion gate | All existing tests pass; scanner off is a golden no-op; gate blocks on a failing check; off sells scanner positions | L |
| 6 | Crypto list | Curated crypto scan, same gates | History and stablecoin filters tested | S |
| 7 | Optional ML scorer | Structured LLM tone on the shortlist | Outputs stored; daily cost logged; forward-only report | M |

Phases 1 to 3 already deliver the scanner view you asked for. Phase 5 is the only one that changes what the system trades.

## 14. Risks

| Risk | Mitigation |
|---|---|
| Survivorship bias makes backtests look better than reality | Treat backtests as exploratory, require shadow evidence, keep the sleeve small, benchmark against the same universe |
| A name stops being Shariah compliant | Daily list refresh, sell at next run, stale-list stops buying, purification file tracked |
| Methodologies disagree on compliance | One named authority per decision 1; the list name is stored on every row |
| Adjusted prices corrupt stored history | Overlap comparison each night, full refresh on mismatch, tested |
| Free-plan limits | REST only, batched calls, caching; measured throughput is 10,000 rows/s against a 200 calls/min cap |
| News attribution noise and duplicates | Tag-count weights, clustering, rules before ML |
| LLM leakage and non-determinism | Forward-only evaluation, raw outputs stored, never a trade decision |
| Too many variants | Fixed at 8; each addition counted by the Deflated Sharpe |
| Overnight gaps and thin names | Liquidity floor, pre-open review, at most 3 new names a day |
| Whole-share rounding on the Nautilus route | Alpaca engine for the sleeve, or a minimum-shares rule |
| Corporate network blocks (Zscaler) | Cache last good list, show staleness, never trade on a stale list |
| Content terms for news | Personal display only; do not republish headlines |

## 15. Cost

Baseline: $0 (free Alpaca plan, issuer files, SEC, rules). Optional: a Shariah screening API from about $29 a month; an LLM tone scorer about $0.5 to $4 a month by our estimate; Alpaca Algo Trader Plus $99 a month, not needed.

## 16. Build status, 2026-11-02

| Phase | Status | Measured |
|---|---|---|
| 0 Decisions | Proceeded on the defaults in section 12; no answers yet | The SEC contact (needed in phase 4) is still open |
| 1 Data foundation | Built | Liquid universe 2,645 against 2,638 in the research (0.3%); all 214 SPUS names match Alpaca; the parser drops the three non-securities; a list older than 7 days stops counting; a simulated split rebuilds only that symbol; cold build of 2,645 symbols took 359 s; the nightly refresh takes about 70 s, almost all of it the census of 12,745 symbols |
| 2 Signals and evaluation | Built | Features pass a truncation test at several cut dates, and so do the strategy returns; the Clenow score matches a scipy least-squares fit; eight variants, all counted; first result below |
| 3 Scanner page and shadow | Built | The page renders on real data in both themes; scan entries verify in the audit chain; forward returns match a hand calculation on three dates |
| 4 to 7 | Not started | |

**First evaluation (real panel, SPUS list of 2026-11-02, 214 names, Dec 2019 to Oct 2026).** The equal-weight benchmark of the same names: CAGR 21.3%, Sharpe 1.00, max drawdown 35.7%. The best variant (trend strength, top 10): CAGR 23.8%, Sharpe 0.92, max drawdown 30.7%, Deflated Sharpe 83%. All eight pass three of the five checks (smaller drawdown, profit in the last 30%, profit at 50 bps) and fail the other two (Sharpe above the benchmark, Deflated Sharpe above 95%). The probability of backtest overfitting is 64%, so the ranking among the variants is not reliable either. Both the lists and the data are survivorship-biased, which flatters every figure here. The promotion gate in section 9 therefore stays closed and phase 5 is not justified by this evidence; the scanner remains a view and a log.

**Departures from the plan.**
- Endpoints: `/scanner/status` replaces `/scanner/funnel` because it returns more than the funnel; `/scanner/evaluate` and `/scanner/evaluation` were added. The evaluation sits on the Scanner page, not in a new Research tab.
- The signal study with `alphalens-reloaded` was not built. It works on pandas 3.0.5 (checked on synthetic data) but brings 23 packages, among them matplotlib and IPython, for what is a few lines of rank correlation; it can be added as a development-only script if the information-coefficient tables are wanted.
- The census of every tradable symbol runs nightly, not weekly: one code path and an always-current funnel for about a minute of background work.
- Throughput was overestimated. Measured on this account: 3,000 rows/s for sequential multi-symbol requests, 23,000 to 25,000 rows/s with four at a time (not the 10,000 assumed). The fetch runs four requests in parallel and saves every 500 symbols.
- The first live build hung for over 20 minutes in a TLS read: `requests` has no default timeout and alpaca-py sets none. Market-data clients now carry a 60 s timeout and 3 retries (`core/broker.py`); the trading client does not, because a retried order could be placed twice.
- sp-funds.com answers 403 to Python's default client name and 200 to `algotr-scanner/0.1`. The scanner identifies itself honestly and does not imitate a browser; HLAL holdings need a CSV saved by hand.

**What this means for the open decisions.** Decision 6 (sleeve policy) is answered by the evidence for now: shadow only. Decision 1 (which indexes you accept) matters more than before, because the benchmark and the picks both depend on the list. Phase 4 (news and events) can still be built as a display and veto layer without any claim about returns.

**Phases 4 to 6, and where they departed.** The research behind them is in [scanner-features-research.md](scanner-features-research.md).
- News: the veto lasts 5 calendar days (the plan said 3 trading days) and counts only articles tagged to three stocks or fewer. Near-duplicate removal was built and dropped: it was quadratic and merged different brokers' notes while catching 11 real copies in 30 days. Earnings dates are an estimate from SEC item 2.02 filings, which need `SEC_CONTACT` in `.env`; there is no calendar. The pre-open review cancels every open stock buy with a new veto since it was placed, not only the scanner's.
- Crypto: a display-only table in its own panel, from the list in `CRYPTO_COINS`; it is not a gate for anything.
- Sleeve: one cap of 25% of the allocation, no separate 30% maximum; the turnover test is left out for want of data; a stale scan or a failed record freezes the sleeve instead of selling; at most 3 new names a run. Orders go through Alpaca, never Nautilus.
