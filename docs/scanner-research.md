# Scanner and news: research notes

Written 2026-11-02. Research only, no code. The plan built on these notes is in [scanner-plan.md](scanner-plan.md).

Each finding is tagged:

- **Measured**: run against our own Alpaca paper account on 2026-11-02. The scratch scripts are not kept in the repo.
- **Documented**: stated by the cited source. Numbers from papers come from abstracts or search excerpts, not full text.
- **Unverified**: could not be confirmed. Listed again in section 8.

## 1. What Alpaca gives us

| Topic | Finding | Tag |
|---|---|---|
| Asset universe | 14,397 active US equity assets: NASDAQ 5,614, NYSE 2,887, ARCA 2,748, BATS 1,676, OTC 1,170, AMEX 292, TXSE 10. Tradable 13,498; tradable and not OTC 13,189; fractionable 7,882. | Measured |
| Asset fields | `symbol, name, exchange, class, status, tradable, fractionable, shortable, marginable, easy_to_borrow, attributes`, margin fields. No sector, industry, market cap, shares outstanding or fundamentals. | Measured |
| Delisted history | 19,159 inactive assets, 18,326 ticker-like (the rest are CUSIP-style ids the data API rejects). Of 40 sampled tickers only 6 had daily bars, ending 2019 to 2022. SIVB, BBBY, FRC, TWTR and ATVI are absent. A survivorship-free history cannot be built from Alpaca. | Measured |
| Plan | Basic (free): IEX real time, SIP data only older than 15 minutes, 30 websocket symbols, 200 historical calls/min. Algo Trader Plus ($99/mo): all exchanges, no recency limit, unlimited websocket symbols, 10,000 calls/min. | Documented |
| Rate limit | `x-ratelimit-limit: 200` on both the bars and news endpoints. | Measured |
| SIP snapshots | Refused on our plan: "subscription does not permit querying recent SIP data". IEX snapshots work but IEX volume is a small share of consolidated volume, so liquidity filters must use SIP daily bars. | Measured |
| Multi-symbol daily bars (SIP, adjusted, `end` 16 min ago) | About 10,000 rows/s. 100 symbols x 1y: 2.7 s. 500 x 1y: 11 s. 12,666 symbols x 45d: 82 s. IEX snapshots for 1,000 symbols: 0.4 s. | Measured |
| Screener | `ScreenerClient` works on our plan but the stock endpoints returned one row before the open (`most_actives`, `movers`). Crypto movers returned 20 gainers and 50 losers, including non-USD pairs. Not a basis for a scanner. | Measured |
| Calendar | `/v2/calendar` returns 1970 to 2029 trading days. | Measured |
| Crypto assets | 73 active pairs, about 36 distinct coins, quoted in USD, USDC, USDT and BTC. Includes stablecoins (USDT, USDC, USDG), a gold token (PAXG) and meme coins. | Measured |
| Crypto history | Daily bars from 2021-01-01 at best. BTC, ETH, DOGE, LINK, LTC, UNI full; SOL 1,714 bars; AVAX from 2021-11; DOT from 2023-08; XRP from 2024-01; ADA, HYPE, ONDO under 270 bars. One multi-symbol request for 18 pairs took 2.8 s. | Measured |

Sizing the investable universe (SIP daily bars, 12,666 listed tickers including ETFs, price at least $5):

| 20-day average dollar volume | Names | Non-ETF by name heuristic | Fractionable |
|---|---|---|---|
| at least $1M | 5,922 | 3,250 | 5,460 |
| at least $5M | 4,182 | 2,695 | 4,043 |
| at least $20M | 2,786 | 1,989 | 2,734 |
| at least $50M | 1,896 | 1,419 | 1,875 |

The ETF heuristic matches name keywords and is approximate. Measured.

## 2. Shariah screening

### 2.1 The standards differ

From the Halal Terminal comparison, which is a vendor article. S&P's own methodology PDF returned 403 to our fetcher, so these figures need checking against it. Documented, secondary source.

| Standard | Impure revenue | Debt | Cash and interest-bearing | Receivables | Denominator |
|---|---|---|---|---|---|
| AAOIFI | under 5% | under 30% | under 30% | under 30% | spot market cap |
| Dow Jones Islamic Market | under 5% | under 33% | under 33% | under 33% | 24-month average market cap |
| FTSE Shariah | under 5% | under 33% | under 33% | under 50% | total assets |
| MSCI Islamic | under 5% | under 33.33% | under 33.33% | under 33.33% | total assets |
| S&P Shariah | under 5% | under 33% | under 49% | under 49% | 36-month average market cap |

S&P's 2023 notice describes a two percentage point buffer on the ratios, and a third-party site says S&P and DJIM retired some ratios in 2023. An August 2026 S&P notice revises the accounting screens; its text was cut off. The conclusion that holds either way: a stock can be compliant under one index and not another, so any automatic trading needs one named methodology.

### 2.2 Authoritative lists we can read for free

- **SPUS (SP Funds).** Measured: `https://www.sp-funds.com/wp-content/uploads/data/TidalFG_Holdings_SPUS.csv` returned HTTP 200, dated 11/02/2026, 217 rows. Columns: Date, Account, StockTicker, CUSIP, SecurityName, Shares, Price, MarketValue, Weightings, NetAssets. Three rows are not tradable securities (`Cash&Other`, a CUSIP-style id, a CVR). The page calls the fund "AAOIFI-Screened". The index is S&P 500 based, so the list is large caps only. The fund page also links a purification calculator.
- **HLAL and UMMA (Wahed).** Measured: the fund pages link "Holdings" to a public Google Sheet and publish quarterly dividend purification CSVs (HLAL Q2 2026: dividend per share 0.019, purification 0.00095). The Sheet export returned a Zscaler block page from our network, so access from this machine is unconfirmed. HLAL tracks the FTSE USA Shariah index, about 190 to 200 holdings by third-party counts.
- The union of these lists is a few hundred large and mid cap names, not "any stock".

### 2.3 Paid Shariah screening APIs

Documented on vendor pages, prices unverified. Halal Terminal: token-metered, free tier 50 calls or tokens a month, paid from $29/mo. Musaffa: REST API with a status and 0 to 5 rank per ticker, tiers by monthly calls (10,000 and 100,000), price by sales call. HalalScreener: API from $299/mo. Zoya: no public API price found.

### 2.4 Screening ourselves from SEC data

Measured feasibility, not a recommendation. `data.sec.gov` returned HTTP 200 from our network with a descriptive User-Agent. `submissions` gives SIC code and filings with 8-K item codes. `companyfacts` gives XBRL concepts (503 us-gaap concepts for Apple, 3.8 MB) including LongTermDebt, cash, AccountsReceivableNetCurrent, Revenues and `dei:EntityCommonStockSharesOutstanding`. The ratio screens are computable. The business-activity screen is judgment-heavy and XBRL tags differ between companies, so this is a project by itself and a weak sole authority. The SEC asks automated clients to identify themselves in the User-Agent.

## 3. What to expect from a scanner

### 3.1 Published signals decay

- McLean and Pontiff, J. Finance 2016: across 97 predictors, returns are 26% lower out of sample and 58% lower after publication. Documented.
- Hou, Xue and Zhang, RFS 2020: of 452 anomalies, 65% fail |t| above 1.96 once microcaps are controlled. Documented.
- Chen and Velikov, JFQA 2023: gross anomaly returns decay about 50% after publication, 72% after 2005, and 93% after trading costs; 7% of post-2005 net t-statistics exceed 2. Documented.

Reading: simple published rules on liquid stocks leave little net edge. The honest value of a scanner here is a transparent, audited candidate process plus risk control, not a promise of alpha.

### 3.2 How many variants we can afford

Annualized Sharpe that luck alone reaches as the best of N variants, from our own `validation.expected_max_sharpe`. Measured on the platform's code.

| Variants N | 4 years of daily bars | 6 years | 8 years |
|---|---|---|---|
| 4 | 0.53 | 0.43 | 0.37 |
| 12 | 0.83 | 0.68 | 0.59 |
| 24 | 0.99 | 0.81 | 0.70 |
| 100 | 1.27 | 1.03 | 0.89 |
| 250 | 1.42 | 1.16 | 1.00 |
| 1,000 | 1.63 | 1.33 | 1.15 |

The Harvey, Liu and Zhu bar (t at least 3, so Sharpe at least 3 over the square root of years) is 1.50 at 4 years, 1.22 at 6, 1.06 at 8 and 0.95 at 10. Scanning thousands of stocks is not the multiple-testing problem; trying many rules and parameters is. The variant count must be small and fixed in advance.

Statistical power also limits forward testing: detecting a true Sharpe of 1.0 at t = 3 takes about 9 years of live data. A shadow period catches bugs and cost surprises; it cannot prove an edge.

### 3.3 Momentum ranking

- Clenow's "Stocks on the Move" score is the annualized exponential regression slope over 90 days times R-squared, with a 100-day average filter, a 15% single-day gap filter, ATR sizing and weekly rebalancing, applied to the S&P 500. Documented by secondary sources; the exact filters are not checked against the book. An independent backtest found no correlation between 5-day forward returns and the score. Documented.
- Measured walking skeleton (SPUS list to Alpaca bars to features, 13 seconds in total): 217 listed, 214 tradable, 212 with 252 days of history, all 214 above $20M volume, 135 above their 200-day average, 117 after the gap filter. The Clenow score and 12-1 month return agree only at Spearman 0.19 over the passing names. Two standard momentum definitions pick very different stocks, which is why the choice must be pre-registered and counted.

## 4. News

### 4.1 What the feed contains. Measured

- Free plan works. `GET /v1beta1/news`, 50 articles per page, 200 calls/min, history to at least 2015-01-01 (earliest AAPL item 2015-01-01). Fields: id, headline, author, created_at, updated_at, summary, content (HTML), images, symbols, source, url. Crypto works with `BTC/USD` and `BTCUSD`.
- Volume: AAPL had 16 articles on 2026-10-28. Market wide had more than 1,250 that day (paging stopped at 25 pages).
- 40 SPUS names over 30 days: 1,121 unique articles, 24 calls. Sources: benzinga 713, fiscal.ai 408. Median 80 articles per day for the 40 names.
- Symbol tagging: 69% of articles tag one symbol, 14% tag six or more (market wraps and macro pieces).
- Type mix by keyword heuristic: analyst rating or price target 26%, earnings or guidance 16%, M&A or legal 3%, market wrap 3%, options flow 3%, post-hoc "why shares are moving" 3%, insider 1%, other 44%.
- Duplicates: 14 exact, and about 11% near-duplicates (similarity above 0.85) in the first 300.
- Analyst-rating headlines are structured ("Evercore ISI Group Maintains Outperform on ExxonMobil Holdings, Raises Price Target to $200") and parseable by rules.

### 4.2 What news is worth

- Boudoukh, Feldman, Kogan and Richardson, RFS 2019: classified firm-specific news (14 categories, 56 subcategories) explains 49.6% of overnight idiosyncratic volatility against 12.4% during trading hours; days with identified news have more than twice the volatility. Event type and tone both matter. Documented. For us: much of the move happens overnight, after our 00:05 UTC run and before the open, so a pre-open check matters more than a better nightly score.
- LLM sentiment: Lopez-Lira and Tang evaluate only after the model's knowledge cutoff and report returns declining as LLM use rises. Glasserman and Lin show look-ahead bias and test anonymizing company names. Gao, Jiang and Yan find memorization substantially inflates apparent predictive power and that masking removes only some of it. Documented. For us: never backtest an LLM on dates inside its training window; evaluate forward only.

### 4.3 NLP options

| Option | Facts | Tag |
|---|---|---|
| Rules | Parse analyst actions, earnings, M&A and legal keywords. Deterministic, free, covers most of the volume. | Measured volume |
| FinBERT (ProsusAI) | BERT-base, 0.1B parameters, trained on Financial PhraseBank with further pretraining on Reuters TRC2, labels positive, negative, neutral. Weights are CC BY-NC-SA 4.0, non-commercial. The model card reports no accuracy. Needs torch and transformers, so it would sit in an isolated environment like the FinRL sidecar. | Documented |
| Claude Haiku 5.5 | $0.10 input and $0.50 output per million tokens, Batch API at half price. The 5.5 family rejects non-default sampling parameters, so outputs cannot be made deterministic: store every raw output. Structured outputs supported. | Documented (Claude API reference, cached 2026-10-06) |

Cost estimate for Haiku 5.5, assuming about 250 input and 80 output tokens per article (our estimate) and our measured 0.93 articles per name per day: about $0.5 a month for the ~250 Shariah-listed names, about $4 a month for all ~2,700 liquid stocks, before the Batch discount and before thinking-token overhead. Cost is not the constraint; determinism and leakage are.

### 4.4 Earnings and events from SEC EDGAR. Measured

8-K Item 2.02 marks an earnings release (Apple: 2026-07-30, 2026-04-30, 2026-01-29). That gives actual past earnings dates for free. Upcoming earnings dates need a calendar source; Finnhub's free-tier limits could not be confirmed (60 calls/min is repeated by secondary sources only).

## 5. Libraries

PyPI facts on 2026-11-02. "Python 3.13 and pandas 3" is our environment.

| Library | Version, date | Facts | Fit |
|---|---|---|---|
| alphalens-reloaded | 0.5.1, 2026-10-26 | Factor IC and quantile analysis; accepts pandas below 4; pulls matplotlib, seaborn, IPython | Research tool, dev dependency |
| pandas-ta-classic | 0.9.0, 2026-10-28 | 2 dependencies; indicators | Not needed; rolling means cover our signals |
| ta, pysentiment2, backtrader | 2023, 2020, 2023 | Stale | Reject |
| zipline-reloaded | 3.2.1, 2026-10-25 | 30 dependencies, own data bundles and calendars; has the Pipeline factor and filter model | Reject; borrow the model only |
| openbb | 5.0.0, 2026-09-29 | 24 dependencies, provider keys | Reject |
| vectorbt | 1.2.0, 2026-10-19 | 18 dependencies, license not in metadata | Reject; we have an engine |
| edgartools | 5.62.1, 2026-10-23 | 22 dependencies | Direct `httpx` to the SEC JSON is enough |
| finnhub-python | 2.4.29, 2026-06-24 | Free-tier coverage unverified | Hold |
| transformers | 5.22.1 | Needs torch | Only inside a sidecar, if FinBERT is chosen |
| rapidfuzz | 3.14.6 | 0 dependencies | Optional; stdlib `difflib` is enough at our volume |
| duckdb | 2.1.0 dev | | Not needed; pandas plus Parquet (pyarrow is already installed) |

## 6. Data caveat that shapes storage

Adjusted bars (`adjustment=ALL`) are rescaled retroactively at every split and dividend, so appending new adjusted bars to stored history silently corrupts it. The design must either store raw bars plus corporate actions, or refetch an overlap window each night and fully refresh any symbol whose overlap changed.

## 7. Environment caveat

This network sits behind a Zscaler gateway. It blocked a Google Sheets export and, intermittently, issuer pages. Any list download must cache the last good copy, report staleness, and never trade from a list older than a set age.

## 8. Unverified

1. S&P's current Shariah thresholds (PDF blocked) and the August 2026 revision.
2. HLAL and UMMA holdings download from our network.
3. Finnhub free-tier endpoints and limits; any source of upcoming earnings dates.
4. FinBERT accuracy figures; license terms of the Loughran and McDonald word lists for commercial use.
5. Exact Clenow filters and ATR rule (book not consulted).
6. Whether the Alpaca screener is useful after the open (we only saw it before).

## 9. Sources

- Alpaca: [market data plans](https://docs.alpaca.markets/docs/about-market-data-api), [news API reference](https://docs.alpaca.markets/us/reference/news-3), [alpaca-py screener](https://alpaca.markets/sdks/python/api_reference/data/stock/screener.html)
- SEC: [EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- Shariah: [Halal Terminal standards comparison](https://www.halalterminal.com/blog/posts/shariah-screening-standards-compared), [S&P 2023 notice](https://spglobal.com/spdji/en/documents/indexnews/announcements/20230804-1465479/1465479_spshariahindices-8-4-2023.pdf), [SP Funds SPUS](https://www.sp-funds.com/spus/), [SPUS holdings CSV](https://www.sp-funds.com/wp-content/uploads/data/TidalFG_Holdings_SPUS.csv), [Wahed HLAL](https://www.wahed.com/hlal), [Halal Terminal API article](https://www.halalterminal.com/blog/posts/best-islamic-finance-apis-2026), [Musaffa pricing](https://musaffa.com/pricing/), [HalalScreener pricing](https://halalscreener.app/en/pricing)
- Evidence: [Hou, Xue and Zhang](https://econpapers.repec.org/RePEc:oup:rfinst:v:33:y:2020:i:5:p:2019-2133.), [Chen and Velikov](https://core-prod.cambridgecore.org/core/services/aop-cambridge-core/content/view/945133D5A3ECEEAF466AEE91551FD225/S0022109022000874a.pdf/zeroing-in-on-the-expected-returns-of-anomalies.pdf), [McLean and Pontiff via vLex](https://law-journals-books.vlex.com/vid/does-academic-research-destroy-855658380), [Boudoukh et al.](https://econpapers.repec.org/RePEc:oup:rfinst:v:32:y:2019:i:3:p:992-1033.), [Glasserman and Lin](https://arxiv.org/abs/2309.17322), [Lopez-Lira and Tang](https://arxiv.org/abs/2304.07619), [Gao, Jiang and Yan](https://arxiv.org/abs/2512.23847)
- Momentum: [Clenow score summary](https://www.quant-investing.com/blog/how-to-find-stocks-on-the-move-with-a-better-momentum-indicator-exponential-regression/), [independent test](https://chrischow.github.io/dataandstuff/2018-11-10-how-not-to-invest-clenow-momentum)
- Models: [FinBERT model card](https://huggingface.co/ProsusAI/finbert)
