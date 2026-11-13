# Scanner features: research

Written 2026-11-02, before building the automatic sleeve, news and events, and the crypto scan. It follows [scanner-research.md](scanner-research.md). Each section says what the documentation states, what was measured on this account, which libraries were considered, and what was decided.

## 1. Automatic trading sleeve

**Where it stands.** The first evaluation ([scanner.md](scanner.md)) found no variant that beats holding the same Shariah-listed names equally (best Sharpe 0.92 against 1.00), none reaches a 95% Deflated Sharpe, and the probability of backtest overfitting is 64%. Both the lists and the data are survivorship-biased. The plan therefore says the sleeve may exist but stays closed behind a promotion gate, and a manual override must be visible. The sleeve is built that way.

**Alpaca fractional orders** ([docs](https://docs.alpaca.markets/docs/fractional-trading)).

- Order types market, limit, stop and stop limit, time in force `day` only; `qty` or `notional`, never both, up to 9 decimals; minimum $1.
- The asset must have `fractionable: true`, otherwise the order is rejected ("requested asset is not fractionable"). Paper and live accounts can both trade fractions.
- Fractional sells are always long; cancelling works as for whole shares.
- Extended hours: limit orders, `day`, `extended_hours=true` ([announcement](https://alpaca.markets/blog/fractional-shares-trading-supports-limit-orders-and-extended-hours/)). The page elsewhere says fractions trade as market orders in normal hours, so the sleeve uses market orders in the regular session only.

**What the planner already does.** `rebalance.build_plan` sizes buys as fractional quantities (six decimals) and `rebalance.execute` sends them to Alpaca as `day` market orders. The Nautilus route trades whole shares, so a $1,417 stock against a $1,250 position is rounded badly (the upstream issue about fractional equity sizes is still open). The sleeve's orders therefore always go through the Alpaca route; the ETF core can stay on Nautilus.

**Controls to build in.** The FIA's 2024 white paper on [automated trading risk controls](https://fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf) recommends pre-trade limits on order and position size, enforced in several independent places, and a kill switch that cancels resting orders. It says nothing specific about daily loss limits; the existing drawdown halt covers that. The sleeve adds: a position cap (10%), a sleeve cap (25% of the allocation, 30% hard maximum), at most 3 new names a day, a per-order notional ceiling, a scan that must be for the last complete session, lists that are not stale, a news veto, and a switch that cancels the sleeve's queued orders.

**Decided, and as built.**

- The sleeve is a separate part of the plan, not a fifth strategy: its names come from the scanner's scheduled holdings (monthly re-rank, weekly drop of names that stop qualifying), not from price signals. The autopilot reaches them through the `SleeveProvider` protocol in `core/sleeve.py`, so neither feature imports the other.
- Modes `off`, `shadow` (the plan shows what it would do, nothing is placed) and `trade`. The default is off with a 0% share, and the share is at most 25% of the autopilot's allocation. (The plan said 25% with a 30% hard maximum; one limit is simpler and the stricter one wins.)
- `trade` needs the promotion record, or an explicit override that labels every order and the dashboard "Failed checks". The record is the variant's five checks from an evaluation no older than 30 days, at least 40 logged sessions and an intact audit chain. The plan also asked for realized turnover within 25% of the backtest; 40 sessions hold about two month-ends, too few to measure turnover, so it is left out until there is a year of shadow data.
- A failed record, a stale scan or an old list freezes the sleeve where it stands, because selling on a data fault would turn an outage into trades. It sells only when switched off, when the drawdown stop fires, or when a name leaves the schedule. Shadow touches nothing it already holds.
- At most 3 new names a run, which is a day under the schedule; a manual re-run starts 3 more.
- With the sleeve off the plan is identical to the old one; a test pins that, and the live account's plan was compared with and without it.

## 2. News and events

**Alpaca news** ([reference](https://docs.alpaca.markets/reference/news-3)). `GET /v1beta1/news`, 1 to 50 articles a page, filters `symbols`, `start`, `end`, `sort`, `include_content`, `exclude_contentless`, `page_token`; fields `id`, `headline`, `author`, `created_at`, `updated_at`, `summary`, `content` (HTML), `images`, `symbols`, `source`, `url`. Limits are per minute with `X-RateLimit-*` headers. The documentation lists no plan restriction, but a forum thread reports a 403 on some accounts; this account works. `alpaca-py` has a `NewsClient` with `get_news(NewsRequest(...))` that pages by itself, so no client code is needed beyond the timeout wrapper.

**One trap.** `NewsClient.get_news` pages by itself, 50 an article page at a time, only when `limit` is not set; a `limit` is the total for the whole call. The first probe passed `limit=50` and saw 482 articles where there are 6,698.

**Measured.** 30 days of headlines for the 214 SPUS names: 6,698 unique articles (223 a day). Sources: Benzinga 71%, fiscal.ai 29%. 72% are tagged to one symbol, 19% to two to five, and 9% to six or more (up to 57 tags: market commentary). By type, with the rules below: 31% analyst actions in Benzinga's fixed wording ("X Maintains Buy on Y, Raises Price Target to $Z"), 9% results, 9% market commentary, 8% contracts and launches, 3% guidance, 2% M&A, 2% regulatory, 1% each for dividends and buybacks, insiders, management and legal, 0.2% offerings, 0.1% halts; 32% match no rule and stay "Other".

**Event typing by rules.** Ordered patterns on the headline only, the first match wins, and the more specific type goes first. Of the 1,994 rating headlines tagged to fewer than six symbols, 1,984 (99.5%) are typed as analyst actions, against the plan's 95% bar. Running the first rule set on the full sample showed what the small one had hidden: "Investigational" matched an investigation rule; "Trying To Buy" matched an M&A rule; a company named "Holdings" matched the fund-selling rule; Broadcom's and Nvidia's "interest in" a Meta share offering matched the equity-offering rule; "Lowers Fiscal 2027 iPhone Forecast for Apple" (a bank's forecast) and "Capital Expenditures Were Lower Than Guidance" were read as guidance cuts; an FTC inquiry across several delivery platforms and a quarry's safety order were rated as serious. The rules now put analyst notes before insider and fund wording, accept a guidance cut only with a verb of lowering next to "guidance" or "outlook", call a headline that raises one guidance and lowers another "mixed", and keep an industry inquiry and a safety order at medium severity.

Of the 6,698 headlines 27 (0.4%, about one a day) are vetoes: 19 guidance cuts, 3 halts or bankruptcies, 3 regulator actions, a recall and an equity offering. Three are about someone else (two headlines on one startup's bankruptcy that name Disney, and a post about an FBI investigation of an employee). They stay: a pattern cannot tell whose bankruptcy it is, and a veto only delays a new buy by five days.

**Libraries considered.**

- Sentiment: `pysentiment2` carries the Loughran-McDonald word lists, but the package is GPL and the lists are free for non-commercial research only ([PyPI](https://pypi.org/project/pysentiment2)); VADER is MIT but general-purpose; FinBERT needs a torch sidecar. Tone is not a score input in the plan, so none is added; events and severity come from rules.
- Near-duplicates: none removed. A first version used `difflib` (similarity of 0.85 or more) and was both slow (2.2 s for one symbol's 248 headlines, quadratic) and wrong. Of 226 similar pairs published within six hours of each other, 11 were the same story from the two feeds and 215 were different brokers' notes in the same template ("Stifel Maintains Buy on Microsoft, Raises Price Target to $650" and "TD Cowen Maintains Buy on Microsoft, Raises Price Target to $650"). Exact matching after lower-casing and dropping punctuation finds only 15 of 3,701. Counting every article is simpler and closer to the truth.
- Storage: `sqlite3` from the standard library, for idempotent upserts by article id and indexed lookups, which a Parquet file cannot give.
- SEC filings: `edgartools` exists, but the need is one JSON endpoint, and a plain `httpx` call is shorter than the dependency.

**SEC EDGAR.** The submissions JSON ([sample for Apple](https://data.sec.gov/submissions/CIK0000320193.json)) has `filings.recent` as parallel arrays (`accessionNumber`, `filingDate`, `reportDate`, `acceptanceDateTime`, `form`, `items`, ...), covering at least a year or 1,000 filings, with older ones in extra files ([API page](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)). `items` holds the 8-K item numbers, for instance 2.02 for results. `company_tickers.json` maps tickers to CIKs ([file](https://www.sec.gov/files/company_tickers.json)). The fair-access rules ([page](https://www.sec.gov/os/accessing-edgar-data)): at most 10 requests a second, a declared User-Agent that names the requester with a contact email, no crawling. The contact must be real and must be the user's to give, so the client reads `SEC_CONTACT` from `.env` and does nothing until it is set; the scanner then runs on news alone and says so.

**Earnings dates.** Alpaca has no earnings calendar. Alpha Vantage's `EARNINGS_CALENDAR` returns the whole market's next 3, 6 or 12 months as one CSV ([notes](https://www.macroption.com/alpha-vantage-earnings-calendar/)), and Finnhub has a `/calendar/earnings` endpoint, but neither's free-tier terms could be confirmed from a primary page and both need a key from the user. They are left out. "Days since the last results" comes from SEC item 2.02 filings, and the next date is an estimate from the quarterly cadence, labelled as such.

**Decided.**

- Ingest every 15 minutes for the tracked names (the Shariah-listed names and the shown candidates, and later anything the sleeve holds), and once after each nightly refresh. The first poll for a name reaches back 60 days, so a name added later is filled in on its own.
- Per name: headlines with type, direction and severity; the count and the attention (the last 3 days against the weeks before) for 3 days; the worst recent negative event; a veto flag; and, with SEC filings, the days since the last results and an estimate of the next.
- A veto blocks new buys for 5 calendar days after a high-severity negative event (a guidance cut or withdrawal, a halt, delisting or bankruptcy, an SEC, DOJ or FTC action, an equity offering, a recall or an FDA rejection, a dividend cut) and after a pending acquisition, which pins the price. Only an article tagged to three symbols or fewer counts. SEC items 1.03, 3.01 and 4.02 count the same. It is a veto and a flag, not a score input and not a return claim.
- A review 45 minutes before the open, from the Alpaca calendar, cancels queued buys whose name has a new veto since the order was placed, and writes a `review` entry to the audit log.

## 3. Crypto scan

**Alpaca crypto** ([docs](https://docs.alpaca.markets/docs/crypto-trading)). 24/7; pairs against USD, USDT, USDC and BTC; market, limit and stop-limit orders, `gtc` and `ioc`; no margin, no shorting; a $200,000 cap per order; taker fees from 25 bps at the lowest volume tier, charged in the asset received and posted at end of day. The documentation says "56 pairs"; this account's asset list shows 73 active crypto assets, 36 of them against USD, so the page is out of date.

**Measured** (daily bars for all 36 USD pairs, 20 s; for the ten default coins 2.5 s):

- Alpaca's own volume is tiny (Bitcoin's median is about $120,000 a day), so a dollar-volume floor means nothing here; only history length can be used as a filter.
- Pairs with at least a year of history: BTC, ETH, SOL, XRP, UNI, LINK, AVAX, DOT, AAVE, DOGE, BCH, LTC and a few others. Cardano has 263 days, and ARB, HYPE, ONDO, RENDER, FIL and WIF under a year.
- Stablecoins and a gold token trade as USD pairs (USDT, USDC, USDG, PAXG) and must be left out of any ranking.

**Evidence.** Liu, Tsyvinski and Wu ([Journal of Finance, 2022](https://www.nber.org/papers/w25882)) find one-, two-, three- and four-week momentum earn significant weekly excess returns (2.5% to 4.1%) on coins above $1 million, sample 2014 to 2018, weaker in small coins. So the crypto scan reports 1, 4 and 13 week returns and the 200-day trend, not the equity variants' 12-1 month windows; the existing Bitcoin and Donchian strategies already use trend rules.

**Shariah.** There is no settled ruling. A practitioner framework screens at the asset level (utility, gharar, maysir, riba exposure) and at the transaction level ([Indonesian framework](https://journal.uinsgd.ac.id/index.php/mua/article/view/55121)); commercial guides say yield farming, lending protocols and interest-bearing stablecoin reserves fail, and that meme coins carry more risk than established projects with clear utility ([HalalScreener](https://halalscreener.app/en/blog/is-cryptocurrency-halal-guide)); a scholar interviewed by [Cointelegraph](https://cointelegraph.com/news/could-crypto-be-compliant-with-sharia-law-experts-answer) says each asset needs its own ruling. This is not for software to decide.

**Decided.**

- Display only: no crypto order is ever placed by the scanner, and the sleeve cannot hold a coin.
- The universe is a list the user owns, `CRYPTO_COINS` in `.env`. The default is ten established coins with a year of history and no stablecoin, gold token, meme coin or lending or yield token: BTC, ETH, SOL, XRP, LINK, AVAX, DOT, LTC, BCH, UNI. This is a convenience, not a ruling, and the page says so with the same note the crypto strategies carry.
- Coins with less than a year of history are dropped with the reason shown.
- Crypto lives in its own panel (`crypto.parquet`), because its calendar has weekends and would put empty rows into the stock panel's windows. Alpaca answers an unknown coin with no bars instead of an error, and the daily bar of the current UTC day is included while it is still open, so it is dropped. The figures were checked against bars recomputed from the API for Bitcoin (1, 4 and 13 week returns and the 200-day trend).

## 4. Order of work

1. News and events, because the sleeve's veto and the review need them.
2. The crypto scan, which shares the page and the feature code but nothing else.
3. The sleeve, last, because it changes what the system can trade.

Each step ends with tests, the documentation and the dashboard, and a commit.
