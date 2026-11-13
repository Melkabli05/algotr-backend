# Strategies: evaluation and research

What the platform trades today, measured against what the research says a solid long-only, daily-bar, Shariah-constrained system looks like. Measurements are this account's data on 12 November 2026 (`/systematic/overview`, the scanner evaluation, the audit log) and the exploratory runs marked as such. Papers are cited from their abstracts and from summaries unless a line says otherwise; where a result could not be checked in the paper itself it says so. Earlier work: [quant-research.md](quant-research.md), [ml-research.md](ml-research.md), [systematic.md](systematic.md).

## What is wrong with the current strategies

| Strategy | Rule | What it measured | Verdict |
| --- | --- | --- | --- |
| `halal-etf-trend` | each of SPUS, HLAL held while above its 200-day average | 5.2 years: CAGR 15.0% against 17.4% for holding, Sharpe 1.12 against 0.98, max drawdown -20.4% against -25.4%. Deflated Sharpe 80% on its own history; passes only on the 1926 US market proxy (p 2.2%) | The one running. Its edge is drawdown control, not return |
| `halal-etf-rotation` | two best of four ETFs over 126 days, if positive | Sharpe 0.82 against 1.01 for holding, turnover 20.6 a year, deflated Sharpe 42% | Fails |
| `btc-trend-200` | Bitcoin above its 200-day average | Sharpe 0.53 against 0.43, p 36%, deflated Sharpe 25% | Fails |
| `crypto-donchian` | nine Donchian breakouts vote, 25% volatility target | Sharpe 0.43, out-of-sample -0.32, deflated Sharpe 17% | Fails |
| scanner stock sleeve | momentum rank of 215 listed names, top 10 or 20 | best variant t = 2.3 (the hurdle is 3), deflated Sharpe 80%, probability of overfitting 49% | Shadow only; which variant wins is noise |
| PPO, SAC, A2C, ensemble, FinRL agents | learned weights | none passes "beats the trend rule after the search" or the deflated Sharpe | Fail, as measured in [ml-research.md](ml-research.md) |
| Kronos (zero-shot forecasts) | candle forecasts from a pretrained model | being judged by the same gates ([ai.md](ai.md)) | Pending; independent tests found no edge after costs |

The structural problems, which are separate from whether a rule happens to pass:

1. **One signal, one parameter.** Every rule is a single fixed window (200, 126) turned on or off. A binary switch at one lookback is the most fragile form of trend following: it flips on a one-day cross, the best window drifts over time, and the result depends on a number picked once. [Zakamulin](https://alphaarchitect.com/the-moving-average-research-king-valeriy-zakamulin/) finds that once look-ahead and data mining are removed, moving-average timing is statistically close to buy and hold on the US market, with the benefit concentrated in bear markets.
2. **Two assets that are one bet.** SPUS and HLAL correlate at 0.94 (daily returns, 1,732 days). "Equal sleeves" of them is one position held twice. The only diversifiers in the catalogue are UMMA, SPSK and the coins, and the rotation rule that uses them fails.
3. **No sizing.** Exposure is 0 or the full sleeve. Nothing scales a position by how risky the asset is today, though that is the part of trend following with the most evidence (below).
4. **Nothing learns.** The rules never adapt. The only online components are the drift monitor (River ADWIN) and the agents' UCB1 weights, and neither touches what the live rules do.
5. **Statistical power.** Five years of two correlated ETFs cannot separate a good rule from luck at 17 trials. The platform's answer, testing on a century of the US market, is sound but only covers the one rule with a proxy.

## What the research says

**Trend following works as a diversified, risk-scaled portfolio, not as one binary rule.** [Hurst, Ooi and Pedersen](https://www.trendfollowing.com/whitepaper/Century_Evidence_Trend_Following.pdf) find positive average time-series momentum returns in every decade since 1880 across many markets, strongest in crises. Their evidence is for diversified, volatility-scaled books of many assets. A search found nothing in it on single-asset rules. [Huang, Li, Wang and Zhou (JFE 2020)](https://profiles.wustl.edu/en/publications/time-series-momentum-is-it-there/) are the sceptical counterpart: asset by asset there is little predictability, the pooled t-statistic does not survive a bootstrap, and a strategy that just holds assets with a positive historical mean does about as well. Together they say that trend is worth having mainly for the diversification across assets and the risk scaling, which is what this platform lacks.

**Volatility scaling is the best-supported improvement, with a caveat.** [Harvey et al. (JPM 2018)](https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf), on more than 60 assets, find that scaling to a volatility target raises the Sharpe ratio for equities (about 0.40 to 0.48-0.51 in the summaries I found, which I could not check in the paper), by a leverage effect that works like momentum, and cuts tail risk everywhere. [Cederburg, O'Doherty, Wang and Yan (JFE 2020)](https://experts.arizona.edu/en/publications/on-the-performance-of-volatility-managed-portfolios/) test 103 strategies in real time and find the managed versions do not systematically beat the unmanaged ones, because the spanning regressions behind the claim are unstable. A 2025 international follow-up finds it most promising for momentum after costs, and [Wang and Yan](https://www.lehigh.edu/~xuy219/research/Downside.pdf) find downside-volatility scaling does better than total volatility. So: use it for risk and drawdown, with a simple trailing estimator and no fitted parameters, and do not count on a Sharpe gain.

**Continuous signals and cost awareness cut turnover without losing the edge.** [Baltas and Kosowski](https://papers.ssrn.com/abstract=2140091) replace the +1/-1 signal with a trend-strength measure and a better volatility estimator and cut turnover by more than a third with no significant loss. [Zakamulin and Giner](https://www.returnstacked.com/academic-review/optimal-trend-following-with-transaction-costs/) show the optimal lookback gets longer as costs rise, so the window is a function of cost and not a constant. [Gârleanu and Pedersen (JF 2013)](https://www.nber.org/papers/w15205) give the portfolio rule: trade part of the way toward an "aim" portfolio that blends today's target with where it is expected to be, with slower signals weighted more. This platform's 5% no-trade band is a crude version of it.

**Combining signals (Carver).** [Robert Carver's framework](https://qoppac.blogspot.com.es/2016/01/correlations-weights-multipliers.html), from secondary sources: several speeds of trend each give a forecast scaled to a common size, the forecasts are combined with weights set from their correlations, a diversification multiplier restores the scale, and the combined forecast is capped. That is the standard way to make trend continuous and multi-speed.

**Online learning, where it has real guarantees.** Nothing in the literature supports an online model that predicts daily returns of liquid ETFs. What does have guarantees is online aggregation of strategies:
- [Li and Hoi's survey](https://arxiv.org/pdf/1212.2129) covers online portfolio selection (universal portfolio, exponential gradient, OLMAR). It notes that most of the work assumes no costs, no liquidity limits and no impact.
- Hedge, AdaHedge and Fixed Share ([AdaHedge](https://www.rhul.wouterkoolen.info/adahedge.pdf), [Fixed Share with AdaHedge](https://arxiv.org/pdf/1808.00741)) weight a set of experts by their cumulative loss and, with Fixed Share, follow a best expert that changes.
- The closest result to this platform is a September 2026 paper, [Cost-Sensitive Online Window Size Selection for Portfolio Management](https://arxiv.org/abs/2609.29887) (Liu and Hsieh): each lookback is an expert, the weights are updated with losses that include turnover, and Hedge and Fixed Share get cost-sensitive regret bounds. I read only its abstract and it reports no empirical results there, so treat it as the design to copy, not as evidence it pays.

**Machine learning on the signals.** [Gu, Kelly and Xiu](https://www.nber.org/papers/w25398.pdf) find trees and neural networks the best of the models they test on monthly US stock returns, with small out-of-sample R-squared; a student replication found much smaller values. [Lim, Zohren and Roberts](https://arxiv.org/abs/1904.04912) train a network on 88 futures to maximise the Sharpe ratio of volatility-scaled positions, roughly doubling the Sharpe of the traditional methods before costs. [Wood, Roberts and Zohren](https://arxiv.org/pdf/2105.13727) add online changepoint detection. These need many assets or many stocks, not two correlated ETFs. [Meta-labeling](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy/) (a second model that decides whether to act on a rule's signal) has no strong out-of-sample evidence on trend rules that I could find; the one controlled test found no reliable edge.

**Regimes.** [Nystrup et al.](https://lup.lub.lu.se/record/ec78eede-c592-4b29-8818-7b55881161f1) find regime-based allocation adds value over static weights and cuts drawdowns, with an adaptive-estimation variant. Another study finds regime strategies no better than static ones out of sample. The platform already has a hidden Markov model of regimes ([ml-research.md](ml-research.md)); it should stay a risk input, not a return forecast.

**Stocks, and the Shariah screen.** Residual momentum, which ranks stocks on their returns after removing market and style exposure, gets about twice the risk-adjusted profit of plain momentum at about half the volatility ([Blitz, Huij and Martens](https://econpapers.repec.org/RePEc:eee:empfin:v:18:y:2011:i:3:p:506-521), per a CXO summary: 12.5% against 22.7%), and crashes less ([Robeco, 2023](https://robeco.com/nl-be/visie/2023/02/quant-chart-taming-momentum-crashes)). [Barroso and Santa-Clara](https://core.ac.uk/download/18511704.pdf) scale momentum to constant volatility to tame its crashes. For Shariah-compliant stocks the evidence is mixed: one study finds no ordinary momentum profit (1996-2009), another finds idiosyncratic momentum profits in US Shariah stocks (1986-2020). The leverage screen pushes the universe toward low-debt, quality-like names.

**Allocation.** [DeMiguel, Garlappi and Uppal](https://econpapers.repec.org/RePEc:oup:rfinst:v:22:y:2009:i:5:p:1915-1953) find none of 14 optimised models consistently beats 1/N out of sample, because the estimation error costs more than optimisation gains; risk-based schemes that estimate only volatility and correlation (what the platform already does with equal risk contribution) avoid the worst of it.

## Exploratory measurements

Run on the US market's daily return since 1926 (the platform's proxy), long-only, cash earning nothing, a one-day lag, 5 bps per unit of turnover. This is exploration: eight variants were tried, so any of them adopted would count as eight more trials in the deflated Sharpe.

| Rule | CAGR | Volatility | Sharpe | Max drawdown | Turnover a year | Sharpe by era (1927-60, 61-90, 91-2008, 2009-26) |
| --- | --- | --- | --- | --- | --- | --- |
| buy and hold | 9.8% | 17.1% | 0.63 | -84% | 0 | 0.50 0.79 0.53 0.85 |
| 200-day trend (current) | 9.7% | 11.2% | 0.88 | -36% | 5.4 | 0.86 1.14 0.72 0.82 |
| five speeds blended (50 to 250 days) | 9.7% | 10.2% | 0.96 | -39% | 8.0 | 0.96 1.30 0.70 0.81 |
| four-speed EWMAC, continuous | 7.0% | 8.2% | 0.87 | -30% | 4.5 | 0.91 1.00 0.72 0.76 |
| 200-day trend, 12% volatility target | 8.5% | 8.8% | 0.96 | -23% | 5.1 | 1.08 1.11 0.76 0.77 |
| five speeds, 12% volatility target | 8.5% | 8.1% | 1.04 | -22% | 7.1 | 1.20 1.26 0.76 0.73 |
| holding, 12% volatility target | 8.5% | 11.3% | 0.78 | -58% | 1.1 | 0.76 0.78 0.72 0.86 |

What it says: blending speeds and targeting volatility each help the Sharpe ratio and the drawdown over a century (0.88 to 1.04, -36% to -22%), at the price of about 1.2 points of CAGR and more turnover. Since 2009 none of the trend variants beats holding (0.73 to 0.82 against 0.85): the trend rules' value is in the bear markets that this period lacked. The single 200-day rule is not wrong, but it is the most basic form and the improvements are modest, not transformational. A gain of 0.1 in Sharpe on a century is not something the five years of ETF data can confirm.

## What a solid system looks like here

Constraints that stay: long-only, no leverage, cash, daily bars, Shariah-screened names, every change judged by the five gates and shown in the lifecycle before it gets capital.

1. **Signal layer.** Per asset, a continuous forecast from several trend speeds (Carver's scheme or the Baltas-Kosowski trend-strength), long-only and capped at full size, replacing the binary 200-day switch. Universe widened beyond the SPUS/HLAL duplicate: UMMA (developed ex-US), SPWO (world ex-US), SPRE (REITs) and SPSK (sukuk) are tradable on Alpaca now (checked 12 November). Gold ETFs are tradable too, but whether gold is acceptable is a judgement for you, as with crypto.
2. **Cross-sectional sleeve.** The scanner's 215 Shariah-listed names ranked by residual momentum (market and size removed), volatility-scaled, capped per name. This is where the "many assets" that the machine-learning papers need actually exist, so it is also where learned models (gradient-boosted trees on the existing features, judged by the gates) have a fair chance.
3. **Sizing layer.** Volatility targeting per strategy with a simple trailing estimator (downside volatility as a variant), never above 1x; partial adjustment toward the aim weights instead of the fixed band (Gârleanu-Pedersen); portfolio risk from a streaming shrunk covariance (River's `covariance.LedoitWolfCovariance` and `EwaCovariance`, River being already in the project at 0.26.1).
4. **Online learning layer.** A cost-aware Fixed-Share Hedge over experts: the trend speeds, the strategies, and cash. It learns which speed has been working and shifts weight toward it slowly, with the turnover cost in the loss, and its weights are logged. Library-first: the `universal-portfolios` package (OLPS: universal portfolio, exponential gradient, online Newton step) already runs these over a table of return relatives, and strategies plus cash can be its "assets". Its 0.4.17 release supports Python 3.11 to 3.14 but pulls in cvxopt, yfinance, plotly and seaborn and pins pandas below 3, so it has to be checked against this project's pins first; Hedge with Fixed Share is a short custom function if it does not fit, and it would be judged like any other model. River's `bandit` and drift detectors stay as they are. The learning is of the weights, not of a return forecast, which is where the evidence supports it.
5. **Evaluation.** Combinatorial purged cross-validation beside the walk-forward, every variant counted in the deflated Sharpe, a minimum of three months of shadow record before capital, and the Huang et al. test: compare against a rule that holds the positive-mean assets, not only against buy and hold.

Not recommended: more reinforcement learning on two ETFs (measured here and in the literature), mean-variance optimisation (DeMiguel et al.), meta-labeling or a return-forecasting online model on ETF trend (no supporting evidence found), and tuning any window to the last five years.

## What was built, and what it measured

**`halal-multi-trend`** (12 November 2026), beside `halal-etf-trend` and judged by the same five gates: the share of five trend windows (50 to 250 days) the close is above, scaled to 12% annual volatility from 60 days (capped at 1x), each weight moving a quarter of the way to its aim every day. Parameters were fixed from the literature and not tuned to the ETFs.

| On SPUS and HLAL, to November 2026 | CAGR | Volatility | Sharpe | Max drawdown | Turnover a year |
| --- | --- | --- | --- | --- | --- |
| holding both, from November 2020 | 17.4% | 18.0% | 0.98 | -25% | 0 |
| `halal-etf-trend` (200 days), from November 2020 | 15.0% | 13.2% | 1.12 | -20% | 4.7 |
| holding both, from March 2021 | 15.2% | 18.1% | 0.87 | -25% | 0 |
| `halal-multi-trend`, from March 2021 | 9.0% | 10.0% | 0.91 | -14% | 3.3 |

(The two rules start on different days because their warm-ups differ, 260 and 310 days.) On the 1926 US market proxy the new rule's Sharpe is 0.88 against 0.63 for the market, p 0.1%, drawdown -30% against -84%, and it passes all five gates. It lowers risk and does not raise return: on the ETFs it is behind the 200-day rule on return and Sharpe and ahead on drawdown (-14% against -20%). Whether it earns its place depends on a bear market none of these five years had, which is what the century proxy and the shadow record are for.

**Cost-aware trading.** The first version traded inside a 5% band and turned over 6.0 a year. Moving a fixed share of the way to the aim every day (Gârleanu and Pedersen) turns over 3.3 a year, with a Sharpe of 0.88 against 0.85 and a CAGR of 8.7% against 8.3% in the lab's comparison. Seven variants were run (volatility target, speeds, rate, band) with Sharpe ratios of 0.84 to 0.88 and a probability of overfitting of 0.93: the ranking among them is noise, so the rate was chosen for the turnover, which is mechanical and does not depend on the sample, and the others stay in the lab as trials.

**The wider universe did not help, on the history there is.** SPSK (sukuk) and SPRE (REITs) were tested as extra assets (daily-bar window 2021-12 to 2026-11, after warm-up). Held alone they made 0.9% and -2.5% a year with Sharpe 0.19 and -0.04, and trend on them made -0.2% and -4.4%. Blended with the equity sleeve, the result was a Sharpe of 0.2 to 0.57 against 0.64 for the equity sleeve alone. That is the interest-rate years 2022 to 2024 working against bonds and property, and four and a half years cannot say whether diversification pays over a cycle, so choosing the universe from this window in either direction would be choosing from the test. They are not in the strategy. UMMA and SPWO start in January 2022 and December 2023, which leaves too little to test.

**Residual momentum on the scanner's names.** `resmom_12_1` joined the scanner's scores (top 10 and top 20, so ten variants and ten trials): the stock's return less yesterday's rolling beta times the equal-weight market's, summed from 12 to 1 months ago over its standard deviation (Blitz, Huij and Martens, with the market as the only factor, since there are no size or value data). On the stored panel (to 11 November 2026, evaluation from June 2020) it has a Sharpe of 0.93 (top 10) and 0.90 (top 20) against 0.86 for plain 12-1 momentum, which is the direction the paper found, but below Clenow's 1.05 and 1.03 and below holding the whole listed set equally (1.19, a figure survivorship flatters). No variant passes "Sharpe above the benchmark". It runs in shadow with the other variants and trades nothing. Two side effects to know: the extra history it needs moved the evaluation's start from December 2019 to June 2020 and raised the trial count from 8 to 10, which lowers every variant's deflated Sharpe; and the scan's window had to grow from 300 to 403 sessions, or the new variants would have had no picks live.

**The online layer waits.** A Hedge or Fixed-Share layer reweights strategies. There are now two trend rules on the same two funds that move together, and the stock sleeve's variants all trail the benchmark, so there is not yet a second source of return to mix. It becomes worth building when one passes its gates.

## Open decisions

- **Stock sleeve.** Residual momentum is in shadow. A plain equal-weight hold of the listed names beat every picking variant, so is the sleeve worth its risk at all, or should it become a volatility-scaled hold of the whole list?
- **Universe.** Revisit SPSK, SPRE, UMMA and SPWO once they have a longer record, or sooner if you accept judging them on the literature alone. Gold stays out unless you decide otherwise.
- **Online layer.** `universal-portfolios` as is, or a Fixed-Share function with turnover in the loss, when there are experts worth mixing.
