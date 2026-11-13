# AI and machine learning features: research

The fifth feature list (the PPO, SAC and A2C agents, UCB1 ensemble weighting and River drift detection) read against the libraries, the literature and what the platform has. Facts come from the libraries' source and documentation, the papers cited, and measurements made on 3 and 4 November 2026 on this account's data. Earlier work: [quant-research.md](quant-research.md) (why the strategies are fixed rules), [systematic.md](systematic.md) (the FinRL lab, which already judged these algorithms once), [orderbook-research.md](orderbook-research.md).

## Claims in the list that the evidence does not support

| Claim | What the evidence says |
| --- | --- |
| PPO is strong in trending markets, SAC beats it in mean-reverting ones | The PPO half comes from one backtest: [Yang, Liu, Zhong and Walid](https://arxiv.org/abs/2511.12120) (30 Dow stocks, January 2016 to May 2020) found PPO had the highest annual return (15.0%) and A2C the lowest drawdown, and says "one agent performs well in a bullish trend but acts bad in a bearish trend". Their paper has no SAC, and I found no study that splits PPO and SAC by regime. In the 881 days measured here PPO beat SAC in every seed, and the market was above its 200-day average on 93% of them, which leaves 59 days (about three months) for the case the claim is about: too few to test it, and what they show is no difference (below). One out-of-sample [comparison of PPO, QR-DDPG and SAC](https://www.researchgate.net/publication/398486187_Deep_Reinforcement_Learning_for_Risk-Aware_Portfolio_Optimization_A_Comparative_Study_of_PPO_QR-DDPG_DDPG_and_SAC_under_Market_Uncertainty) found the three statistically equivalent |
| Agents are trained monthly on three years of rolling data | Possible. A retrain of 30,000 steps takes about 5 s for PPO, 4 s for A2C and about a minute for SAC on one core; 43 months of retraining took 7.5 minutes for a PPO seed, 5 for an A2C seed and 76 for a SAC seed (the three of each ran side by side on a busy machine), so the nine jobs of the evaluation take about 80 minutes. The data is the limit: SPUS and HLAL have 6.9 years of aligned history, and with UMMA and SPSK only 4.8, so the agents get the first two (plus cash), as the autopilot's ETF strategy has |
| The observation holds OHLCV, indicators, sentiment, order book features and portfolio state | OHLCV, FinRL's own indicators and the weights the portfolio holds are used. Sentiment is not: the news store starts 60 days back for a name and keeps headlines with a typed event and, since 3 November, FinBERT's score ([sentiment.md](sentiment.md)), so there are 60 days of scores and a model trained on three years cannot have them. Order book features are not either: the book is crypto only, three days of it are kept, and the agents trade ETFs |
| The agent outputs target portfolio weights | Yes, but none of FinRL's environments could: `StockPortfolioEnv` rewards the portfolio *value* instead of a return and never applies its transaction cost (the argument is stored and unused), `PortfolioOptimizationEnv` speaks the old `gym` API for FinRL's own agent, and `StockTradingEnv` outputs share trades. A small Gymnasium environment is custom code, with the reasons in `finrl_sidecar/weights_env.py` |
| SAC's entropy regularisation keeps it flexible | Stable-Baselines3's SAC learns the coefficient (`ent_coef="auto"`), which is what is used. Whether that helps is measured below |
| UCB1 "adapts automatically to regime changes" | UCB1 ([Auer, Cesa-Bianchi and Fischer](https://homes.di.unimi.it/cesa-bianchi/Pubblicazioni/ml-02.pdf), 2002) assumes rewards in [0, 1] drawn independently from a fixed law for each arm, and its guarantee is for the arm it plays. Here every agent's reward is seen each round, so every count is equal and the exploration bonus is the same for all: the score is the mean reward and the ranking is by the average record. The bonus differs only for an agent that is new or was reset. A rule built for full information (Hedge, exponential weights) is the textbook choice; UCB1 is built as asked, with the rewards squashed into (0, 1) |
| Drift is detected with ADWIN | ADWIN detects a change in the mean of a stream whose values lie in a range of about 1. River's cut rule (`rust_src/adwin.rs`) is \|mean0 − mean1\| > sqrt(2 · variance · m · ln(2 ln n / δ)) + (2/3) · ln(2 ln n / δ) · m, whose second term assumes a range of 1: on raw daily returns (about 0.01) it never fires, not even for a fall from a Sharpe of +0.8 to −3.2 (0 of 200 runs). On returns divided by the baseline's deviation it finds a doubling of volatility in 100% of runs, after 42 days at the median, with at most 1% false alarms in five years; a realistic fall of the mean (Sharpe +0.8 to −0.8) is not found within a year by any transform |
| Drift is a Wasserstein distance above 0.1 | The number is Evidently's default for large samples (a reference of more than 1,000 rows), and that library uses the Kolmogorov-Smirnov test for smaller ones. Measured in baseline deviations, with no change at all a window of 20 days is over 0.1 every time, 60 days 94% of the time, 120 days 67%, 250 days 33%, and fat tails make it worse; used alone it would halve every position all the time. It is kept as the smallest change that matters, and a change must also be raised by ADWIN and confirmed by a KS test |
| Drift detection prevents models silently degrading when regimes change | In part. A change in volatility is found within weeks; a loss of edge (a mean that falls by a realistic amount) is not found within a year at the daily frequency, for the same reason a Sharpe ratio takes years to establish |
| A drift event is answered by cutting the size by 50% and retraining | The strategies are fixed rules, so only the agents can be retrained; for a strategy the cut is the whole answer. On the real strategies the cut does not help: see below |

## Libraries

| Need | Choice | Why |
| --- | --- | --- |
| The three agents | Stable-Baselines3 2.9.2 (`PPO`, `SAC`, `A2C`) in the existing FinRL sidecar | FinRL's agent wrappers add nothing for a custom environment, and the sidecar already pins the stack. Default hyperparameters except `n_steps` (PPO 512, A2C 16) and SAC's `learning_starts` (500) |
| Features | FinRL's `FeatureEngineer` (stockstats) | The indicators FinRL ships (MACD, Bollinger bands, RSI, CCI, DX and two moving averages) plus returns over 1, 5, 20 and 60 days and the 20-day volatility, standardised on the training window |
| Weights environment | custom, about 90 lines | None of FinRL's three fits (above); checked against an independent recomputation of every return from the reported weights, which breaks when any of the cost, the day alignment, the weight drift, cash or the sum of the weights is changed |
| Drift detector | River 0.26.1 `drift.ADWIN` (BSD-3, Python 3.11 to 3.15, wheels for macOS, Linux and Windows, adds `narwhals`) | What was asked for. Its `width` is a float although documented as an int, and it forgets everything after a detection |
| Distances and tests | `scipy.stats.wasserstein_distance`, `ks_2samp` | Already a dependency |
| UCB1 | 3 lines of arithmetic | River's `bandit.UCB(delta=1)` is UCB1 but pulls one arm and keeps its scores private; the platform needs a weight for each agent |

## What the agents did

Walk-forward from 1 May 2023 to 2 November 2026, 881 trading days: each month every agent was trained again on the 756 days before it, for three seeds, and traded the month blind, with 5 bps charged on each unit of turnover. Nothing was tuned on these results (30,000 steps and the library's defaults, fixed beforehand). Over the same days the 200-day trend rule had a Sharpe ratio of 1.26 and holding SPUS and HLAL equally 1.21.

| | Sharpe by seed (out of sample) | CAGR | Max drawdown | In-sample Sharpe | Mean cash | Weights change a day | Gates passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 0.65, 0.81, 0.73 | 8.7% to 10.8% | -20% to -24% | 1.08 to 1.21 | 24% | 39% | 2 of 5 |
| SAC | 0.35, 0.11, -0.06 | -1.5% to 3.5% | -18% to -32% | 0.03 to 0.09 | 31% | 92% | 1 of 5 |
| A2C | 0.78, 0.49, 0.76 | 5.7% to 10.4% | -14% to -25% | 0.61 to 0.69 | 34% | 31% | 2 of 5 |
| UCB1 ensemble | 0.64, 0.54, 0.57 | 6.3% to 7.5% | -20% to -22% | 0.60 to 0.64 | | | 2 of 5 |

No agent beats the trend rule or holding the two ETFs in any seed, and none passes the gates that matter: the chance that the best seed of PPO is luck against the trend rule is 98% (A2C 97%, SAC 59%, the ensemble 60%), and after counting 46 trials (the 34 of the FinRL lab and these 12) the deflated Sharpe ratio is 19% for PPO, 2% for SAC, 20% for A2C and 12% for the ensemble, where 95% is needed. The two gates they pass say little: the out-of-sample Sharpe is at least half the in-sample one (SAC's is near zero in sample), and the months are mostly profitable.

They trade far too much. The absolute changes of their target weights add up to 31% (A2C), 39% (PPO) and 92% (SAC) of the portfolio on an average day, which at 5 bps costs about 3.9%, 4.9% and 11.5% a year; the trend rule trades a few times a year. SAC's learned entropy shows up here as weights that move almost completely every day, and at 30,000 steps it learned little (its in-sample Sharpe is 0.03 to 0.09). That is a result about these settings, not about the algorithm: a penalty for turnover or more steps might change it, and nothing here was tuned because tuning on the walk-forward would spend the test.

**Regimes.** The market was above its 200-day average on 822 of the 881 days. On the 59 days it was below it, the three seeds' mean Sharpe was 0.75 for SAC, 0.80 for PPO and 0.82 for A2C, with SAC's seeds ranging from 0.01 to 1.28: no difference, and too few days for one. The claim that SAC does better when the market does not trend is not supported, and not testable on this history either.

**The ensemble.** UCB1 weighting is not better than averaging: 0.64, 0.54 and 0.57 by seed against 0.65, 0.50 and 0.53 for the plain average of the same three agents, and below the best single agent of each seed (0.78, 0.81, 0.76). The agents are close to one another (the daily returns of PPO and A2C correlate at 0.84 to 0.90, PPO and SAC at 0.76 to 0.83), so there is little to diversify, and rewards squashed around one half give weights that stay near a third each (PPO 33% to 44%, SAC 26% to 30%, A2C 30% to 38% on average over the months). A sharper softmax (temperature 0.02) scored 0.59, 0.62 and 0.63; it was looked at once and not adopted, because choosing it from these results would be choosing it from the test.

## What the monitor does on the real strategies

The settings were fixed before this was run: a baseline of the first 500 days, ADWIN with `delta` 0.002 checked at every day, confirmation over 60 days (distance at least 0.1, KS p under 0.001, on the magnitudes for a volatility change), an adverse change cutting the size to 50% for 30 days.

| Strategy | Days monitored | Changes found | Adverse | Days under the cut | Sharpe without and with the cut | Max drawdown without and with |
| --- | --- | --- | --- | --- | --- | --- |
| `btc-trend-200` | 1,412 (3.9 years) | 6 | 4 | 8% | 0.96 and 0.85 | -32.8% and -32.9% |
| `crypto-donchian` | 1,252 (3.4 years) | 7 | 4 | 9% | 0.56 and 0.64 | -18.3% and -16.8% |
| `halal-etf-trend` | 1,007 (4.0 years) | 2 | 0 | 0% | 1.23 and 1.23 | -10.3% and -10.3% |
| `halal-etf-rotation` | 563 (2.2 years) | 2 | 1 | 4% | 0.69 and 0.67 | -14.2% and -14.2% |

Changes are rare (at most 1.2 adverse a year) and the strategy the autopilot actually runs by default never triggered one. The cut does not reduce risk-adjusted loss in any consistent way: it lowered the Sharpe ratio of two strategies, raised one and left the maximum drawdown where it was in three of four. A volatility alarm arrives when volatility is already high, and for a trend rule that is often when it is about to be paid. It is built as asked, and `DRIFT_ACTION=log` keeps the events without cutting anything.

## Decisions

**Agents in shadow.** The agents hold their weights in a pretend portfolio and never place an order: the platform's rule is that nothing changes what the autopilot trades until it passes every gate, and the evaluation above says none does. The first run of the desk gives each agent a three-year record (monthly retraining, each month unseen) so that there is something to judge and to watch; the live days then continue it.

**Judged like everything else.** Each agent runs over the same walk-forward months for three seeds and goes through the five gates of the rule lab ([systematic.md](systematic.md)): stable across seeds, beats the trend rule after the search, real skill after counting every trial (the deflated Sharpe, which also counts the 34 FinRL trials already recorded), holds up out of sample, consistent through time. The UCB1 ensemble is judged the same way, built from the agents of each seed.

**UCB1 over monthly rounds.** A round is a month: it is the retraining interval, so each retrained model's month is one position that closes. The reward is the month's return squashed into (0, 1) by tanh against the pooled spread of all monthly returns seen so far, and the weights are a softmax of the UCB1 scores (temperature 0.1) over a floor of 10% for each agent. Until every agent has a month they share equally. The replay uses only the months before the one being weighted; a test changes a month and checks nothing before it moves.

**Off by default.** `AI_AGENTS` is false: the first run uses the CPU for about 50 minutes and writes `data/ai/`. The drift monitor is on (`act`) as asked, and its state is in `data/drift.json`; events before the day it first ran are history, not events.

## Not built

Sentiment and order book inputs (no data to train on), a regime switch between PPO and SAC (no evidence for one), trading by any agent or by the ensemble, a drift action for the autopilot's scanner sleeve (it is not a strategy with a return series), and Hedge in place of UCB1.
