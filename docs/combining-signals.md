# Combining strategies, signals, backtests, research and news into automatic trading

Whether the pieces the platform already has can be joined into one automatic decision, how the research says to do it, and what this account's own measurements say about each ingredient. Papers are cited from abstracts and summaries unless a line says otherwise. Earlier work: [strategy-research.md](strategy-research.md), [sentiment.md](sentiment.md), [ai.md](ai.md), [systematic.md](systematic.md).

## What is already connected

```
prices ─► strategies (weights) ─► gates (backtest, lab, proxy) ─► selection with hysteresis
                                         │                                  │
              lifecycle stage (shadow, canary 10%, live by approval)        ▼
              drift scale, drawdown cut, kill switch ─────────────► plan ─► risk-parity split ─► orders ─► fills
                                                                      ▲                                    │
scanner picks ─► news and filing vetoes ─► sleeve ───────────────────┘                                    ▼
agents, Kronos ─► gates (they fail, so they trade nothing)                          fill costs, outcomes, drift monitor
```

- **Strategies and signals** produce target weights, one set per strategy. A weight is the common contract: the autopilot never needs to know how a strategy decided.
- **Backtests and research** are the gates. Every strategy is judged by five checks, with the rule lab counting every variant tried, the 1926 proxy for the significance tests, and the lifecycle's own gates before a strategy trades (`research`, `validated`, `shadow`, `canary` at 10% of its capital, `live` by approval).
- **The autopilot** picks the strategies that pass on ten days in a row, sizes each by the drift monitor and the drawdown stop, splits capital by equal risk contribution, and places the orders.
- **News** reaches the decision in two places only. A serious negative headline or filing vetoes a new buy of a scanner name (`watch.vetoed`), and the day's mean sentiment is a feature the learned agents see. Nothing in the trend strategies reads news.

So combining is not a missing capability: strategies, research, gating and execution are one chain. What is missing is narrower, and listed at the end.

## What the evidence says about each ingredient

**News sentiment as a direction signal does not work here, and the literature agrees.** The app scores its own signals against the next trading day (`GET /systematic/outcomes`). FinBERT headlines scoring 0.2 or more either way were right 50.2% of the time over 365 days (33,131 signals), 48.5% over 90 days and 49.3% over 30. By confidence: 51.0% for headlines at 0.8 to 0.9 and 49.7% at 0.9 to 1.0, so confidence says nothing either. A pre-registered test of [FinBERT on US stocks](https://github.com/NakulPillai2407/news-sentiment-research) found sentiment tied to the return of the article's own day, but on the next day, the first a trade can honestly use, the effect was about the size of the trading cost (held-out net Sharpe -0.34 against 0.94 for equal weight; a GitHub study, not peer reviewed). Other backtests find the same, or find that costs of 4 to 5 bps erase the gain ([FinDPO](https://arxiv.org/pdf/2507.18417)). [Tetlock (2007)](https://www.columbia.edu/~pt2238/papers/Tetlock_Media_Sentiment_JF.pdf) found media pessimism followed by a reversal, which means the sign of any effect is not even settled.

**Language models add a bias a backtest cannot see.** A model trained after the period it is tested on has read the outcome. [Glasserman and Lin](https://arxiv.org/html/2309.17322) show GPT sentiment strategies do better in-sample on anonymised headlines, and that any backtest inside a model's training window is tainted; [Gao, Jiang and Yan](https://arxiv.org/pdf/2512.23847) put the contamination at about 37% of the effect in the best-known ChatGPT result. FinBERT's weights predate the headlines used here, so they are not exposed to this; a newer general model would be.

**News is worth having as a risk overlay, with thin evidence.** Avoiding or shrinking positions into serious negative events (guidance cuts, legal, offerings, halts) is standard practice and cheap, but I found no peer-reviewed study that isolates what such an overlay adds to a systematic equity strategy. It is the right job for news, and it has to be measured here before it is widened.

**Combining signals: keep it simple.** The [forecast combination puzzle](https://econweb.ucsd.edu/~gelliott/Combination.pdf) is that estimated optimal weights often lose to a plain average, because estimating the weights costs more than the optimum gains (Smith and Wallis). The fundamental law (information ratio about the information coefficient times the square root of breadth, times a transfer coefficient for how well a view becomes a position) says where gains come from: independent bets and keeping views intact through costs and constraints, not cleverer weights. Carver's scheme is the practical form: scale each forecast to a common size, average with weights from correlations, cap.

**Backtests overstate, so the live record is part of the evidence.** [McLean and Pontiff](https://ivey.uwo.ca/media/3775549/pontiff.pdf) find the average published anomaly's return falls about 35% after publication, and critics note part of that is data mining. The standard governance answer is staged: a shadow period of 30 to 60 days of logged, not traded, decisions, a small canary, and a human sign-off before full size. That is the lifecycle already in the platform.

**Architectures.** [FinRL-X](https://arxiv.org/abs/2603.21330) (March 2026) proposes the same chain as above: a weight vector as the contract between selection, allocation, timing and a portfolio risk overlay, with rule-based and learned components pluggable without changing execution. It reports no numbers in the abstract. A January 2026 preprint ([arXiv 2601.19504](https://arxiv.org/abs/2601.19504)) combines trend, mean reversion, FinBERT, XGBoost and a regime filter and reports a 135% return over 24 months without a Sharpe, costs or drawdown, which is not evidence of anything.

## How to combine them here

1. **Every signal is a weight-producing strategy that has passed the gates.** Trend, the stock sleeve, and later Kronos or the agents join the same way: nothing enters the combination until it passes its own five checks and its shadow record. A signal that does not pass is shown and logged, not weighted.
2. **Combine by risk and by simple weights.** Equal risk contribution across the strategies that pass, as now; the online Hedge/Fixed-Share layer, when there are two independent strategies to mix, only moves those weights slowly and with turnover in its loss.
3. **News is an overlay on the combined plan, not a forecast.** A serious negative event on a held or about-to-be-bought name blocks the buy, and may later cut the size; a day of high market-wide news risk may cut gross exposure. Each rule is first run in shadow and its worth measured: did the vetoed names do worse than the ones let through over the next 1, 5 and 21 days? News as a feature for the learned models stays inside their gates.
4. **Test the portfolio as it will trade.** Strategies are backtested one at a time today. The combined portfolio (selection, risk-parity split, overlays, costs) should be backtested and judged as one return stream, so the gates see what the account holds.
5. **Close the loop.** Fill costs, signal outcomes and the drift monitor already measure what happened; they should feed the selection, the Hedge weights and retirement, so a signal that stops working loses weight without a person noticing first.

## What is missing

| Gap | Why it matters |
| --- | --- |
| A backtest of the combined portfolio | The gates judge each strategy; nothing judges the risk-parity mix with its overlays and costs |
| A measured value for the news veto | Done (`POST /scanner/vetoes`): 458 events since 2018 show no measurable protection, and none of the size that would justify widening it; see below |
| One scoreboard of every signal | Hit rate, gates, shadow days and cost are in four places (outcomes, the lab, the lifecycle, the agents tab) |
| The online weighting layer | Waits for a second independent strategy that passes |
| A market-wide news risk overlay | Untested; the sign of any sentiment effect is disputed, so it starts in shadow |

## What the veto study found

The stored headlines go back to 2020, and the veto is a rule on stored event types, so it could be tested on the whole history instead of waiting for a live log (`scanner/vetoes.py`, [scanner.md](scanner.md)). 458 veto events on 128 names: the vetoed names did about as well as the names the veto let through (excess +0.01% at one session, -0.03% at five, -0.44% at 21, none significant, vetoed below peers 49% to 53% of the time). The intervals exclude large protection (more than about 0.4% over a week, 1.3% over a month) and say nothing about a small one. So news earns the role it has, blocking a buy at a small cost, and no wider one: cutting size on news, or acting on market-wide news risk, has nothing under it yet.

## Order of work

1. ~~The veto's outcome measurement.~~ Done, above.
2. The combined-portfolio backtest, judged by the same gates.
3. The signal scoreboard.
4. The online layer, once step 2 shows two strategies worth mixing.
