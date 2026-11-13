# Explainable AI: research

The seventh feature list (6.1, SHAP explanations, and 6.2, their storage and audit trail) read against what the system's models are, the SHAP library, the evidence on how far Shapley explanations can be trusted, and the regulatory text the list leans on. Figures were measured on 4 November 2026 on seven agents trained for this research in a scratch folder (PPO with five seeds, A2C and SAC with one, 30,000 steps each, on the 756 days before the last 60 trading days of SPUS and HLAL), each explained on those last 60 days (61 decisions). Earlier work: [ml-research.md](ml-research.md), [sentiment-research.md](sentiment-research.md).

## Claims in the list that the evidence does not support

| Claim | What the evidence says |
| --- | --- |
| Every AI-generated trading signal gets SHAP values | The only models that make a signal are the PPO, SAC and A2C weight agents and their UCB1 blend, and they are shadow only: no order follows from them, and all of them fail the gates ([ml-research.md](ml-research.md)). Everything the autopilot trades comes from rules, and a rule's reason is the rule itself applied to the prices (exact, with nothing to approximate). FinBERT's sentiment score is a model output but not a trading signal |
| It tells which technical indicator, sentiment score or order book metric drove the decision | The agents are shown indicators, returns, volatility and the weights they hold, and nothing else. No model takes a sentiment score (60 days of them exist) or an order book metric (crypto only, three days kept) as an input, so neither can appear in an attribution. The explanations name indicators and the weights held |
| Global and local explanations are both computed and stored | Local explanations are stored for every decision. A model's global one, the mean size of each indicator's push over 60 training days, is stored once per model. The average over the agent's own decisions is derived from the stored local ones when asked and is not stored a second time |
| They go to an `xai_explanations` table partitioned weekly | PostgreSQL is not in the stack. They are `xai` entries in the hash-chained audit log, one for each run that explained something. A year of daily explanations for three agents is 2.1 MB (5.7 KB a day); reading the newest 30 decisions of an agent from that takes 0.03 s and verifying the whole chain 0.06 s, so partitions would buy nothing |
| The trail is complete and immutable | The chain is tamper-evident, not immutable: it shows an entry edited, removed or reordered anywhere but the end, and it cannot see entries cut off the end or a file rewritten with every hash recomputed ([audit.py](../src/app/core/audit.py)). It is complete only for decisions made after the feature is on: the 36-month record the agents were given was made by monthly models that were not kept, so those decisions cannot be explained. An explanation can be re-run only while its model is in force, because a retrain overwrites the model file; each entry therefore holds the observation, the background it was set against, the model's version and a fingerprint of the prices |
| Explanations are required for regulatory compliance | In the texts I read, nothing says so. RTS 6 ([Regulation 2017/589](https://www.legislation.gov.uk/eur/2017/589/contents)) asks investment firms for testing and deployment controls (Articles 5 to 8), an annual self-assessment and validation (Article 9, [text](https://www.legislation.gov.uk/eur/2017/589/article/9)), a kill functionality (Article 12) and, for high-frequency techniques, order records kept for five years (Article 28, [text](https://www.legislation.gov.uk/eur/2017/589/article/28)); neither of those two articles asks for a model's decisions to be explained. The EU AI Act's high-risk list ([Annex III](https://artificialintelligenceact.eu/annex/3/)) names credit scoring (5(b)) and does not name algorithmic trading. Neither applies to a personal paper account as built |
| It answers "why did the system sell AAPL on March 15?" | The system does not trade AAPL with a model. The question that can be answered is "why did PPO cut HLAL from 38% to 25% after the close of March 15", for a day after the feature was on, and the answer is what moved the network's output, not what moved the market ([Kumar et al., 2020](https://proceedings.mlr.press/v119/kumar20e.html) argue that Shapley-value explanations have properties that do not match what people expect of a feature's importance, and advise caution outside narrow settings) |

## The library

[shap](https://pypi.org/project/shap/) 0.54.1: MIT licence, Python 3.12 or newer, numpy 2 or newer, and apart from packages the sidecar already has it adds only `slicer`. It goes in the FinRL sidecar, next to torch and the agents' models.

| Option | Verdict |
| --- | --- |
| `KernelExplainer` | Samples coalitions, and since 0.47.0 its default `l1_reg` is `num_features(10)`, which keeps ten features and drops the rest ([docs](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html)). A default that silently zeroes most of 27 inputs is not what an audit record should hold |
| `DeepExplainer`, `GradientExplainer` | Approximations of the network. Papers that explain a portfolio agent use them ([arXiv 2509.14385](https://arxiv.org/abs/2509.14385)); the network here is small enough not to need one |
| `PermutationExplainer` over the 27 inputs | Agrees with the choice below: summed back to the 13 groups it gives cosine 1.00 with the exact values on every decision tested, with a largest single gap of 3.8 points of a weight and 0.1 s a decision at 40 permutations. It is approximate and random, and 75% of the coalitions it draws from (if drawn uniformly) split the three held weights, which are not then a portfolio |
| `ExactExplainer` with 13 players | Chosen. The docs say it works well for models with fewer than about 15 features that vary from the background ([docs](https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html)); a masker is a documented function of the mask and the sample that may return several rows, which SHAP averages ([custom masker](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/maskers/custom.html)). The custom code is that masker, ten lines, which makes each indicator one player across both ETFs and the weights held one player |

Why those players: the two ETFs' versions of an indicator correlate at 0.90 to 0.98 (DX 0.90, everything else 0.95 or more), so splitting an indicator's credit between SPUS and HLAL is arbitrary; and the held weights add up to one, so a mix of the held weights of two days is not something the agent has met.

## How far the explanations can be trusted

| Check | Result |
| --- | --- |
| Equal to a brute-force enumeration of every coalition (an independent implementation) | the largest difference over the 39 values of a decision was 6.6e-9; the repository's unit tests repeat the comparison on random networks |
| The base and the pushes add up to the agent's weights | to 1.2e-7 (float32) |
| Cost | 0.3 to 0.5 s a decision on four threads, 0.7 s on one (the sidecar runs with one), with 64 background days |
| Changing the background: another draw, 16 or 256 days instead of 64, only the last 252 days | the local values keep a cosine of 0.97 to 1.00 with the first draw and the ranking of indicators a rank correlation of 0.98 to 0.99 |
| Deleting the top-attributed indicators (replacing them by the background) against deleting the same number at random | the weights move 2 to 3 times as much: with one indicator removed, PPO (seed 1) 0.67 against 0.22, A2C 0.35 against 0.11, SAC 0.90 against 0.39 (the sum of absolute changes of the three weights) |
| Randomising the network (same architecture, new weights) | the attribution has a cosine of -0.19 (-0.56 to 0.21) with the trained agent's, so it depends on what was learned ([Adebayo et al.](https://arxiv.org/abs/1810.03292) propose this check) |
| The same decision across five PPO seeds | over the ten pairs of seeds the local values have a cosine of 0.16 (-0.18 to 0.39) and the ranking of indicators a rank correlation of 0.34 (-0.07 to 0.72); the seeds' target weights themselves differ by 1.10 (sum of absolute differences, at most 2) |

The last row is the finding that matters. The method is faithful to the network it explains and reproducible given the model, the observation and the background. The agents are not reproducible: another seed explains the same day in another way because it is a different policy that chose different weights. An explanation says why this model chose this, which is what an audit of one model needs, and it is no evidence that the choice was right.

## What the agents respond to

Over the 61 decisions the indicators that moved the weights most were short: the 1-day return was in the top three of three PPO seeds, the 5-day return of three, DX of four; A2C responded most to CCI, the 20-day return and the Bollinger position, SAC to the price against its 30-day average, CCI and the 5-day return. The weights held accounted for 3% to 13% of the attributed push, so the agents are not mostly inertial. They chase one to five days of price and a noisy oscillator, and they changed their target by 0.29 to 1.03 a day (sum of absolute changes), which is consistent with the turnover wave 5 measured and with none of them passing the gates. This is what the explanations are for: model debugging, not a reason to trust an agent.

## Decisions

- **Exact values over 13 players** (above), with a fixed seed for the background, so the same observation, model and background always give the same numbers.
- **What is explained is the weights the environment applies**: the softmax of four times the agent's action, so a push is in points of weight, and the three pushes of an indicator add up to zero.
- **Where the baseline comes from**: 64 days drawn from the agent's training window with the weights it held when it rolled through them from cash, so a held-weight is as plausible as a market value. The explanation is "against an average training day".
- **Each decision is explained with the model that made it.** The desk queues a decision with the model's version, and a decision whose model was replaced before it was explained is recorded as unexplained.
- **The audit log, not a table.** One entry for each run, written before the queue is cleared, so a crash can repeat a decision and never lose one; a repeat shows as two entries with the same decision date.
- **No setting.** Explaining runs whenever `AI_AGENTS` does.

## Not built

The ensemble's own explanation (its weights are a blend of the agents', so its pushes would be the same blend of theirs, but the blend changes by month and was left out), token-level SHAP of FinBERT's headline scores (not a trading signal), any explanation of a rule-based strategy (its own definition is the exact one), a store of every model version so old decisions could be re-explained, and a weekly partition or a table.

## Not verified live

The measurements above are from agents trained in a scratch folder for this research. The live agents' own first explanations, and what they cost on the sealed one-thread sidecar, are in [xai.md](xai.md#first-live-run).
