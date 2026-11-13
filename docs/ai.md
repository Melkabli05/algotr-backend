# AI agents and drift detection

Three reinforcement learning agents (PPO, SAC, A2C) that choose portfolio weights, a UCB1 ensemble over them, and a drift monitor over every strategy and agent. Each agent holds what it chooses in a pretend portfolio. The UCB1 ensemble of the three can trade: it is one more candidate for the autopilot, admitted only like the rules are (below). Why they are built this way, and what they were measured to do: [ml-research.md](ml-research.md). Why each of them chose its weights, as SHAP values in the audit log: [xai.md](xai.md).

| Endpoint | What it does |
| --- | --- |
| `GET /systematic/agents` | whether the sidecar is set up, each agent's model, record and the weights it holds for tomorrow, the ensemble's weights and the latest judgement by the gates |
| `GET /systematic/agents/{algorithm}/explanations` | why the agent chose its weights on its newest decisions, and what its model relies on ([xai.md](xai.md)) |
| `GET /drift/status` | the rules, each stream's changes since its baseline, the size cut in force and the events recorded |
| `POST /drift/check` | scan every stream now and act on any new event |

| Setting (environment) | Default | What it does |
| --- | --- | --- |
| `AI_AGENTS` | `false` | train and follow the agents; the first run gives each a three-year record and took about 50 minutes, 45 of them SAC's, before SAC updated once every 4 steps (below) |
| `DRIFT_ACTION` | `act` | `act` cuts the size of a drifted strategy and asks for a retrain, `log` only records, `off` checks nothing |
| `DRIFT_THRESHOLD` | `0.1` | the smallest change that counts, as a Wasserstein distance in baseline standard deviations |
| `DRIFT_SIGNIFICANCE` | `0.001` | the Kolmogorov-Smirnov p-value a change must beat |
| `DRIFT_SCALE` | `0.5` | the multiplier on a drifted strategy's positions |
| `DRIFT_HOLD_DAYS` | `30` | how long the cut lasts after the latest adverse change |
| `DRIFT_BASELINE_DAYS` | `500` | the days at the start of a stream that define normal |

## The agents

**What they are.** PPO, SAC and A2C from Stable-Baselines3 2.9.2 with default hyperparameters (apart from PPO's `n_steps` 512, A2C's 16 and SAC's `learning_starts` 500), trained for 30,000 steps in the FinRL sidecar ([systematic.md](systematic.md)). Each sees, for SPUS and HLAL, FinRL's indicators (MACD, Bollinger position, RSI, CCI, DX, two moving averages), returns over 1, 5, 20 and 60 days and the 20-day volatility, standardised on its training window, and the weights its portfolio holds now. It answers with weights over the two ETFs and cash: a softmax, so long-only, never above the capital and with no leverage; cash earns nothing, as the account's does. The reward is the day's log return after 5 bps for each unit of turnover, the cost the ETF strategies use. The environment is `finrl_sidecar/weights_env.py`.

**Training.** Each month the agent is trained again on the last 756 trading days (three years), one seed. The sidecar keeps the model and the scaling it was given in `data/ai/models/`, and the audit log records each training with the fingerprint of the prices (`ai` entries). A drift event on an agent asks for the same, which happens at the next check (the first 5 seconds after start, then every 30 minutes, the desk looks for what is due).

**The record.** The first run backfills each agent: 36 months of retraining, each month traded unseen, which gives a record to judge and a baseline for the drift monitor. After that each day's return is added from the weights it chose the day before (the sidecar task `ai_shadow`), with the weights held between days drifting with prices and the turnover charged. State is `data/ai/state.json`. A day that was missed while the machine slept is caught up on the next check.

**The ensemble.** UCB1 ([Auer, Cesa-Bianchi and Fischer, 2002](https://homes.di.unimi.it/cesa-bianchi/Pubblicazioni/ml-02.pdf)): the score of an agent is its mean reward plus sqrt(2 ln n / n_j), a round is a month and the reward is the month's return squashed into (0, 1). The weights are a softmax of the scores (temperature 0.1) over a floor of 10% for each agent, so none is locked out; until every agent has a month they share equally. Each month's weights use only the months before it. With every agent's reward seen each round the counts are equal and the bonus is the same for all: it matters only for an agent with a shorter record.

**SAC's update rate.** SAC takes one gradient step for every environment step by default, and that is nearly all of its time. It now takes one every 4 steps (`train_freq=4`, `SAC_TRAIN_FREQ`): measured on one retrain window with the same data and seed, 55 s instead of 177 s, with an in-sample Sharpe of 4.19 against 4.29; every 8 steps saved little more (41 s) and cut the in-sample Sharpe to 2.82, every 2 steps took 113 s. The setting is part of the evaluation's configuration, so it counts as new trials in the deflated Sharpe, and the earlier evaluations were run without it.

**Judging.** `uv run poe ai-evaluate` (`python -m app.jobs ai-evaluate`, with the headline mood) retrains every agent each month from May 2023 for three seeds (about 80 minutes: nine jobs on the machine's cores, a SAC seed taking 76 minutes of it), builds the ensemble from each seed's agents, and judges each by the five gates of the rule lab against the 200-day trend rule. The result is in the audit log (`ai-evaluation`), counts every trial toward the deflated Sharpe together with the FinRL lab's, and is what the dashboard shows. Nothing becomes eligible for trading by being built.

## From research to orders

**The ensemble as a strategy.** The autopilot's automatic selection judges the ensemble once a day beside the rules: it passes when the latest `ai-evaluation` in the audit log found the UCB1 ensemble group eligible by all five gates, and like a rule it starts after passing on 10 days in a row and stops after failing on 10 (`ensemble_active` and `ensemble_streak` in the overview). Admitted, it is the strategy `ai-ensemble`: its weights over SPUS and HLAL for tomorrow are each agent's target mixed by the ensemble's weights, its share of the allocation comes from equal risk contribution over its record and the rules' returns, and it has its own book, so the kill switch, the drift monitor's `ai-ensemble` cut and the fill costs apply to it as to any strategy. Manual strategy selection never includes it. Today it fails the gates, so it stays out; the path is live for when an evaluation passes.

**Purged and embargoed walk-forward.** Each month's training window stops 1 day (the purge: the reward is the next day's return, which would reach into the month traded) plus 5 days (the embargo: the serially correlated days just before it) before the month it trades, as López de Prado (2018, chapter 7) prescribes for financial series. The live monthly retrain needs neither, since what it trades has not happened. The gap is part of the evaluation's configuration, so an evaluation with it counts as new trials.

**Headline sentiment in the observation.** The agents see two more features, the same for both funds: the mean FinBERT score of the day's headlines on the Shariah-listed stocks (-1 to +1) and their mean rule risk (1 to 5, scaled to 0 to 1). A headline counts for the session it can first inform: before the 16:00 New York close its own day, after it the next, and a weekend's go to Monday, so nothing after a session's close is in its row. A day without headlines reads as neutral. The two columns bypass FinRL's `FeatureEngineer`, which fills gaps backwards and would put a later day's mood into an earlier one. History comes from `uv run poe news-backfill`, which fetches the listed stocks' headlines a month at a time back to May 2020 (resumable, `data/scanner/news-backfill.json`) and scores them; before that, or where the backfill has not reached, the feature is neutral. Models trained before the feature keep working (their scaling names their own columns) until their next retrain. FinBERT's next-day predictive power on its own was nil (sentiment-research.md); the agents and the gates decide whether it adds anything.

## Drift monitor

**What is watched.** The daily returns of the four strategies' rules (from the same prices as the research) and of every agent and the ensemble. A stream needs its baseline (the first 500 days) plus 60 days before it is checked.

**How a change is found.** Returns are divided by the baseline's deviation, because ADWIN's cut rule assumes values in a range of about 1 and never fires on raw returns. Two River `ADWIN` detectors (`delta` 0.002, checked every day) run over the whole series: one on the returns for a change of the mean, one on their absolute values for a change of volatility. When one fires, the last 60 days are held against the baseline for up to 60 days: the change counts once their Wasserstein distance from the baseline is at least `DRIFT_THRESHOLD` and a Kolmogorov-Smirnov test (on the magnitudes for a volatility change) puts p under `DRIFT_SIGNIFICANCE`. The distance alone cannot decide it: over 60 days noise is usually above 0.1.

**What it does.** Every change is shown. One is adverse when the mean fell or the volatility rose. An adverse change on the day the monitor first ran or later (earlier ones are history): is written to the audit log (`drift`), cuts the strategy's positions to `DRIFT_SCALE` for `DRIFT_HOLD_DAYS` days after the latest one (the autopilot's plan reads the multiplier; the rest stays in cash), and asks for a retrain. Only an agent can be retrained; the ensemble asks for all three; for a rule the answer is "not applicable". ADWIN forgets after a detection, so a long shift is one event.

**When.** Once a day at 00:01 UTC, and whoever asks first on a day the machine slept through it (the autopilot's plan asks for the multiplier of each strategy it plans). If the data cannot be read the last answer stands.

## Where the code is

| Module | What it holds |
| --- | --- |
| `finrl_sidecar/weights_env.py`, `runner.py` | the weights environment and the tasks `ai_walk_forward`, `ai_train`, `ai_shadow` |
| `features/systematic/agents.py` | `AgentDesk`: training, the record, following the days, retrain requests, the schedule |
| `features/systematic/ensemble.py` | UCB1 scores, weights and the replay over months |
| `features/systematic/agent_lab.py` | the walk-forward evaluation and its judgement (`uv run poe ai-evaluate`) |
| `features/drift/` | `DriftMonitor`, the scan, the router and models |
| `core/drift.py` | the `ReturnSource` and `SizingGuard` protocols that connect them (`main.py` wires them) |
| `features/systematic/explanations.py` | the SHAP explanations of each decision as audit entries, and reading them back ([xai.md](xai.md)) |
| `frontend/src/views/Research.tsx`, `Signals.tsx` | the agents and their judgement (Research), the drift banner and decisions (Signals and XAI) |
