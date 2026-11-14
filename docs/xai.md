# Explaining the agents

Each decision of the PPO, SAC and A2C agents ([ai.md](ai.md)) is explained with SHAP values: how far each indicator moved each of the weights it chose from what the same agent gives on an average training day. The explanations go to the audit log one run at a time and are read back on the AI agents tab. What the list asked for, what was measured about how far the values can be trusted, and what is not built: [xai-research.md](xai-research.md).

| Endpoint | What it does |
| --- | --- |
| `GET /systematic/agents/{algorithm}/explanations?limit=30` | the agent's newest explained decisions (at most 365), what its latest model relies on, the typical push over the decisions listed and how many decisions were lost |

There is no setting: explaining runs whenever `AI_AGENTS` does. `uv run poe finrl-setup` installs [shap](https://pypi.org/project/shap/) 0.54.1 into the sidecar with the rest, and a sidecar set up before this needs that command once more.

## What is explained

The value explained is the weights the environment applies, the softmax of four times the agent's action, so a push is in points of weight and the three pushes of an indicator add up to zero. There are 13 players: the 12 indicators (the 1, 5, 20 and 60-day returns, the 20-day volatility, MACD, the Bollinger position, RSI, CCI, DX, and the price against its 30 and 60-day averages), each with its values for both ETFs, and the weights held, as one.

The baseline is an average training day: 64 days drawn with a fixed seed from the days the agent was trained on, each with the weights the agent held when it was rolled through them from cash. A coalition of players keeps the decision's values for its members and takes a background day's for the rest, and the weights the policy gives are averaged over the 64. All 8,192 coalitions are enumerated by `shap.ExactExplainer` through a masker function, so the values are exact and the same observation, model and background always give the same numbers. A decision takes 0.7 s in the sealed sidecar.

## The record

Each run that explained something writes one `xai` entry to `data/audit.jsonl`:

| Field | What it holds |
| --- | --- |
| `agents.<name>.model` | the model that made the decisions: version, trained date, window, seed, steps |
| `agents.<name>.shap` | the library's version in the sidecar |
| `agents.<name>.background` | the window the background was drawn from, its days, the sample size and the seed |
| `agents.<name>.decisions[]` | `decided` (the close the decision was made after), `weights` (HLAL, SPUS, cash), `base` (the average over the background), `values` (each indicator's push on each weight), `inputs` (what the agent saw: standardised indicators, the weights held), `residual` (how far base plus pushes miss the weights, about 1e-7) |
| `agents.<name>.overall` | on the first run of a model only: the mean size of each push over 60 training days spread across its window |
| `unexplained[]` | a decision whose model was replaced before it was explained, with the reason |
| `manifest` | what the run used: the explainer, the sample, a fingerprint of the prices, the code's commit and the API's package versions |

Numbers are rounded to six places. `inputs` and `decided` with the model's version are what a re-run needs, so an entry can be checked while its model is in force; a retrain overwrites the model file.

## When it runs

The desk's check (5 seconds after start, then every 30 minutes) follows an agent, which returns the observation of each decision it made, queues those decisions with the model's version, and sends them to the sidecar in one `ai_explain` job together with the model's overall if it has none. The entry is written when the run ends; only then are the explained decisions dropped from the queue, so a crash can repeat a decision and never lose one. A failed explanation leaves the decisions queued for the next check and the agent keeps being followed. Decisions made before this existed are explained once, for the latest one, at the first check after it; earlier ones are not.

## On the dashboard

The AGENTS view of the Terminal has a Why section: choose an agent, one of its last 30 explained decisions and a weight. A bar shows each indicator's push on that weight (green raises it, red lowers it); a table lists what the agent saw and the push on every weight in points; and a second table shows how big a push usually is on the model's training days and on the decisions listed. Decisions waiting to be explained are counted.

## What the values mean

They explain the network: why this model, given this observation and measured against an average training day, gave these weights. They do not say what moved the market, they are not causal, and another seed of the same agent explains the same day differently because it is a different policy ([xai-research.md](xai-research.md#how-far-the-explanations-can-be-trusted)).

## Where the code is

| Module | What it holds |
| --- | --- |
| `finrl_sidecar/explain.py` | the players, the masker and the exact values |
| `finrl_sidecar/runner.py` | the tasks `ai_explain` and `selftest`, and the observation of each decision in `ai_shadow` |
| `finrl_sidecar/tests/` | unit tests against a brute-force enumeration; `selftest` runs them (`tests/test_ai_sidecar.py`) |
| `features/systematic/agents.py` | the queue, the explain step and the audit write |
| `features/systematic/explanations.py` | the record's shape and reading it back |
| algotr-frontend: `src/views/Signals.tsx`, `xai.ts` | the waterfall, the feature table and the narrative |

## First live run

Switching `AI_AGENTS` on at 10:57 UTC on 4 November 2026 gave the agents their records (PPO about 3 minutes, SAC about 45, A2C about 3) and then showed a fault in the earlier agent code that no test had met. The desk's folder is the relative `data/ai` and the sidecar runs in a temporary directory, so each model and its scaling were saved there, and the next job failed with `No such file or directory: data/ai/models/ppo.json`. The earlier live check had used an absolute scratch folder. The paths sent to the sidecar are now absolute and an agent whose files are missing counts as due for training, which is why every live model is version 2.

The restart at 11:48 trained, followed and explained the three agents in 3.5 minutes. One `xai` entry (9 KB, number 73 of an intact chain) holds the decisions made after the close of 3 November, and each one's pushes and base add up to its weights within 2e-7:

| Agent | Chose: HLAL, SPUS, cash | An average training day | Biggest single push, in points |
| --- | --- | --- | --- |
| PPO | 14.7%, 34.8%, 50.5% | 32.6%, 28.0%, 39.4% | 5-day return: +23.7, +12.2, -35.9 |
| SAC | 1.1%, 98.3%, 0.6% | 35.6%, 27.3%, 37.1% | CCI: +11.6, +27.6, -39.2 |
| A2C | 0.0%, 99.9%, 0.0% | 28.2%, 33.3%, 38.5% | DX: -12.7, +12.3, +0.4 |

The pushes are large and mostly cancel: PPO's weights are 17.9 points under an average day for HLAL, 6.8 over for SPUS and 11.1 over for cash, while single indicators push them by 20 to 36 points. All three saw prices far below their training window (the indicators of both ETFs were 1 to 2 deviations down). A PPO trained the same way in a scratch folder before the live run, with the same seed and prices, gave the same weights and values, so the same inputs reproduce the same explanation across trainings.

On real prices in the sealed sidecar an `ai_explain` job took 29 s for one decision with the 60-day overall and 6.9 s for five decisions without it; `ai_shadow` took 4 to 5 s.

Not yet seen live: the explanation of a later day, a catch-up after days away (rehearsed on real prices in a scratch folder with the record set back five sessions, and tested with fakes), the second model of a month (the next retrain is on 1 December, or sooner if drift asks for one) and a decision whose model was replaced before it was explained (tests only).
