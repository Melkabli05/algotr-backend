"""Runs one FinRL job inside the sidecar environment: `runner.py job.json result.json`.

The host writes the job (task name, parameters, paths to OHLCV files) and reads the result. Nothing
here touches the network or any broker: FinRL only ever sees the price files it is given.
"""

import importlib
import json
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

ALGORITHMS = ("a2c", "ddpg", "ppo", "sac", "td3")
AGENT_SETTINGS = {
    "ppo": {"n_steps": 512, "batch_size": 64},
    "a2c": {"n_steps": 16},
    "sac": {"learning_starts": 500},
}


def returns_of(account: pd.DataFrame, column: str = "account_value") -> dict[str, list]:
    """Daily returns from an account-value table with `date` and the value column."""
    values = account.set_index(pd.to_datetime(account["date"]))[column].astype(float)
    daily = values.pct_change().dropna()
    return {"dates": [d.strftime("%Y-%m-%d") for d in daily.index], "returns": daily.tolist()}


def sharpe(returns: list[float]) -> float:
    series = pd.Series(returns, dtype=float)
    return float(series.mean() / series.std() * np.sqrt(252)) if series.std() > 0 else 0.0


def load(job: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The traded universe and the wider market, both aligned to days every ticker traded."""
    frames = []
    for name in ("data", "market"):
        frame = pd.read_csv(job["files"][name])
        counts = frame.groupby("date")["tic"].transform("nunique")
        frames.append(frame[counts == frame["tic"].nunique()].reset_index(drop=True))
    return frames[0], frames[1]


def prepare(job: dict) -> pd.DataFrame:
    """FinRL's own FeatureEngineer: technical indicators plus a turbulence index from the market."""
    from finrl.config import INDICATORS
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer

    data, market = load(job)
    engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    processed = engineer.preprocess_data(data)
    turbulence = engineer.calculate_turbulence(market)
    processed = processed.merge(turbulence, on="date", how="left")
    processed["turbulence"] = processed["turbulence"].fillna(0.0)
    processed = processed.dropna().sort_values(["date", "tic"]).reset_index(drop=True)
    processed.index = processed["date"].factorize()[0]
    return processed


def env_kwargs(frame: pd.DataFrame, job: dict) -> dict:
    from finrl.config import INDICATORS

    stocks = frame["tic"].nunique()
    costs = [job["params"]["cost"]] * stocks
    return {
        "hmax": job["params"]["hmax"],
        "initial_amount": job["params"]["initial_amount"],
        "num_stock_shares": [0] * stocks,
        "buy_cost_pct": costs,
        "sell_cost_pct": costs,
        "state_space": 1 + 2 * stocks + len(INDICATORS) * stocks,
        "stock_dim": stocks,
        "tech_indicator_list": INDICATORS,
        "action_space": stocks,
        "reward_scaling": 1e-4,
    }


def window(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    from finrl.meta.preprocessor.preprocessors import data_split

    return data_split(frame, start, end)


def task_inventory(job: dict) -> dict:
    """Which FinRL features import and run in this environment, and exactly what is installed."""
    checks = dict(job["params"]["modules"])
    status = {}
    for feature, module in checks.items():
        try:
            importlib.import_module(module)
            status[feature] = {"ok": True, "detail": module}
        except Exception as exc:
            status[feature] = {
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}".splitlines()[0],
            }
    direct = metadata.distribution("finrl").read_text("direct_url.json")
    commit = json.loads(direct)["vcs_info"]["commit_id"] if direct else None
    return {
        "commit": commit,
        "python": sys.version.split()[0],
        "packages": {
            name: metadata.version(name)
            for name in ("finrl", "stable-baselines3", "gymnasium", "torch", "elegantrl", "optuna")
            if _installed(name)
        },
        "features": status,
    }


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def task_features(job: dict) -> dict:
    """What FeatureEngineer adds to the prices, and which days the turbulence index flags."""
    from finrl.config import INDICATORS

    frame = prepare(job)
    days = frame.drop_duplicates("date").set_index("date")["turbulence"]
    return {
        "rows": len(frame),
        "tickers": sorted(frame["tic"].unique().tolist()),
        "indicators": INDICATORS,
        "columns": [c for c in frame.columns if c not in ("date", "tic")],
        "turbulence_top_days": {d: float(v) for d, v in days.nlargest(5).items()},
        "turbulence_median": float(days.median()),
    }


def train_and_trade(frame, job, seed, fold, algorithm, env_name):
    """One agent through FinRL's own train and predict path, on one walk-forward fold."""
    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    classes = {"stock": StockTradingEnv}
    if env_name == "stoploss":
        from finrl.meta.env_stock_trading.env_stocktrading_stoploss import StockTradingEnvStopLoss

        classes["stoploss"] = StockTradingEnvStopLoss
    if env_name == "cashpenalty":
        from finrl.meta.env_stock_trading.env_stocktrading_cashpenalty import (
            StockTradingEnvCashpenalty,
        )

        classes["cashpenalty"] = StockTradingEnvCashpenalty
    env_class = classes[env_name]
    kwargs = env_kwargs(frame, job)
    train = window(frame, fold["train_start"], fold["train_end"])
    trade = window(frame, fold["test_start"], fold["test_end"])
    train_env, _ = env_class(df=train, **kwargs).get_sb_env()
    agent = DRLAgent(env=train_env)
    model = agent.get_model(algorithm, seed=seed, verbose=0)
    trained = agent.train_model(
        model=model, tb_log_name=f"{algorithm}-{seed}", total_timesteps=job["params"]["timesteps"]
    )
    held_out, _ = DRLAgent.DRL_prediction(model=trained, environment=env_class(df=trade, **kwargs))
    inside, _ = DRLAgent.DRL_prediction(model=trained, environment=env_class(df=train, **kwargs))
    return returns_of(held_out), sharpe(returns_of(inside)["returns"])


def task_agents(job: dict) -> dict:
    """FinRL's five stable-baselines3 agents over seeds and anchored walk-forward folds."""
    frame = prepare(job)
    params = job["params"]
    runs = []
    for algorithm in params["algorithms"]:
        for seed in params["seeds"]:
            for index, fold in enumerate(params["folds"]):
                held_out, inside = train_and_trade(
                    frame, job, seed, fold, algorithm, params.get("env", "stock")
                )
                runs.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "fold": index,
                        "in_sample": inside,
                        **held_out,
                    }
                )
    return {"runs": runs}


def series_of(account: pd.DataFrame) -> dict[str, list]:
    """Daily returns from FinRL's account tables, whichever column they carry."""
    if "account_value" in account.columns:
        return returns_of(account)
    daily = account.set_index(pd.to_datetime(account["date"]))["daily_return"].astype(float)
    return {"dates": [d.strftime("%Y-%m-%d") for d in daily.index], "returns": daily.tolist()}


def validation_sharpe(summary: pd.DataFrame) -> float | None:
    """The mean Sharpe the chosen model had on each window's validation days, before it traded."""
    scores = summary.iloc[:, 4:].apply(pd.to_numeric, errors="coerce")
    best = scores.max(axis=1).dropna()
    return float(best.mean()) if len(best) else None


def task_ensemble(job: dict) -> dict:
    """FinRL's ensemble: five agents retrained each window, the best validation Sharpe trades."""
    from finrl.agents.stablebaselines3 import models
    from finrl.agents.stablebaselines3.models import DRLEnsembleAgent
    from finrl.config import (
        A2C_PARAMS,
        DDPG_PARAMS,
        INDICATORS,
        PPO_PARAMS,
        SAC_PARAMS,
        TD3_PARAMS,
    )

    # Upstream loops over every model it knows (TQC, CrossQ too) but only takes parameters for
    # the five of its documented ensemble, so the table is narrowed to those five.
    models.MODELS = {name: models.MODELS[name] for name in ALGORITHMS}
    frame = prepare(job)
    params = job["params"]
    kwargs = env_kwargs(frame, job)
    agent = DRLEnsembleAgent(
        df=frame,
        train_period=tuple(params["train_period"]),
        val_test_period=tuple(params["val_test_period"]),
        rebalance_window=params["rebalance_window"],
        validation_window=params["validation_window"],
        stock_dim=kwargs["stock_dim"],
        hmax=kwargs["hmax"],
        initial_amount=kwargs["initial_amount"],
        buy_cost_pct=params["cost"],
        sell_cost_pct=params["cost"],
        reward_scaling=kwargs["reward_scaling"],
        state_space=kwargs["state_space"],
        action_space=kwargs["action_space"],
        tech_indicator_list=INDICATORS,
        print_verbosity=10**9,
    )
    steps = dict.fromkeys(ALGORITHMS, params["timesteps"])
    summary = agent.run_ensemble_strategy(
        A2C_PARAMS, PPO_PARAMS, DDPG_PARAMS, SAC_PARAMS, TD3_PARAMS, steps
    )
    files = sorted(Path("results").glob("account_value_trade_ensemble_*.csv"))
    pieces = [pd.read_csv(f) for f in files]
    account = pd.concat(pieces, ignore_index=True)
    if "date" not in account.columns:
        dates = frame.drop_duplicates("date")["date"]
        account.insert(0, "date", dates.iloc[-len(account) :].to_numpy())
    chosen = summary.iloc[:, 3].tolist() if summary.shape[1] > 3 else []
    return {
        "in_sample": validation_sharpe(summary),
        **series_of(account.drop(columns=[c for c in account.columns if c.startswith("Unnamed")])),
        "models_used": chosen,
        "windows": len(files),
    }


def portfolio_frame(frame: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """The covariance-of-returns column FinRL's portfolio-allocation environment expects."""
    covariances, dates = [], []
    unique = frame["date"].unique()
    for position in range(lookback, len(unique)):
        window_ = frame.loc[position - lookback : position]
        prices = window_.pivot_table(index="date", columns="tic", values="close")
        covariances.append(prices.pct_change().dropna().cov().to_numpy())
        dates.append(unique[position])
    merged = frame.merge(pd.DataFrame({"date": dates, "cov_list": covariances}), on="date")
    merged = merged.sort_values(["date", "tic"]).reset_index(drop=True)
    merged.index = merged["date"].factorize()[0]
    return merged


def task_portfolio(job: dict) -> dict:
    """FinRL's portfolio-allocation environment: the agent chooses weights over the assets."""
    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.config import INDICATORS
    from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv

    frame = portfolio_frame(prepare(job))
    params, fold = job["params"], job["params"]["folds"][-1]
    stocks = frame["tic"].nunique()
    kwargs = {
        "stock_dim": stocks,
        "hmax": params["hmax"],
        "initial_amount": params["initial_amount"],
        "transaction_cost_pct": params["cost"],
        "reward_scaling": 1e-4,
        "state_space": stocks,
        "action_space": stocks,
        "tech_indicator_list": INDICATORS,
    }
    train = window(frame, fold["train_start"], fold["train_end"])
    trade = window(frame, fold["test_start"], fold["test_end"])
    train_env, _ = StockPortfolioEnv(df=train, **kwargs).get_sb_env()
    runs = []
    for algorithm in params["algorithms"]:
        for seed in params["seeds"]:
            agent = DRLAgent(env=train_env)
            model = agent.get_model(algorithm, seed=seed, verbose=0)
            trained = agent.train_model(
                model=model, tb_log_name=f"p-{algorithm}", total_timesteps=params["timesteps"]
            )
            held_out, _ = DRLAgent.DRL_prediction(
                model=trained, environment=StockPortfolioEnv(df=trade, **kwargs)
            )
            inside, _ = DRLAgent.DRL_prediction(
                model=trained, environment=StockPortfolioEnv(df=train, **kwargs)
            )
            runs.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "in_sample": sharpe(series_of(inside)["returns"]),
                    **series_of(held_out),
                }
            )
    return {"runs": runs}


def task_eiie(job: dict) -> dict:
    """FinRL's portfolio-optimisation environment with its EIIE policy-gradient agent."""
    from finrl.agents.portfolio_optimization.architectures import EIIE
    from finrl.agents.portfolio_optimization.models import DRLAgent
    from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
        PortfolioOptimizationEnv,
    )

    frame = prepare(job)
    params, fold = job["params"], job["params"]["folds"][-1]
    train = window(frame, fold["train_start"], fold["train_end"])
    trade = window(frame, fold["test_start"], fold["test_end"])
    kwargs = {
        "initial_amount": params["initial_amount"],
        "comission_fee_pct": params["cost"],
        "time_window": 50,
        "features": ["close", "high", "low"],
        "normalize_df": None,
    }
    runs = []
    for seed in params["seeds"]:
        import torch

        torch.manual_seed(seed)
        env = PortfolioOptimizationEnv(train, **kwargs)
        agent = DRLAgent(env=env)
        model = agent.get_model("pg", device="cpu", model_kwargs={"lr": 0.01, "policy": EIIE})
        DRLAgent.train_model(model, episodes=params["episodes"])
        held_out = eiie_series(model, PortfolioOptimizationEnv(trade, **kwargs))
        inside = eiie_series(model, PortfolioOptimizationEnv(train, **kwargs))
        runs.append(
            {
                "algorithm": "eiie",
                "seed": seed,
                "in_sample": sharpe(inside["returns"]),
                **held_out,
            }
        )
    return {"runs": runs}


def eiie_series(model, env) -> dict[str, list]:
    """Daily returns of a trained EIIE policy run through one environment."""
    from finrl.agents.portfolio_optimization.models import DRLAgent

    DRLAgent.DRL_validation(model, env)
    values = pd.DataFrame(
        {
            "date": [str(d)[:10] for d in env._date_memory],
            "account_value": env._asset_memory["final"],
        }
    )
    return series_of(values)


def task_tune(job: dict) -> dict:
    """FinRL's Optuna tuner, scored on a validation window that comes BEFORE the test period.

    FinRL's tuner scores on whatever `env_trade` it is given; handing it the held-out test data
    would be data snooping, so the host passes a validation window instead.
    """
    from finrl.agents.stablebaselines3.tune_sb3 import LoggingCallback, TuneSB3Optuna
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    frame = prepare(job)
    params, fold = job["params"], job["params"]["folds"][-1]
    kwargs = env_kwargs(frame, job)
    train = window(frame, fold["train_start"], fold["validation_start"])
    validation = window(frame, fold["validation_start"], fold["train_end"])
    train_env, _ = StockTradingEnv(df=train, **kwargs).get_sb_env()
    callback = LoggingCallback(
        threshold=10**9, trial_number=params["trials"], patience=params["trials"]
    )
    tuner = TuneSB3Optuna(
        env_train=train_env,
        model_name=params["algorithms"][0],
        env_trade=StockTradingEnv(df=validation, **kwargs),
        logging_callback=callback,
        total_timesteps=params["timesteps"],
        n_trials=params["trials"],
    )
    study = tuner.run_optuna()
    return {
        "trials": len(study.trials),
        "best_validation_sharpe": float(study.best_value),
        "best_params": {
            k: (v if isinstance(v, int | float | str) else str(v))
            for k, v in study.best_params.items()
        },
    }


def task_stats(job: dict) -> dict:
    """FinRL's pyfolio backtest statistics for return series the host sends."""
    from finrl.plot import backtest_stats

    out = {}
    for name, series in job["params"]["series"].items():
        values = (1 + pd.Series(series["returns"], index=pd.to_datetime(series["dates"]))).cumprod()
        table = pd.DataFrame({"date": values.index, "account_value": values.to_numpy() * 100_000})
        out[name] = {
            k: float(v) for k, v in backtest_stats(table, value_col_name="account_value").items()
        }
    return {"stats": out}


def task_sentiment(job: dict) -> dict:
    """FinBERT's class probabilities for each text, from the local copy of the weights only."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    params = job["params"]
    folder = params["model_dir"]
    texts = pd.read_csv(job["files"]["data"], keep_default_na=False)["text"].astype(str).tolist()
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    passes = int(params.get("passes", 0))
    attention = {"attn_implementation": "eager"} if passes else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        folder, local_files_only=True, **attention
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.eval().to(device)
    torch.manual_seed(params.get("seed", 0))
    labels = {name.lower(): i for i, name in model.config.id2label.items()}
    positive, negative = labels["positive"], labels["negative"]
    order = np.argsort([len(text) for text in texts])
    probabilities = np.zeros((len(texts), model.config.num_labels))
    spreads = np.zeros(len(texts))
    for start in range(0, len(texts), params["batch"]):
        index = order[start : start + params["batch"]]
        encoded = tokenizer(
            [texts[i] for i in index],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            probabilities[index] = model(**encoded).logits.float().softmax(-1).cpu().numpy()
            if passes:
                spreads[index] = dropout_spread(model, encoded, passes, positive, negative)
    record = json.loads(Path(folder, "revision.json").read_text())
    return {
        "labels": [model.config.id2label[i] for i in range(model.config.num_labels)],
        "probabilities": probabilities.tolist(),
        "uncertainty": spreads.tolist() if passes else None,
        "device": device,
        "revision": record["revision"],
    }


def dropout_spread(model, encoded, passes: int, positive: int, negative: int) -> np.ndarray:
    """Monte Carlo dropout (Gal and Ghahramani, 2016): the standard deviation of the score,
    positive minus negative, over `passes` forward passes with dropout left on."""
    model.train()
    try:
        scores = []
        for _ in range(passes):
            found = model(**encoded).logits.float().softmax(-1)
            scores.append((found[:, positive] - found[:, negative]).cpu().numpy())
    finally:
        model.eval()
    return np.std(np.stack(scores), axis=0)


MOOD_COLUMNS = ("sentiment", "news_risk")


def prepare_agents(job: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The features and daily returns of the traded assets, with FinRL's own indicators.

    The headline columns skip FeatureEngineer, which fills gaps backwards as well as forwards
    and would put a later day's mood into an earlier one; they rejoin it by date and fund.
    """
    from finrl.config import INDICATORS
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer
    from weights_env import asset_features

    engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    raw = pd.read_csv(job["files"]["data"])
    mood = [column for column in MOOD_COLUMNS if column in raw.columns]
    processed = engineer.preprocess_data(raw.drop(columns=mood))
    if mood:
        keys = raw[["date", "tic", *mood]].assign(date=raw["date"].astype(str))
        processed = processed.assign(date=processed["date"].astype(str))
        processed = processed.merge(keys, on=["date", "tic"], how="left")
    return asset_features(processed)


def fit_agent(algorithm: str, features: np.ndarray, returns: np.ndarray, params: dict, seed: int):
    """One stable-baselines3 agent trained on episodes drawn from the days given."""
    import stable_baselines3 as sb3
    from weights_env import EPISODE_DAYS, WeightsEnv

    env = WeightsEnv(features, returns, params["cost"], episode_days=EPISODE_DAYS)
    settings = dict(AGENT_SETTINGS[algorithm])
    if algorithm == "sac" and "train_freq" in params:
        settings["train_freq"] = params["train_freq"]
    agent = getattr(sb3, algorithm.upper())(
        "MlpPolicy", env, seed=seed, device="cpu", verbose=0, **settings
    )
    agent.learn(total_timesteps=params["timesteps"])
    return agent


def standardized(frame: pd.DataFrame, mean: pd.Series, std: pd.Series) -> np.ndarray:
    return ((frame - mean) / std).to_numpy()


def cash_weights(assets: int) -> np.ndarray:
    return np.eye(assets + 1)[-1]


def task_ai_walk_forward(job: dict) -> dict:
    """One agent retrained every month on a rolling window, then trading the next month unseen.

    The window stops `purge` + `embargo` days before the month it trades: the purge drops the
    days whose reward (the next day's return) reaches into it, and the embargo keeps the
    serially correlated days just before it out of training (Lopez de Prado, 2018, ch. 7).
    """
    from weights_env import roll_out

    features, returns = prepare_agents(job)
    params, days = job["params"], features.index
    cost, assets = params["cost"], returns.shape[1]
    starts = [int(days.searchsorted(pd.Timestamp(day))) for day in params["retrain_dates"]]
    ends = [*starts[1:], len(days)]
    held, dates, nets, weights, inside = cash_weights(assets), [], [], [], []
    gap = params.get("purge", 0) + params.get("embargo", 0)
    for start, end in zip(starts, ends, strict=True):
        stop = start - gap
        first = stop - params["lookback"]
        if first < 0:
            continue
        window = features.iloc[first:stop]
        mean, std = window.mean(), window.std().replace(0, 1)
        moves, train = returns.iloc[first:stop].to_numpy(), standardized(window, mean, std)
        agent = fit_agent(params["algorithm"], train, moves, params, params["seed"])
        trained, _, _ = roll_out(agent, train, moves, cost, cash_weights(assets))
        inside.append(sharpe(trained))
        block = slice(start - 1, end)
        net, chosen, held = roll_out(
            agent,
            standardized(features.iloc[block], mean, std),
            returns.iloc[block].to_numpy(),
            cost,
            held,
        )
        dates += [day.strftime("%Y-%m-%d") for day in days[start:end]]
        nets += net
        weights += chosen
    return {
        "dates": dates,
        "returns": nets,
        "weights": weights,
        "assets": list(returns.columns),
        "held": held.tolist(),
        "in_sample": float(np.mean(inside)) if inside else None,
    }


def task_kronos_walk_forward(job: dict) -> dict:
    """Kronos forecasting each next day's candle from its last `context` days, traded unseen.

    A fund is held in equal share when the forecast close is above today's, otherwise that share
    stays in cash. The days from `since` on come after the model's published training, so they are
    the unseen ones; the `seen_days` before them are the nearest thing it has to in-sample.
    """
    from weights_env import SCALE, WeightsEnv

    params = job["params"]
    sys.path.insert(0, str(Path(params["model_dir"]) / "code"))
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    folder = Path(params["model_dir"])
    predictor = KronosPredictor(
        Kronos.from_pretrained(folder / "small"),
        KronosTokenizer.from_pretrained(folder / "tokenizer"),
        device="cpu",
        max_context=512,
    )
    raw = pd.read_csv(job["files"]["data"]).sort_values(["tic", "date"])
    bars = {tic: frame.reset_index(drop=True) for tic, frame in raw.groupby("tic")}
    tics = sorted(bars)
    dates = pd.to_datetime(bars[tics[0]]["date"])
    returns = np.column_stack([bars[tic]["close"].pct_change().to_numpy() for tic in tics])
    first = int(dates.searchsorted(pd.Timestamp(params["since"])))
    begin, context = first - params["seen_days"] - 1, params["context"]
    columns = ["open", "high", "low", "close", "volume"]
    torch.manual_seed(params["seed"])
    env = WeightsEnv(np.zeros((len(dates), 1)), np.nan_to_num(returns), params["cost"])
    env.reset()
    env.day = begin
    nets = []
    for day in range(begin, len(dates) - 1):
        window = slice(day - context + 1, day + 1)
        ahead = pd.Series([dates.iloc[day + 1]])
        found = predictor.predict_batch(
            [bars[tic].iloc[window][columns].reset_index(drop=True) for tic in tics],
            [dates.iloc[window].reset_index(drop=True)] * len(tics),
            [ahead] * len(tics),
            pred_len=1,
            sample_count=params["samples"],
            verbose=False,
        )
        rising = np.array(
            [
                f["close"].iloc[0] > bars[tic]["close"].iloc[day]
                for f, tic in zip(found, tics, strict=True)
            ]
        )
        target = np.append(rising / len(tics), 1 - rising.sum() / len(tics))
        _, _, _, _, info = env.step(np.log(target + 1e-12) / SCALE)
        nets.append(info["net"])
    stamps = [d.strftime("%Y-%m-%d") for d in dates.iloc[begin + 1 :]]
    split = first - begin - 1
    return {
        "dates": stamps[split:],
        "returns": nets[split:],
        "in_sample": sharpe(nets[:split]),
        "assets": tics,
    }


def task_ai_train(job: dict) -> dict:
    """Train one agent on the last `lookback` days and keep it with its feature scaling."""
    from weights_env import roll_out

    features, returns = prepare_agents(job)
    params = job["params"]
    features, returns = features.iloc[-params["lookback"] :], returns.iloc[-params["lookback"] :]
    mean, std = features.mean(), features.std().replace(0, 1)
    train = standardized(features, mean, std)
    agent = fit_agent(params["algorithm"], train, returns.to_numpy(), params, params["seed"])
    agent.save(params["model_path"])
    Path(params["scaler_path"]).write_text(
        json.dumps(
            {"mean": mean.to_dict(), "std": std.to_dict(), "columns": list(features.columns)}
        )
    )
    nets, _, _ = roll_out(
        agent, train, returns.to_numpy(), params["cost"], cash_weights(returns.shape[1])
    )
    return {
        "first": features.index[0].strftime("%Y-%m-%d"),
        "last": features.index[-1].strftime("%Y-%m-%d"),
        "days": len(features),
        "assets": list(returns.columns),
        "in_sample_sharpe": sharpe(nets),
    }


def task_ai_shadow(job: dict) -> dict:
    """Run a trained agent over the days since the last one handled and choose for the next day."""
    import stable_baselines3 as sb3
    from weights_env import SCALE, WeightsEnv, policy_confidence, softmax

    features, returns = prepare_agents(job)
    params, days = job["params"], features.index
    scaler = json.loads(Path(params["scaler_path"]).read_text())
    shown = standardized(
        features[scaler["columns"]], pd.Series(scaler["mean"]), pd.Series(scaler["std"])
    )
    agent = getattr(sb3, params["algorithm"].upper()).load(params["model_path"], device="cpu")
    first = (
        int(days.searchsorted(pd.Timestamp(params["since"]))) if params["since"] else len(days) - 1
    )
    held = np.asarray(params["held"] or cash_weights(returns.shape[1]), dtype=float)
    env = WeightsEnv(shown[first:], returns.to_numpy()[first:], params["cost"], initial=held)
    observation, _ = env.reset()
    dates, nets, applied, decisions = [], [], [], []
    confidence: dict[str, list[float]] = {}
    done = len(shown) - first <= 1
    while not done:
        if not dates and params["target"]:
            action = np.log(np.maximum(np.asarray(params["target"]), 1e-12)) / SCALE
        else:
            decided = days[first + len(dates)].strftime("%Y-%m-%d")
            decisions.append({"date": decided, "observation": observation.tolist()})
            confidence[decided] = policy_confidence(agent, observation, env.held).tolist()
            action, _ = agent.predict(observation, deterministic=True)
        observation, _, done, _, info = env.step(action)
        dates.append(days[first + len(dates) + 1].strftime("%Y-%m-%d"))
        nets.append(info["net"])
        applied.append(info["weights"].tolist())
    choice, _ = agent.predict(observation, deterministic=True)
    decisions.append({"date": days[-1].strftime("%Y-%m-%d"), "observation": observation.tolist()})
    confidence[days[-1].strftime("%Y-%m-%d")] = policy_confidence(
        agent, observation, env.held
    ).tolist()
    return {
        "confidence": confidence,
        "dates": dates,
        "returns": nets,
        "applied": applied,
        "held": env.held.tolist(),
        "target": softmax(SCALE * np.asarray(choice, dtype=float)).tolist(),
        "assets": list(returns.columns),
        "last": days[-1].strftime("%Y-%m-%d"),
        "decisions": decisions,
    }


def task_ai_explain(job: dict) -> dict:
    """SHAP values of an agent's weights for the observations given and for its training days."""
    import explain as xai
    import shap
    import stable_baselines3 as sb3
    from weights_env import roll_out

    features, returns = prepare_agents(job)
    params = job["params"]
    scaler = json.loads(Path(params["scaler_path"]).read_text())
    agent = getattr(sb3, params["algorithm"].upper()).load(params["model_path"], device="cpu")
    first, last, assets = params["first"], params["last"], returns.shape[1]
    mean, std = pd.Series(scaler["mean"]), pd.Series(scaler["std"])
    shown = standardized(features.loc[first:last, scaler["columns"]], mean, std)
    seen: list[np.ndarray] = []
    roll_out(
        agent, shown, returns.loc[first:last].to_numpy(), params["cost"], cash_weights(assets), seen
    )
    pool = np.array(seen)
    names, groups = xai.layout(scaler["columns"], assets)
    weights, background = xai.weights_of(agent), xai.background_of(pool)
    wanted = np.asarray(params["observations"], dtype=np.float32).reshape(-1, pool.shape[1])
    overall = xai.explain(weights, xai.evenly(pool, params["sample"]), background, groups)
    return {
        "groups": names,
        "assets": [*returns.columns, "cash"],
        "columns": [*scaler["columns"], *(f"held:{name}" for name in [*returns.columns, "cash"])],
        "background": {
            "first": first,
            "last": last,
            "days": len(pool),
            "rows": len(background),
            "seed": xai.SEED,
        },
        "decisions": xai.explain(weights, wanted, background, groups),
        "overall": np.abs([item["values"] for item in overall]).mean(axis=0).tolist()
        if overall
        else None,
        "sample": len(overall),
        "shap": shap.__version__,
    }


def task_selftest(job: dict) -> dict:
    """Run the sidecar's own unit tests, which need packages the project itself does not have."""
    import unittest

    folder = Path(__file__).parent / "tests"
    result = unittest.TestResult()
    unittest.defaultTestLoader.discover(str(folder)).run(result)
    return {
        "ran": result.testsRun,
        "problems": [f"{test}: {trace[-600:]}" for test, trace in result.failures + result.errors],
    }


TASKS = {
    "ai_walk_forward": task_ai_walk_forward,
    "ai_train": task_ai_train,
    "kronos_walk_forward": task_kronos_walk_forward,
    "ai_shadow": task_ai_shadow,
    "ai_explain": task_ai_explain,
    "selftest": task_selftest,
    "sentiment": task_sentiment,
    "inventory": task_inventory,
    "features": task_features,
    "agents": task_agents,
    "ensemble": task_ensemble,
    "portfolio": task_portfolio,
    "eiie": task_eiie,
    "tune": task_tune,
    "stats": task_stats,
}


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text())
    signal.signal(signal.SIGALRM, lambda *_: os._exit(3))
    signal.alarm(int(job.get("limit", 1500)))
    started = time.monotonic()
    workdir = tempfile.mkdtemp(prefix="finrl-")
    os.chdir(workdir)
    for folder in ("results", "trained_models", "tensorboard_log", "datasets"):
        os.makedirs(folder, exist_ok=True)
    try:
        outcome = {"status": "ok", "data": TASKS[job["task"]](job)}
    except Exception as exc:
        outcome = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-2000:],
        }
    outcome |= {"task": job["task"], "seconds": round(time.monotonic() - started, 2)}
    Path(sys.argv[2]).write_text(json.dumps(outcome))
    os.chdir(tempfile.gettempdir())
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
