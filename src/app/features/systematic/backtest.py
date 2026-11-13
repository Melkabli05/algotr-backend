import pandas as pd

from app.features.systematic import signals
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.schemas import (
    Evidence,
    GrowthPoint,
    ResearchResult,
    StrategySpec,
)
from app.quant import performance, proxy, validation
from app.quant.history import NotEnoughHistoryError

MIN_DAYS = 60
GROWTH_POINTS = 500
PROXY_LAG = 2


def sample(growth: pd.DataFrame) -> pd.DataFrame:
    step = -(-len(growth) // GROWTH_POINTS)
    return pd.concat([growth.iloc[::step], growth.iloc[[-1]]]).loc[lambda f: ~f.index.duplicated()]


def fill_prices(spec: StrategySpec, opens: pd.DataFrame | None) -> pd.DataFrame | None:
    """The opens a rule's trades fill at, or None when they fill at the close it decided on.

    Stock orders placed after the close wait for the next open; a coin trades minutes after its
    daily bar closes.
    """
    return opens if spec.fills == "next_open" else None


def daily_returns(
    closes: pd.DataFrame,
    spec: StrategySpec,
    fee_bps: float | None = None,
    opens: pd.DataFrame | None = None,
) -> pd.Series:
    """The daily returns of `spec` after costs, from the first day its rule is warm."""
    warmup = signals.warmup_days(spec)
    held = signals.weights(closes, spec).shift(1).fillna(0.0).iloc[warmup:]
    fills = fill_prices(spec, opens)
    return performance.strategy_returns(
        closes.iloc[warmup:],
        held,
        spec.fee_bps if fee_bps is None else fee_bps,
        None if fills is None else fills.iloc[warmup:],
    )


def on_proxy(
    spec: StrategySpec, market: pd.Series, fee: float, trials: int, spread: float | None
) -> Evidence:
    """The rule run on the market's long history, filled at the close after its signal.

    The proxy has no opens, so the trade waits a whole day: later than the live fill at the
    open, which errs against the rule.
    """
    level = (1 + market).cumprod().to_frame(proxy.NAME)
    warmup = signals.warmup_days(spec)
    held = signals.weights(level, spec).shift(PROXY_LAG).fillna(0.0).iloc[warmup:]
    returns = performance.strategy_returns(level.iloc[warmup:], held, fee)
    benchmark = market.iloc[warmup:]
    hold = pd.DataFrame(1.0, index=benchmark.index, columns=level.columns)
    periods = spec.periods_per_year
    return Evidence(
        name=proxy.NAME,
        start=returns.index[0],
        end=returns.index[-1],
        days=len(returns),
        metrics=performance.metrics(returns, held, periods),
        benchmark=performance.metrics(benchmark, hold, periods),
        sharpe_p=validation.sharpe_difference_p(returns, benchmark),
        deflated_sharpe=validation.deflated_sharpe(returns, trials, spread),
    )


def run(
    closes: pd.DataFrame,
    spec: StrategySpec,
    fee_bps: float | None = None,
    opens: pd.DataFrame | None = None,
    trials: int | None = None,
    spread: float | None = None,
    market: pd.Series | None = None,
) -> ResearchResult:
    """Backtest `spec` on daily closes against an equal-weight buy-and-hold of the same universe.

    Given `opens`, a rule whose orders wait for the next open is filled there (`fill_prices`).
    `trials` and `spread` describe the search the rule came out of (`lab.search`): every rule and
    variant ever tried, and how far their Sharpe ratios scatter; they set the deflated Sharpe.
    A rule marked `proxy` whose universe has a short history is tested for significance on the
    `market`'s century instead (`on_proxy`); the other checks stay on its own history.
    """
    fee = spec.fee_bps if fee_bps is None else fee_bps
    trials = STRATEGIES.trials() if trials is None else trials
    target = signals.weights(closes, spec)
    warmup = signals.warmup_days(spec)
    if len(closes) < warmup + MIN_DAYS:
        raise NotEnoughHistoryError(
            f"Need {warmup + MIN_DAYS} daily bars for {', '.join(closes)}; got {len(closes)}"
        )
    held = target.shift(1).fillna(0.0).iloc[warmup:]
    window = closes.iloc[warmup:]
    fills = fill_prices(spec, opens)
    fills = None if fills is None else fills.iloc[warmup:]
    returns = performance.strategy_returns(window, held, fee, fills)
    hold_all = pd.DataFrame(1 / len(window.columns), index=window.index, columns=window.columns)
    benchmark = (hold_all * window.pct_change()).sum(axis=1)
    periods = spec.periods_per_year
    split = int(len(returns) * performance.IN_SAMPLE_SHARE)
    growth = pd.DataFrame(
        {
            "strategy": 10_000 * (1 + returns).cumprod(),
            "benchmark": 10_000 * (1 + benchmark).cumprod(),
        }
    )
    sensitivity = [
        performance.sensitivity_at(bps, window, held, periods, fills)
        for bps in performance.SENSITIVITY_FEES
    ]
    deflated = validation.deflated_sharpe(returns, trials, spread)
    sharpe_p = validation.sharpe_difference_p(returns, benchmark)
    overall = performance.metrics(returns, held, periods)
    benchmark_metrics = performance.metrics(benchmark, hold_all, periods)
    out_of_sample = performance.metrics(returns.iloc[split:], held.iloc[split:], periods)
    evidence = (
        on_proxy(spec, market, fee, trials, spread)
        if spec.proxy and market is not None and len(market) > len(returns)
        else None
    )
    tested_p, tested_dsr = (
        (evidence.sharpe_p, evidence.deflated_sharpe) if evidence else (sharpe_p, deflated)
    )
    return ResearchResult(
        strategy=spec,
        start=window.index[0],
        end=window.index[-1],
        days=len(window),
        fee_bps=fee,
        metrics=overall,
        benchmark=benchmark_metrics,
        in_sample=performance.metrics(returns.iloc[:split], held.iloc[:split], periods),
        out_of_sample=out_of_sample,
        sensitivity=sensitivity,
        deflated_sharpe=deflated,
        sharpe_p=sharpe_p,
        evidence=evidence,
        checks=performance.checks(
            overall,
            benchmark_metrics,
            out_of_sample,
            sensitivity,
            tested_dsr,
            trials,
            tested_p,
            f"{evidence.name} since {evidence.start:%Y}" if evidence else None,
        ),
        growth=[
            GrowthPoint(time=time, strategy=row.strategy, benchmark=row.benchmark)
            for time, row in sample(growth).iterrows()
        ],
        signals={symbol: float(target[symbol].iloc[-1]) for symbol in target},
    )
