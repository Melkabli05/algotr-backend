from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from app.features.scanner import features, rank, universe
from app.features.scanner.features import Features
from app.features.scanner.schemas import Candidate, Explanation, Reason, Scan, ScanFunnel

LOOKBACK = features.BETA_WINDOW // 2 + features.LONG + 25
TOP_SHOWN = 50
PAGE = rank.Variant("rank_avg", 20)


@dataclass(frozen=True)
class Analysis:
    """The market on its last session: features, who qualifies, and what each variant would hold."""

    found: Features
    ok: pd.DataFrame
    scored: dict[str, pd.DataFrame]
    members: list[str]
    within: pd.Series
    picks: dict[str, dict[str, float]]
    scheduled: dict[str, dict[str, float]]


def tail(panel: pd.DataFrame, sessions: int = LOOKBACK) -> pd.DataFrame:
    """The last `sessions` sessions, which is all any feature or eligibility test reads."""
    dates = np.sort(panel["date"].unique())
    return panel[panel["date"] >= dates[-sessions:][0]]


def subset(found: Features, symbols: list[str]) -> Features:
    return replace(
        found, **{name: getattr(found, name)[symbols] for name in Features.__dataclass_fields__}
    )


def analyse(panel: pd.DataFrame, members: Collection[str]) -> Analysis:
    found = features.compute(tail(panel))
    ok = rank.eligible(found)
    named = [symbol for symbol in found.close.columns if symbol in members]
    within = subset(found, named)
    ok_within = rank.eligible(within)
    scored = rank.scores(within, ok_within)
    within_ranking = scored[PAGE.score].iloc[-1].where(ok_within.iloc[-1]).dropna()
    picks, scheduled = {}, {}
    for variant in rank.VARIANTS:
        chosen = rank.select(scored[variant.score], ok_within, variant.top_n)
        weighted = rank.weights(chosen, within.vol60)
        picks[variant.label] = held(weighted.iloc[-1])
        scheduled[variant.label] = held(rank.targets(weighted, ok_within).iloc[-1])
    return Analysis(
        found,
        ok,
        rank.scores(found, ok),
        named,
        within_ranking.sort_values(ascending=False),
        picks,
        scheduled,
    )


def held(weight: pd.Series) -> dict[str, float]:
    return {symbol: round(float(value), 6) for symbol, value in weight[weight > 0].items()}


def finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def lists_of(symbol: str, lists: Mapping[str, Collection[str]]) -> list[str]:
    return sorted(name for name, symbols in lists.items() if symbol in symbols)


def scan(
    analysis: Analysis,
    names: Mapping[str, str],
    lists: Mapping[str, Collection[str]],
    stale_lists: list[str],
) -> Scan:
    found, ok = analysis.found, analysis.ok
    close, adv = found.close.iloc[-1], found.adv20.iloc[-1]
    liquid = (close >= universe.MIN_PRICE) & (adv >= universe.MIN_DOLLAR_VOLUME)
    listed = close.index.isin(analysis.members)
    qualified = ok.iloc[-1]
    ranking = analysis.scored[PAGE.score].iloc[-1].where(qualified).dropna()
    ranking = ranking.sort_values(ascending=False)
    picked = set(analysis.picks[PAGE.label])
    shown = [*ranking.index[:TOP_SHOWN], *(s for s in ranking.index[TOP_SHOWN:] if s in picked)]
    candidates = [
        Candidate(
            rank=int(ranking.index.get_loc(symbol)) + 1,
            member_rank=(
                int(analysis.within.index.get_loc(symbol)) + 1
                if symbol in analysis.within.index
                else None
            ),
            symbol=symbol,
            name=names.get(symbol),
            score=float(ranking[symbol]),
            mom_12_1=finite(found.mom_12_1[symbol].iloc[-1]),
            mom_6_1=finite(found.mom_6_1[symbol].iloc[-1]),
            clenow_90=finite(found.clenow_90[symbol].iloc[-1]),
            price=float(close[symbol]),
            dollar_volume=float(adv[symbol]),
            lists=lists_of(symbol, lists),
            picked=symbol in picked,
            weight=analysis.picks[PAGE.label].get(symbol),
        )
        for symbol in shown
    ]
    return Scan(
        session=found.close.index[-1].date(),
        variant=PAGE.label,
        funnel=ScanFunnel(
            liquid=int(liquid.sum()),
            listed=int((liquid & listed).sum()),
            trending=int((liquid & found.trend.iloc[-1]).sum()),
            eligible=int(qualified.sum()),
            listed_eligible=int((qualified & listed).sum()),
            picked=len(picked),
        ),
        candidates=candidates,
        picks=analysis.picks,
        benchmark=sorted(liquid.index[liquid & listed]),
        stale_lists=stale_lists,
    )


def explain(
    analysis: Analysis,
    symbol: str,
    names: Mapping[str, str],
    lists: Mapping[str, Collection[str]],
) -> Explanation:
    """Every test a name must pass, with its own figure beside each, and what that adds up to."""
    found = analysis.found
    if symbol not in found.close.columns or pd.isna(found.close[symbol].iloc[-1]):
        return Explanation(
            symbol=symbol,
            name=names.get(symbol),
            verdict="excluded",
            reasons=[Reason(check="Has a bar for the last session", passed=False, detail="none")],
        )
    last = {name: getattr(found, name)[symbol].iloc[-1] for name in Features.__dataclass_fields__}
    average = found.close[symbol].rolling(features.TREND_WINDOW).mean().iloc[-1]
    member = symbol in analysis.members
    on = lists_of(symbol, lists)
    reasons = [
        Reason(
            check=f"Price at least ${universe.MIN_PRICE:,.0f}",
            passed=bool(last["close"] >= universe.MIN_PRICE),
            detail=f"${last['close']:,.2f}",
        ),
        Reason(
            check=f"Dollar volume at least ${universe.MIN_DOLLAR_VOLUME / 1e6:,.0f}M a day",
            passed=bool(last["adv20"] >= universe.MIN_DOLLAR_VOLUME),
            detail=f"${last['adv20'] / 1e6:,.1f}M",
        ),
        Reason(
            check=f"At least {rank.MIN_HISTORY} sessions of history",
            passed=bool(last["history"] >= rank.MIN_HISTORY),
            detail=f"{int(last['history'])} sessions",
        ),
        Reason(
            check="Closed above its 200-day average",
            passed=bool(last["trend"]),
            detail=f"${last['close']:,.2f} against ${average:,.2f}"
            if pd.notna(average)
            else "fewer than 200 sessions",
        ),
        Reason(
            check=f"No daily move above {rank.MAX_GAP:.0%} in 90 sessions",
            passed=bool(last["gap90"] <= rank.MAX_GAP),
            detail=f"largest move {last['gap90']:.1%}" if pd.notna(last["gap90"]) else "no figure",
        ),
    ]
    qualifies = bool(analysis.ok[symbol].iloc[-1])
    picked = symbol in analysis.picks[PAGE.label]
    reasons.append(
        Reason(
            check="On a Shariah list",
            passed=member,
            detail=", ".join(on) if on else "no current list holds it",
        )
    )
    if qualifies and member:
        position = int(analysis.within.index.get_loc(symbol)) + 1
        reasons.append(
            Reason(
                check=f"Among the top {PAGE.top_n} qualifying Shariah-listed names",
                passed=picked,
                detail=f"rank {position} of {len(analysis.within)}",
            )
        )
    return Explanation(
        symbol=symbol,
        name=names.get(symbol),
        session=found.close.index[-1].date(),
        verdict=verdict(picked, qualifies, member),
        reasons=reasons,
    )


def verdict(
    picked: bool, qualifies: bool, member: bool
) -> Literal["picked", "eligible", "scan only", "excluded"]:
    if picked:
        return "picked"
    if qualifies:
        return "eligible" if member else "scan only"
    return "excluded"
