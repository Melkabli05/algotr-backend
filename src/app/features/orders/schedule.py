from collections.abc import Sequence
from datetime import datetime, timedelta
from itertools import pairwise

from app.features.orders.schemas import Interval
from app.quant.volume import BUCKET, BUCKET_MINUTES, Profile

PRECISION = 9


def uniform(
    start: datetime, minutes: int, count: int, weights: Sequence[float] | None = None
) -> list[Interval]:
    """`count` equal intervals from `start`, weighted as given or all alike (a TWAP)."""
    step = timedelta(minutes=minutes) / count
    return [
        Interval(start=start + step * index, weight=1.0 if weights is None else weights[index])
        for index in range(count)
    ]


def volume_weighted(start: datetime, end: datetime, profile: Profile) -> list[Interval]:
    """Intervals cut at the five-minute marks, each weighted by its bucket's share of the volume."""
    edge = start.replace(
        minute=start.minute - start.minute % BUCKET_MINUTES, second=0, microsecond=0
    )
    edges = [start]
    while (edge := edge + BUCKET) < end:
        edges.append(edge)
    edges.append(end)
    return [
        Interval(start=begin, weight=profile.share_at(begin) * (finish - begin) / BUCKET)
        for begin, finish in pairwise(edges)
    ]


def next_slice(
    left: float, weights: Sequence[float], index: int, *, whole: bool, lean: float = 0.0
) -> float:
    """The slice for interval `index`: its weight's share of what is left."""
    if index >= len(weights) - 1:
        return left
    ahead = sum(weights[index:])
    share = left * weights[index] / ahead if ahead > 0 else left
    share = min(share * (1 + lean), left)
    return float(round(share)) if whole else round(share, PRECISION)


def preview(qty: float, weights: Sequence[float], *, whole: bool) -> list[float]:
    """The slices if every one fills in full."""
    left, sizes = qty, []
    for index in range(len(weights)):
        size = next_slice(left, weights, index, whole=whole)
        sizes.append(size)
        left = round(left - size, PRECISION)
    return sizes
