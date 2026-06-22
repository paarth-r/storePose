"""Compare a predicted occupancy timeline against ground truth.

Occupancy is the upstream signal that drives the busy label, so before trusting
any Low/Med/High number we check that the per-frame waiting count itself is
right. Both timelines are ``(t_seconds, occupancy)`` pairs; we align them on
shared timestamps and report error and correlation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OccupancyReport:
    n: int
    mae: float
    bias: float          # mean(pred - gt); positive = systematic over-count
    corr: float          # Pearson; 0.0 if a series is constant or no overlap
    gt_mean: float
    pred_mean: float

    def format(self) -> str:
        return "\n".join(
            [
                f"samples scored : {self.n}",
                f"occupancy MAE  : {self.mae:.3f}",
                f"bias (pred-gt) : {self.bias:+.3f}",
                f"correlation    : {self.corr:.3f}",
                f"mean occ gt    : {self.gt_mean:.3f}",
                f"mean occ pred  : {self.pred_mean:.3f}",
            ]
        )


def _pearson(xs: list[int], ys: list[int]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / ((vx ** 0.5) * (vy ** 0.5))


def occupancy_eval(
    gt: list[tuple[float, int]], pred: list[tuple[float, int]]
) -> OccupancyReport:
    """Align GT and predicted occupancy on shared timestamps and score them."""
    gd = {round(t, 3): occ for t, occ in gt}
    pdct = {round(t, 3): occ for t, occ in pred}
    keys = sorted(set(gd) & set(pdct))
    n = len(keys)
    if n == 0:
        return OccupancyReport(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    gs = [gd[k] for k in keys]
    ps = [pdct[k] for k in keys]
    mae = sum(abs(g - p) for g, p in zip(gs, ps)) / n
    bias = sum(p - g for g, p in zip(gs, ps)) / n
    return OccupancyReport(
        n=n,
        mae=mae,
        bias=bias,
        corr=_pearson(gs, ps),
        gt_mean=sum(gs) / n,
        pred_mean=sum(ps) / n,
    )
