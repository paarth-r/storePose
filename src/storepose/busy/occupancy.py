"""Reconstruct an occupancy timeline from completed-wait intervals.

The live pipeline knows occupancy directly (people currently waiting), but for
*offline* analysis of an existing ``waits.csv`` we only have completed intervals
``[entered_s, exited_s)``. Occupancy at time ``t`` is the number of intervals
covering ``t``; sampling it at a fixed step yields the per-frame-style signal the
:class:`~storepose.busy.aggregator.BusyAggregator` expects.

Caveat: a wait log only contains people who reached the WAITING state and then
left, so this undercounts anyone still in line at the end of the clip. It is a
faithful reconstruction of the logged waits, not of raw detections.
"""

from __future__ import annotations

from ..queue.types import CompletedWait


def occupancy_at(waits: list[CompletedWait], t: float) -> int:
    """Number of waits whose ``[entered_s, exited_s)`` interval covers ``t``."""
    return sum(1 for w in waits if w.entered_s <= t < w.exited_s)


def sample_occupancy(
    waits: list[CompletedWait],
    step: float = 1.0,
    t_start: float = 0.0,
    t_end: float | None = None,
) -> list[tuple[float, int]]:
    """Sample occupancy on ``[t_start, t_end)`` every ``step`` seconds.

    ``t_end`` defaults to the latest ``exited_s``. Returns ``(t, occupancy)``
    pairs; empty if there are no waits.
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if not waits:
        return []
    if t_end is None:
        t_end = max(w.exited_s for w in waits)
    out: list[tuple[float, int]] = []
    t = t_start
    # guard against runaway loops on degenerate input
    while t < t_end:
        out.append((t, occupancy_at(waits, t)))
        t += step
    return out
