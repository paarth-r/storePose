"""Reject implausibly short completed visits (phantom detections).

A flaky detection can yield a "visit" that appears and checks out in ~2s when
real visits cluster much higher. :class:`OutlierFilter` judges each completed
visit against the typical duration **for its own outcome**, flagging (not
destroying) the too-short ones. Pure and stream-ordered, so the same instance can
scrub a recorded wait log offline. See
``docs/superpowers/specs/2026-06-15-wait-outlier-filter-design.md``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from statistics import median

from .types import CompletedWait

_WINDOW_MAX = 200  # trailing accepted durations kept per outcome


class OutlierFilter:
    """Flags completed visits shorter than a per-outcome plausibility threshold.

    Threshold = ``max(floor, frac * median(recent durations for this outcome))``
    once ``warmup`` accepted samples exist for the outcome; only ``floor`` applies
    before that. Rejected visits do **not** update the running median, so phantoms
    cannot drag it down.
    """

    def __init__(self, floor: float = 2.0, frac: float = 0.25, warmup: int = 10):
        self.floor = max(0.0, floor)
        self.frac = min(1.0, max(0.0, frac))
        self.warmup = max(0, warmup)
        self._windows: dict[str, deque[float]] = {}

    def threshold(self, outcome: str) -> float:
        """Current too-short cutoff (seconds) for ``outcome``."""
        win = self._windows.get(outcome)
        if win and len(win) >= self.warmup:
            return max(self.floor, self.frac * median(win))
        return self.floor

    def judge(self, wait: CompletedWait) -> CompletedWait:
        """Return ``wait`` (or a ``rejected=True`` copy) and update per-outcome stats.

        Accepted durations feed the running median; rejected ones do not.
        """
        duration = wait.exited_s - wait.entered_s
        if duration < self.threshold(wait.outcome):
            return replace(wait, rejected=True)
        self._windows.setdefault(wait.outcome, deque(maxlen=_WINDOW_MAX)).append(duration)
        return wait
