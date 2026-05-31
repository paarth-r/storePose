"""Rolling-average frames-per-second meter."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable


class FpsMeter:
    """Measures FPS as a rolling average over the last ``window`` frames.

    Call :meth:`tick` once per processed frame; it returns the current
    smoothed FPS (``0.0`` until at least two ticks have been recorded).
    """

    def __init__(self, window: int = 30, clock: Callable[[], float] = time.perf_counter):
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self._clock = clock
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Record a frame and return the current rolling FPS."""
        self._timestamps.append(self._clock())
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
