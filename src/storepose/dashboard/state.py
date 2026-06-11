"""Thread-safe live metric buffers the runner pushes into for the dashboard."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


@dataclass
class Visit:
    t: float
    wait_seconds: float
    serving_seconds: float
    outcome: str


class DashboardState:
    """Bounded, lock-guarded occupancy samples + completed-visit events."""

    def __init__(self, occ_interval: float = 1.0, max_samples: int = 3600,
                 max_visits: int = 5000):
        self._lock = threading.Lock()
        self._occ: deque = deque(maxlen=max_samples)      # (t, waiting, serving)
        self._visits: deque = deque(maxlen=max_visits)    # Visit
        self._occ_interval = occ_interval
        self._last_occ_t: float | None = None

    def observe(self, t: float, waiting: int, serving: int) -> None:
        with self._lock:
            if self._last_occ_t is None or (t - self._last_occ_t) >= self._occ_interval:
                self._occ.append((float(t), int(waiting), int(serving)))
                self._last_occ_t = t

    def add_visit(self, t: float, wait_seconds: float, serving_seconds: float,
                  outcome: str) -> None:
        with self._lock:
            self._visits.append(
                Visit(float(t), float(wait_seconds), float(serving_seconds), str(outcome))
            )

    def snapshot(self) -> tuple[list, list]:
        with self._lock:
            return list(self._occ), list(self._visits)
