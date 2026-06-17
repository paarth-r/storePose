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
                 max_visits: int = 5000, max_busy: int = 4000):
        self._lock = threading.Lock()
        self._occ: deque = deque(maxlen=max_samples)      # (t, waiting, serving)
        self._visits: deque = deque(maxlen=max_visits)    # Visit
        self._busy: deque = deque(maxlen=max_busy)        # (t, level, value)
        self._busy_current: tuple | None = None
        self._occ_interval = occ_interval
        self._last_occ_t: float | None = None
        self._debug_frame: int | None = None       # frame index being viewed
        self._debug_rows: list = []                 # per-person reasoning rows
        self.num_mashgins: int = 1                  # parallel Mashgin kiosks (display)

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

    def set_busy(self, t: float, level: str, value: float) -> None:
        with self._lock:
            entry = (float(t), str(level), float(value))
            self._busy_current = entry
            self._busy.append(entry)

    def set_debug(self, frame: int, rows: list) -> None:
        with self._lock:
            self._debug_frame = int(frame)
            self._debug_rows = list(rows)

    def snapshot(self) -> tuple[list, list]:
        with self._lock:
            return list(self._occ), list(self._visits)

    def busy_snapshot(self) -> tuple[tuple | None, list]:
        with self._lock:
            return self._busy_current, list(self._busy)

    def debug_snapshot(self) -> tuple[int | None, list]:
        with self._lock:
            return self._debug_frame, list(self._debug_rows)
