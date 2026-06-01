"""One-Euro filtering for low-jitter, low-lag signal smoothing."""

from __future__ import annotations

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Adaptive low-pass filter (Casiez et al.).

    Cutoff rises with the signal's speed, so it is smooth when still and
    responsive when moving. State is per-instance; use one per scalar channel.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if dt <= 0:
            dt = 1e-6
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class KeypointSmoother:
    """One-Euro filter per keypoint (independent x and y channels)."""

    def __init__(self, num_keypoints: int = 17, min_cutoff: float = 1.0, beta: float = 0.007):
        self._fx = [OneEuroFilter(min_cutoff, beta) for _ in range(num_keypoints)]
        self._fy = [OneEuroFilter(min_cutoff, beta) for _ in range(num_keypoints)]
        self._last: np.ndarray | None = None

    def update(self, keypoints: np.ndarray, dt: float) -> np.ndarray:
        """Return a smoothed copy of ``(num_keypoints, 2)`` keypoints."""
        out = np.empty_like(keypoints, dtype=float)
        for i in range(len(keypoints)):
            out[i, 0] = self._fx[i](float(keypoints[i, 0]), dt)
            out[i, 1] = self._fy[i](float(keypoints[i, 1]), dt)
        self._last = out
        return out

    @property
    def last(self) -> np.ndarray | None:
        return self._last
