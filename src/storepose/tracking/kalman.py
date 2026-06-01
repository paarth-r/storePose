"""SORT-style constant-velocity Kalman filter for a single box."""

from __future__ import annotations

import numpy as np


def _box_to_z(box: np.ndarray) -> np.ndarray:
    """``xyxy`` -> measurement ``[cx, cy, area, aspect]`` as a ``(4, 1)``."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w / 2.0
    cy = box[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 0.0
    return np.array([[cx], [cy], [s], [r]], dtype=float)


def _x_to_box(x: np.ndarray) -> np.ndarray:
    """State -> ``xyxy`` box ``(4,)``."""
    cx, cy, s, r = float(x[0, 0]), float(x[1, 0]), float(x[2, 0]), float(x[3, 0])
    w = np.sqrt(max(s * r, 1e-6))
    h = s / w if w > 0 else 0.0
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], float)


class KalmanBoxTracker:
    """Tracks one box with state ``[cx, cy, area, aspect, vx, vy, v_area]``."""

    def __init__(self, box: np.ndarray):
        self.F = np.eye(7)
        for i in range(3):  # position components gain a velocity term
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 7))
        for i in range(4):
            self.H[i, i] = 1.0

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # high uncertainty on unobserved velocities
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[6, 6] *= 0.01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0  # area/aspect are noisier measurements

        self.x = np.zeros((7, 1))
        self.x[:4] = _box_to_z(box)

    def predict(self) -> np.ndarray:
        """Advance the state by one step; returns the predicted box."""
        if (self.x[6] + self.x[2])[0] <= 0:  # keep area non-negative
            self.x[6] *= 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box

    def update(self, box: np.ndarray) -> np.ndarray:
        """Correct the state with a measured box; returns the filtered box."""
        z = _box_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        return self.box

    @property
    def box(self) -> np.ndarray:
        return _x_to_box(self.x)
