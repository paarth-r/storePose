"""A single tracked person: identity, motion, lifecycle, and smoothing."""

from __future__ import annotations

import numpy as np

from .kalman import KalmanBoxTracker
from .smoothing import KeypointSmoother

# Distinct BGR colors cycled by track id.
_PALETTE: list[tuple[int, int, int]] = [
    (56, 56, 255), (56, 255, 56), (255, 56, 56), (56, 255, 255),
    (255, 56, 255), (255, 255, 56), (56, 153, 255), (255, 153, 56),
    (153, 56, 255), (56, 255, 153),
]


def color_for(track_id: int) -> tuple[int, int, int]:
    """Stable BGR color for a track id."""
    return _PALETTE[track_id % len(_PALETTE)]


class Track:
    """One person's track. Created from a detection, updated or coasted per frame."""

    def __init__(
        self,
        track_id: int,
        box: np.ndarray,
        keypoints: np.ndarray | None,
        scores: np.ndarray | None,
        dt: float,
        *,
        min_hits: int,
        smooth: bool,
        min_cutoff: float,
        beta: float,
    ):
        self.id = track_id
        self.kalman = KalmanBoxTracker(box)
        self.hits = 1
        self.time_since_update = 0
        self.min_hits = min_hits
        self.confirmed = min_hits <= 1
        self.color = color_for(track_id)
        self._smoother = (
            KeypointSmoother(min_cutoff=min_cutoff, beta=beta) if smooth else None
        )
        self.keypoints: np.ndarray | None = None
        self.scores: np.ndarray | None = None
        self._ingest_pose(keypoints, scores, dt)

    def _ingest_pose(self, keypoints, scores, dt) -> None:
        self.scores = scores
        if keypoints is None:
            return
        if self._smoother is not None:
            self.keypoints = self._smoother.update(keypoints, dt)
        else:
            self.keypoints = np.asarray(keypoints, float)

    def predict(self) -> None:
        """Advance motion; mark the track as coasting for this frame."""
        self.kalman.predict()
        self.time_since_update += 1

    def update(self, box, keypoints, scores, dt) -> None:
        """Correct with a matched detection."""
        self.kalman.update(box)
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self.min_hits:
            self.confirmed = True
        self._ingest_pose(keypoints, scores, dt)

    @property
    def box(self) -> np.ndarray:
        return self.kalman.box

    @property
    def coasting(self) -> bool:
        return self.time_since_update > 0
