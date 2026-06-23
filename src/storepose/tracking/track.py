"""A single tracked person: identity, motion, lifecycle, and smoothing."""

from __future__ import annotations

import numpy as np

from .kalman import KalmanBoxTracker
from .smoothing import KeypointSmoother

# Distinct BGR colors cycled by track id. No near-orange entry, so a person's
# color never collides with the orange queue zone fill (drawing.ZONE_COLOR).
_PALETTE: list[tuple[int, int, int]] = [
    (56, 56, 255), (56, 255, 56), (255, 56, 56), (56, 200, 200),
    (255, 56, 255), (255, 255, 56), (255, 153, 56), (255, 56, 153),
    (153, 56, 255), (56, 255, 153),
]


def color_for(track_id: int) -> tuple[int, int, int]:
    """Stable BGR color for a track id."""
    return _PALETTE[track_id % len(_PALETTE)]


# How long (seconds) a "RE-ID <sim>" notification stays armed after a re-attach.
REID_NOTIF_SECONDS = 1.0


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
        appearance_mem: object | None = None,
        det_score: float | None = None,
        drift: bool = True,
    ):
        self.id = track_id
        self.det_score = det_score  # detector confidence of the last detection
        self.kalman = KalmanBoxTracker(box)
        self._drift = drift  # False: a coasting track's box stays at last_box
        self.last_box = np.asarray(box, float)  # last *detected* box (not coasted)
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
        self.appearance_mem = appearance_mem  # opaque; owned by the AppearanceModel
        self.reid_sim: float | None = None     # similarity of the last re-attach
        self.reid_time_left = 0.0              # seconds the RE-ID notif stays armed
        self._ingest_pose(keypoints, scores, dt)

    def _ingest_pose(self, keypoints, scores, dt) -> None:
        self.scores = scores
        if keypoints is None:
            return
        if self._smoother is not None:
            self.keypoints = self._smoother.update(keypoints, dt)
        else:
            self.keypoints = np.asarray(keypoints, float)

    def predict(self, dt: float = 0.0) -> None:
        """Advance motion; mark the track as coasting for this frame.

        ``dt`` (seconds since the last frame) ages the RE-ID notification so it
        fades after ``REID_NOTIF_SECONDS`` regardless of frame rate.
        """
        self.kalman.predict()
        self.time_since_update += 1
        if self.reid_time_left > 0.0:
            self.reid_time_left = max(0.0, self.reid_time_left - dt)

    def update(self, box, keypoints, scores, dt, appearance_mem=None, det_score=None) -> None:
        """Correct with a matched detection."""
        self.kalman.update(box)
        self.last_box = np.asarray(box, float)
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self.min_hits:
            self.confirmed = True
        self.det_score = det_score
        self._ingest_pose(keypoints, scores, dt)
        if appearance_mem is not None:
            self.appearance_mem = appearance_mem

    def reactivate(self, box, keypoints, scores, dt, appearance_mem=None,
                   det_score=None, reid_sim=None) -> None:
        """Revive a lost/coasting track at a new detection, keeping id and color.

        ``reid_sim`` (appearance similarity that drove the re-attach) arms the
        RE-ID notification for ``REID_NOTIF_SECONDS``.
        """
        self.kalman = KalmanBoxTracker(box)
        self.last_box = np.asarray(box, float)
        self.time_since_update = 0
        self.hits += 1
        self.confirmed = True
        self.det_score = det_score
        if reid_sim is not None:
            self.reid_sim = reid_sim
            self.reid_time_left = REID_NOTIF_SECONDS
        self._ingest_pose(keypoints, scores, dt)
        if appearance_mem is not None:
            self.appearance_mem = appearance_mem

    @property
    def box(self) -> np.ndarray:
        # With predictive drift off, a coasting track holds its last detected
        # position instead of extrapolating along Kalman velocity (which would
        # drift the box away from the person and cause mis-association/swaps).
        if not self._drift and self.coasting:
            return self.last_box
        return self.kalman.box

    @property
    def coasting(self) -> bool:
        return self.time_since_update > 0
