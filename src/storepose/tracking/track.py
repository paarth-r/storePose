"""A single tracked person: identity, motion, lifecycle, and smoothing."""

from __future__ import annotations

from collections import deque

import numpy as np

from .kalman import KalmanBoxTracker
from .smoothing import KeypointSmoother

# Cap how far back detection centers are retained for the stationary test.
_STATIONARY_HISTORY_SECONDS = 60.0

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
        self._t = 0.0                          # cumulative track age (seconds)
        self._centers: deque = deque()         # (t, cx, cy) of detected boxes
        self._record_center(box)
        self._ingest_pose(keypoints, scores, dt)

    def _record_center(self, box) -> None:
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
        self._centers.append((self._t, cx, cy))

    def stationary(self, window: float, radius: float) -> bool:
        """True if the detected center stayed within ``radius`` px over the last
        ``window`` seconds — and we have a full window of history to judge.

        Used to suppress fixed props (a track that never moves). A real person
        who shifts beyond ``radius`` within the window resets it, so only a
        genuinely motionless object is flagged.
        """
        if window <= 0 or not self._centers:
            return False
        # Window ends at the most recent *detection*, not "now": a prop that has
        # since been coasting (no new detections) must still read as stationary
        # at cull time, when it would otherwise enter the re-id gallery.
        last_t = self._centers[-1][0]
        cutoff = last_t - window
        pts = [(t, x, y) for (t, x, y) in self._centers if t >= cutoff]
        if not pts or (last_t - pts[0][0]) < window:
            return False  # not enough detection history to judge
        xs = [x for _, x, _ in pts]
        ys = [y for _, _, y in pts]
        span = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
        return span <= radius

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
        self._t += dt
        if self.reid_time_left > 0.0:
            self.reid_time_left = max(0.0, self.reid_time_left - dt)
        cutoff = self._t - _STATIONARY_HISTORY_SECONDS
        while self._centers and self._centers[0][0] < cutoff:
            self._centers.popleft()

    def update(self, box, keypoints, scores, dt, appearance_mem=None, det_score=None) -> None:
        """Correct with a matched detection."""
        self.kalman.update(box)
        self.last_box = np.asarray(box, float)
        self._record_center(box)
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
        self._record_center(box)
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
