"""Public result type for the tracking layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedPerson:
    """A confirmed track emitted for one frame.

    Attributes:
        id: Stable integer identity for this person.
        box: Smoothed ``(4,)`` xyxy bounding box (Kalman filter output).
        keypoints: Smoothed ``(17, 2)`` keypoints, or ``None`` while coasting.
        scores: ``(17,)`` per-keypoint confidences, or ``None`` while coasting.
        coasting: True when the box is predicted (no detection this frame).
        color: Stable BGR color for this id.
    """

    id: int
    box: np.ndarray
    keypoints: np.ndarray | None
    scores: np.ndarray | None
    coasting: bool
    color: tuple[int, int, int]
