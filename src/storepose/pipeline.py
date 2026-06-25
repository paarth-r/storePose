"""Composition of detection + pose into a single per-frame step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AppConfig
from .detector import PersonDetector
from .model_zoo import resolve
from .pose import PoseEstimator
from .queue.zone import Zone


@dataclass(frozen=True)
class FrameResult:
    """Detections for a single frame.

    Attributes:
        boxes: ``(N, 4)`` person boxes in ``xyxy``.
        keypoints: ``(N, 17, 2)`` keypoint coordinates.
        scores: ``(N, 17)`` per-keypoint confidences.
        det_scores: ``(N,)`` per-person detector confidence, aligned to ``boxes``.
    """

    boxes: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray
    det_scores: np.ndarray

    @property
    def count(self) -> int:
        """Number of people detected in the frame."""
        return int(len(self.boxes))


class PosePipeline:
    """Runs person detection then pose estimation on each frame.

    ``detector`` and ``pose`` are injectable for testing; by default they are
    built from the rtmlib model zoo for the configured mode/device.
    """

    def __init__(
        self,
        config: AppConfig,
        detector: PersonDetector | None = None,
        pose: PoseEstimator | None = None,
    ):
        spec = resolve(config.mode)
        self._detector = detector or PersonDetector(spec.detector, config)
        self._pose = pose or PoseEstimator(spec.pose, config)
        self._ignore_zone = Zone.load(config.ignore_zone) if config.ignore_zone else None

    def _drop_ignored(self, boxes, det_scores):
        """Remove detections whose box center is inside the ignore zone.

        Masks a fixed prop at the source: it never reaches pose or the tracker,
        so it cannot be tracked as a person or hand its id to a passer-by."""
        if self._ignore_zone is None or len(boxes) == 0:
            return boxes, det_scores
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        keep = np.array(
            [not self._ignore_zone.contains((x, y)) for x, y in zip(cx, cy)],
            dtype=bool,
        )
        return boxes[keep], det_scores[keep]

    def process(self, frame: np.ndarray) -> FrameResult:
        """Detect people and estimate their poses for ``frame``."""
        boxes, det_scores = self._detector.detect(frame)
        boxes, det_scores = self._drop_ignored(boxes, det_scores)
        keypoints, scores = self._pose.estimate(frame, boxes)
        return FrameResult(boxes=boxes, keypoints=keypoints, scores=scores,
                           det_scores=det_scores)
