"""Composition of detection + pose into a single per-frame step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AppConfig
from .detector import PersonDetector
from .model_zoo import resolve
from .pose import PoseEstimator


@dataclass(frozen=True)
class FrameResult:
    """Detections for a single frame.

    Attributes:
        boxes: ``(N, 4)`` person boxes in ``xyxy``.
        keypoints: ``(N, 17, 2)`` keypoint coordinates.
        scores: ``(N, 17)`` per-keypoint confidences.
    """

    boxes: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray

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

    def process(self, frame: np.ndarray) -> FrameResult:
        """Detect people and estimate their poses for ``frame``."""
        boxes = self._detector.detect(frame)
        keypoints, scores = self._pose.estimate(frame, boxes)
        return FrameResult(boxes=boxes, keypoints=keypoints, scores=scores)
