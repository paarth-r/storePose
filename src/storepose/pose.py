"""Pose estimation stage (RTMPose via rtmlib)."""

from __future__ import annotations

import numpy as np
from rtmlib import RTMPose

from .config import AppConfig
from .model_zoo import ModelSpec

NUM_KEYPOINTS = 17  # COCO body


def _empty_pose() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, NUM_KEYPOINTS, 2), dtype=np.float32),
        np.empty((0, NUM_KEYPOINTS), dtype=np.float32),
    )


class PoseEstimator:
    """Estimates a 17-keypoint COCO skeleton for each person box.

    The ``model`` is injectable to keep the empty-box short-circuit unit
    testable without loading real weights.
    """

    def __init__(self, spec: ModelSpec, config: AppConfig, model=None):
        self._model = model or RTMPose(
            spec.url,
            model_input_size=spec.input_size,
            to_openpose=False,
            backend="onnxruntime",
            device=config.device,
        )

    def estimate(
        self, frame: np.ndarray, boxes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(keypoints (N,17,2), scores (N,17))`` for ``boxes``.

        Short-circuits to empty arrays when there are no boxes; rtmlib would
        otherwise pose-estimate the entire frame, producing a phantom skeleton.
        """
        if boxes is None or len(boxes) == 0:
            return _empty_pose()
        keypoints, scores = self._model(frame, bboxes=boxes)
        return np.asarray(keypoints), np.asarray(scores)
