"""Person detection stage (YOLOX via rtmlib)."""

from __future__ import annotations

import numpy as np
from rtmlib import YOLOX

from .config import AppConfig
from .model_zoo import ModelSpec


class PersonDetector:
    """Detects people and returns their bounding boxes.

    Wraps rtmlib's human-mode YOLOX, which returns an ``(N, 4)`` array of
    ``xyxy`` boxes (one row per detected person).
    """

    def __init__(self, spec: ModelSpec, config: AppConfig):
        self._model = YOLOX(
            spec.url,
            model_input_size=spec.input_size,
            mode="human",
            score_thr=config.det_conf,
            backend="onnxruntime",
            device=config.device,
        )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return person boxes as an ``(N, 4)`` float array of ``xyxy``."""
        boxes = self._model(frame)
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        return boxes.reshape(-1, 4)
