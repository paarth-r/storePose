"""Person detection stage (YOLOX via rtmlib)."""

from __future__ import annotations

import numpy as np
from rtmlib import YOLOX

from .config import AppConfig
from .model_zoo import ModelSpec


def suppress_contained_boxes(boxes: np.ndarray, thr: float) -> np.ndarray:
    """Drop a box that is more than ``thr`` *contained* within a larger box.

    Containment = intersection / area-of-the-smaller-box. Unlike IoU-NMS this
    targets duplicate boxes nested on one person (a partial sub-box sits mostly
    inside the full-person box) without merging two distinct people standing
    close, whose boxes overlap but neither contains the other. The larger box of
    a colliding pair is kept; surviving boxes keep their original order.
    """
    n = len(boxes)
    if n <= 1 or thr >= 1.0:
        return boxes
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)  # largest first, so a kept box is never smaller
    kept: list[int] = []
    for i in order:
        b = boxes[i]
        contained = False
        for k in kept:
            a = boxes[k]
            x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
            x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if areas[i] > 0 and inter / areas[i] > thr:
                contained = True
                break
        if not contained:
            kept.append(i)
    return boxes[sorted(kept)]


class PersonDetector:
    """Detects people and returns their bounding boxes.

    Wraps rtmlib's human-mode YOLOX, which returns an ``(N, 4)`` array of
    ``xyxy`` boxes (one row per detected person).
    """

    def __init__(self, spec: ModelSpec, config: AppConfig):
        self._overlap = config.det_overlap
        self._model = YOLOX(
            spec.url,
            model_input_size=spec.input_size,
            mode="human",
            score_thr=config.det_conf,
            backend="onnxruntime",
            device=config.device,
        )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return person boxes as an ``(N, 4)`` float array of ``xyxy``.

        Overlapping duplicate boxes nested on one person are suppressed (see
        :func:`suppress_contained_boxes`) so each person yields a single box.
        """
        boxes = self._model(frame)
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        return suppress_contained_boxes(boxes.reshape(-1, 4), self._overlap)
