"""Person detection stage (YOLOX via rtmlib)."""

from __future__ import annotations

import numpy as np
from rtmlib import YOLOX
from rtmlib.tools.object_detection.yolox import multiclass_nms

from .config import AppConfig
from .model_zoo import ModelSpec


class _ScoredYOLOX(YOLOX):
    """YOLOX that keeps the detection scores rtmlib's ``human`` mode drops.

    Stock ``YOLOX.postprocess`` computes ``final_scores`` but returns only the
    boxes in human mode. This override mirrors that logic (pinned rtmlib version)
    and returns ``(final_boxes, final_scores)`` so we can overlay per-person
    confidence. Box/score order is preserved and kept aligned.
    """

    def postprocess(self, outputs, ratio=1.0):  # noqa: C901 - mirrors rtmlib
        if outputs.shape[-1] == 4 or outputs.shape[-1] > 5:  # onnx without nms
            grids, expanded_strides = [], []
            strides = [8, 16, 32]
            hsizes = [self.model_input_size[0] // s for s in strides]
            wsizes = [self.model_input_size[1] // s for s in strides]
            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)
            outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
            outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

            predictions = outputs[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]
            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
            boxes_xyxy /= ratio
            dets, _ = multiclass_nms(boxes_xyxy, scores,
                                     nms_thr=self.nms_thr, score_thr=self.score_thr)
            if dets is not None:
                final_boxes, final_scores = dets[:, :4], dets[:, 4]
                keep = final_scores > self.nms_thr
                final_boxes, final_scores = final_boxes[keep], final_scores[keep]
            else:
                final_boxes = np.empty((0, 4), dtype=np.float32)
                final_scores = np.empty((0,), dtype=np.float32)
        elif outputs.shape[-1] == 5:  # onnx contains nms module
            final_boxes, final_scores = outputs[0, :, :4], outputs[0, :, 4]
            final_boxes = final_boxes / ratio
            keep = final_scores > self.score_thr  # honor the configured threshold
            final_boxes, final_scores = final_boxes[keep], final_scores[keep]
        else:
            raise NotImplementedError(f"unexpected YOLOX output shape {outputs.shape}")
        return final_boxes, final_scores


def _contained_keep_indices(boxes: np.ndarray, thr: float) -> list[int]:
    """Indices of boxes to keep after duplicate-on-one-person suppression.

    Containment = intersection / area-of-the-smaller-box. Unlike IoU-NMS this
    targets duplicate boxes nested on one person (a partial sub-box sits mostly
    inside the full-person box) without merging two distinct people standing
    close, whose boxes overlap but neither contains the other. The larger box of
    a colliding pair is kept; returned indices are in original order.
    """
    n = len(boxes)
    if n <= 1 or thr >= 1.0:
        return list(range(n))
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
    return sorted(kept)


def suppress_contained_boxes(boxes: np.ndarray, thr: float) -> np.ndarray:
    """Drop a box more than ``thr`` *contained* within a larger one.

    See :func:`_contained_keep_indices` for the containment rule.
    """
    return boxes[_contained_keep_indices(boxes, thr)]


def filter_confident(
    boxes: np.ndarray, scores: np.ndarray, conf: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only boxes whose score is ``>= conf`` (scores stay aligned).

    Applied in :meth:`PersonDetector.detect` so the configured confidence is
    always honored, regardless of any threshold baked into the ONNX export.
    """
    if boxes.size == 0:
        return boxes, scores
    mask = scores >= conf
    return boxes[mask], scores[mask]


class PersonDetector:
    """Detects people and returns their bounding boxes.

    Wraps rtmlib's human-mode YOLOX, which returns an ``(N, 4)`` array of
    ``xyxy`` boxes (one row per detected person).
    """

    def __init__(self, spec: ModelSpec, config: AppConfig):
        self._overlap = config.det_overlap
        self._conf = config.det_conf
        self._model = _ScoredYOLOX(
            spec.url,
            model_input_size=spec.input_size,
            mode="human",
            score_thr=config.det_conf,
            backend="onnxruntime",
            device=config.device,
        )

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(boxes (N,4) xyxy, scores (N,))`` for detected people.

        Overlapping duplicate boxes nested on one person are suppressed (see
        :func:`suppress_contained_boxes`) so each person yields a single box;
        ``scores`` stays aligned to the surviving boxes.
        """
        boxes, scores = self._model(frame)
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if boxes.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        # Honor the configured confidence regardless of the ONNX's baked-in
        # threshold (some exports embed NMS at a fixed low score floor).
        boxes, scores = filter_confident(boxes, scores, self._conf)
        if boxes.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        idx = _contained_keep_indices(boxes, self._overlap)
        return boxes[idx], scores[idx]
