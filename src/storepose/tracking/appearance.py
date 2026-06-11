"""Appearance descriptors for re-id. v1: HSV torso-color histogram.

Injectable behind the AppearanceModel protocol so a stronger ReID embedding
(e.g. OSNet ONNX) can replace the histogram without touching the tracker.
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

# COCO keypoint indices for the torso quad.
_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 5, 6, 11, 12

_H_BINS, _S_BINS = 32, 32
_SAT_MIN, _VAL_MIN, _VAL_MAX = 40, 40, 250  # background/shadow/highlight mask (0-255)


class AppearanceModel(Protocol):
    """Extracts a per-person descriptor and scores descriptor similarity."""

    def extract(self, frame, box, keypoints, scores) -> np.ndarray | None: ...
    def similarity(self, a: np.ndarray | None, b: np.ndarray | None) -> float: ...


def _torso_rect(box, keypoints, scores, kpt_thr) -> tuple[int, int, int, int]:
    """Torso pixel rect from shoulder/hip keypoints, else the box's chest band."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if keypoints is not None and scores is not None:
        idx = (_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP)
        if all(float(scores[i]) >= kpt_thr for i in idx):
            xs = [float(keypoints[i][0]) for i in idx]
            ys = [float(keypoints[i][1]) for i in idx]
            if max(xs) > min(xs) and max(ys) > min(ys):
                return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    w, h = x2 - x1, y2 - y1
    cx = x1 + w / 2.0
    return (int(cx - w * 0.25), int(y1 + h * 0.15),
            int(cx + w * 0.25), int(y1 + h * 0.45))


class HsvHistogramAppearance:
    """Normalized H-S histogram of the masked torso crop; correlation similarity."""

    def __init__(self, kpt_thr: float = 0.5):
        self.kpt_thr = kpt_thr

    def extract(self, frame, box, keypoints, scores) -> np.ndarray | None:
        h_img, w_img = frame.shape[:2]
        rx1, ry1, rx2, ry2 = _torso_rect(box, keypoints, scores, self.kpt_thr)
        rx1 = max(0, min(rx1, w_img - 1))
        rx2 = max(0, min(rx2, w_img))
        ry1 = max(0, min(ry1, h_img - 1))
        ry2 = max(0, min(ry2, h_img))
        if rx2 - rx1 < 2 or ry2 - ry1 < 2:
            return None
        hsv = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        mask = ((s >= _SAT_MIN) & (v >= _VAL_MIN) & (v <= _VAL_MAX)).astype(np.uint8) * 255
        if not mask.any():
            return None
        hist = cv2.calcHist([hsv], [0, 1], mask, [_H_BINS, _S_BINS], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.astype(np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return -1.0
        return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
