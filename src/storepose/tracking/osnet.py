"""OSNet ReID embedding appearance model (learned, replaces the histogram).

Implements the AppearanceModel seam with a per-track feature gallery scored by
max cosine similarity (min cosine distance) -- the DeepSORT/StrongSORT standard.
Weights are auto-downloaded ONNX (see reid_zoo); --reid-weights overrides.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

_INPUT_H, _INPUT_W = 256, 128
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
_GALLERY_K = 10  # embeddings retained per track
_MIN_CROP = 4    # reject crops smaller than this (px) on either side


def _providers(device: str) -> list[str]:
    if device == "mps":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OsnetAppearance:
    """Full-body crop -> OSNet ONNX -> L2-normalized embedding; gallery memory."""

    def __init__(self, weights_path: str, device: str = "cpu"):
        self._session = ort.InferenceSession(
            weights_path, providers=_providers(device)
        )
        self._input = self._session.get_inputs()[0].name

    def _crop(self, frame, box):
        h, w = frame.shape[:2]
        x1 = max(0, min(int(box[0]), w - 1))
        x2 = max(0, min(int(box[2]), w))
        y1 = max(0, min(int(box[1]), h - 1))
        y2 = max(0, min(int(box[3]), h))
        if x2 - x1 < _MIN_CROP or y2 - y1 < _MIN_CROP:
            return None
        return frame[y1:y2, x1:x2]

    def _preprocess(self, crop):
        img = cv2.resize(crop, (_INPUT_W, _INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return (img - _MEAN) / _STD

    def extract(self, frame, box, keypoints, scores):
        return self.extract_batch(frame, [box], [keypoints], [scores])[0]

    def extract_batch(self, frame, boxes, keypoints, scores):
        out: list[np.ndarray | None] = [None] * len(boxes)
        crops, idx = [], []
        for i, box in enumerate(boxes):
            c = self._crop(frame, box)
            if c is not None:
                crops.append(self._preprocess(c))
                idx.append(i)
        if not crops:
            return out
        batch = np.stack(crops).astype(np.float32)
        emb = np.asarray(self._session.run(None, {self._input: batch})[0], np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        for j, i in enumerate(idx):
            out[i] = emb[j]
        return out

    def new_memory(self, desc):
        g: deque = deque(maxlen=_GALLERY_K)
        if desc is not None:
            g.append(np.asarray(desc, np.float32))
        return g

    def update_memory(self, mem, desc):
        if mem is None:
            mem = deque(maxlen=_GALLERY_K)
        if desc is not None:
            mem.append(np.asarray(desc, np.float32))
        return mem

    def score(self, mem, desc) -> float:
        if mem is None or len(mem) == 0 or desc is None:
            return -1.0
        d = np.asarray(desc, np.float32)
        return float(max(float(np.dot(e, d)) for e in mem))
