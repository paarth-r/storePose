"""Face anonymization: pixelate each person's face as a post-process step.

Localizes the face from the COCO-17 face keypoints (nose, eyes, ears) when at
least two are visible; otherwise falls back to the top quarter of the person's
bounding box. Applied to the finished canvas so both the live window and any
recorded mp4 are anonymized.
"""
from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

# COCO-17 face keypoints: nose, left/right eye, left/right ear.
FACE_KEYPOINTS = (0, 1, 2, 3, 4)


def face_region(
    box: np.ndarray,
    keypoints: np.ndarray | None,
    scores: np.ndarray | None,
    kpt_thr: float,
    frame_w: int,
    frame_h: int,
    pad: float = 0.6,
) -> tuple[int, int, int, int] | None:
    """Return an integer ``(x1, y1, x2, y2)`` face region clamped to the frame.

    Prefers the extent of visible face keypoints (>= 2 scoring above
    ``kpt_thr``), padded outward to cover forehead and chin. Falls back to the
    top quarter of ``box`` when keypoints are missing or too sparse. Returns
    ``None`` for a degenerate (empty) region.
    """
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    region: tuple[float, float, float, float] | None = None

    if keypoints is not None and scores is not None:
        pts = [keypoints[i] for i in FACE_KEYPOINTS
               if i < len(scores) and float(scores[i]) >= kpt_thr]
        if len(pts) >= 2:
            arr = np.asarray(pts, dtype=float)
            fx1, fy1 = float(arr[:, 0].min()), float(arr[:, 1].min())
            fx2, fy2 = float(arr[:, 0].max()), float(arr[:, 1].max())
            margin = pad * max(fx2 - fx1, fy2 - fy1) + 1.0
            # extend further up for forehead/hair than down for chin
            region = (fx1 - margin, fy1 - margin * 1.3, fx2 + margin, fy2 + margin)

    if region is None:
        region = (x1, y1, x2, y1 + (y2 - y1) * 0.25)

    rx1 = int(max(0, min(region[0], frame_w)))
    ry1 = int(max(0, min(region[1], frame_h)))
    rx2 = int(max(0, min(region[2], frame_w)))
    ry2 = int(max(0, min(region[3], frame_h)))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return (rx1, ry1, rx2, ry2)


def _mosaic(canvas: np.ndarray, region: tuple[int, int, int, int], blocks: int) -> None:
    """Pixelate ``region`` of ``canvas`` in place into roughly ``blocks`` cells."""
    x1, y1, x2, y2 = region
    roi = canvas[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    bw = max(1, w // blocks)
    bh = max(1, h // blocks)
    small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
    canvas[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def blur_faces(
    canvas: np.ndarray,
    faces: Iterable[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]],
    kpt_thr: float,
    *,
    blocks: int = 12,
) -> np.ndarray:
    """Pixelate each person's face on ``canvas`` in place.

    ``faces`` is an iterable of ``(box, keypoints, scores)`` triples; keypoints
    and scores may be ``None`` (e.g. a coasting track), in which case the box's
    top quarter is used. Returns ``canvas``.
    """
    h, w = canvas.shape[:2]
    for box, keypoints, scores in faces:
        region = face_region(box, keypoints, scores, kpt_thr, w, h)
        if region is not None:
            _mosaic(canvas, region, blocks)
    return canvas
