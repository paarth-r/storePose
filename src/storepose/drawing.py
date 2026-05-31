"""Frame annotation: bounding boxes, skeletons, and FPS overlay."""

from __future__ import annotations

import cv2
import numpy as np
from rtmlib import draw_skeleton

from .config import AppConfig
from .pipeline import FrameResult

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
FPS_COLOR = (0, 215, 255)


def _draw_boxes(frame: np.ndarray, boxes: np.ndarray) -> None:
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"person {i + 1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def annotate(
    frame: np.ndarray,
    result: FrameResult,
    config: AppConfig,
    fps: float | None = None,
) -> np.ndarray:
    """Draw boxes, skeletons, and (optionally) FPS onto a copy of ``frame``.

    Reuses rtmlib's :func:`draw_skeleton` for the correct COCO-17 topology and
    per-keypoint confidence gating (``kpt_thr``). Safe on empty results.
    """
    canvas = frame.copy()

    if result.count > 0:
        canvas = draw_skeleton(
            canvas,
            result.keypoints,
            result.scores,
            kpt_thr=config.kpt_thr,
            radius=3,
            line_width=2,
        )
        _draw_boxes(canvas, result.boxes)

    header = f"people: {result.count}"
    if config.show_fps and fps is not None:
        header += f"   fps: {fps:4.1f}"
    cv2.putText(
        canvas,
        header,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        FPS_COLOR,
        2,
        cv2.LINE_AA,
    )
    return canvas
