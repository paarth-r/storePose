"""Frame annotation: bounding boxes, skeletons, and FPS overlay."""

from __future__ import annotations

import cv2
import numpy as np
from rtmlib import draw_skeleton

from .config import AppConfig
from .pipeline import FrameResult
from .queue.types import QueueResult
from .queue.zone import Zone
from .tracking.types import TrackedPerson

ZONE_COLOR = (0, 180, 255)
IN_LINE_COLOR = (0, 200, 0)

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


def annotate_tracked(
    frame: np.ndarray,
    people: list[TrackedPerson],
    config: AppConfig,
    fps: float | None = None,
) -> np.ndarray:
    """Draw tracked people: stable-color box + ``ID n`` label, skeleton when
    not coasting. Safe on an empty list and on coasting (poseless) people."""
    canvas = frame.copy()

    for p in people:
        x1, y1, x2, y2 = (int(round(v)) for v in p.box[:4])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), p.color, 2)
        label = f"ID {p.id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), p.color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if p.keypoints is not None and p.scores is not None:
            canvas = draw_skeleton(
                canvas, p.keypoints[None, ...], p.scores[None, ...],
                kpt_thr=config.kpt_thr, radius=3, line_width=2,
            )

    header = f"people: {len(people)}"
    if config.show_fps and fps is not None:
        header += f"   fps: {fps:4.1f}"
    cv2.putText(canvas, header, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                FPS_COLOR, 2, cv2.LINE_AA)
    return canvas


def annotate_queue(
    canvas: np.ndarray,
    people: list[TrackedPerson],
    result: QueueResult,
    zone: Zone | None,
    config: AppConfig,
) -> np.ndarray:
    """Overlay the queue zone, a ``WAIT n.n s`` tag on each waiting person, and
    a live ``in line: N`` count. Draws in place and returns ``canvas``."""
    if zone is not None and len(zone.points) >= 2:
        pts = np.array(zone.points, np.int32).reshape(-1, 1, 2)
        if len(zone.points) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], ZONE_COLOR)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
        cv2.polylines(canvas, [pts], True, ZONE_COLOR, 2)

    status_by_id = {s.id: s for s in result.statuses}
    for p in people:
        s = status_by_id.get(p.id)
        if s is None:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in p.box[:4])

        if s.waiting:
            # solid translucent fill over the whole box, in the person's color
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), p.color, -1)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
            tag = f"WAIT {s.wait_seconds:0.1f}s"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = min(y2 + th + 6, canvas.shape[0] - 2)
            cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty), p.color, -1)
            cv2.putText(canvas, tag, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 2, cv2.LINE_AA)
        elif s.candidate:
            # "sheer" fill rising from the bottom as inclusion progresses
            fill_h = int(round((y2 - y1) * max(0.0, min(s.progress, 1.0))))
            fy1 = y2 - fill_h
            if fill_h > 0:
                overlay = canvas.copy()
                cv2.rectangle(overlay, (x1, fy1), (x2, y2), p.color, -1)
                cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
            pct = f"{int(round(s.progress * 100))}%"
            cv2.putText(canvas, pct, (x1 + 3, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, p.color, 2, cv2.LINE_AA)

    cv2.putText(canvas, f"in line: {result.count}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ZONE_COLOR, 2, cv2.LINE_AA)
    return canvas


# BGR badge color per busy level.
_BUSY_COLORS = {"Low": (0, 200, 0), "Medium": (0, 200, 255), "High": (0, 0, 255)}


def annotate_busy(
    canvas: np.ndarray, level_label: str, metric_value: float, window_remaining_s: float
) -> np.ndarray:
    """Top-right badge showing the *current window's* live busy estimate.

    This is the running label for the in-progress window, not a finalized one,
    so it can change as the window fills; the authoritative per-window labels are
    written to the busy report at the end."""
    color = _BUSY_COLORS.get(level_label, (200, 200, 200))
    text = f"BUSY: {level_label.upper()}"
    sub = f"~{metric_value:.1f}  next in {window_remaining_s:0.0f}s"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    (sw, _sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    w = max(tw, sw) + 16
    x0 = canvas.shape[1] - w - 10
    cv2.rectangle(canvas, (x0, 10), (x0 + w, 10 + th + 28), (0, 0, 0), -1)
    cv2.rectangle(canvas, (x0, 10), (x0 + w, 10 + th + 28), color, 2)
    cv2.putText(canvas, text, (x0 + 8, 10 + th + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, sub, (x0 + 8, 10 + th + 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas
