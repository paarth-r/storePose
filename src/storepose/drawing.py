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

ZONE_COLOR = (0, 180, 255)   # line zone — orange
POS_COLOR = (70, 200, 60)    # Mashgin checkout — green
ALT_COLOR = (60, 60, 235)    # non-Mashgin checkout — red
IN_LINE_COLOR = (0, 200, 0)

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
FPS_COLOR = (0, 215, 255)


def _draw_boxes(frame: np.ndarray, boxes: np.ndarray,
                scores: np.ndarray | None = None, show_conf: bool = False) -> None:
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"person {i + 1}"
        if show_conf and scores is not None and i < len(scores):
            label += f"  {float(scores[i]):.2f}"
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
        _draw_boxes(canvas, result.boxes, result.det_scores, config.show_conf)

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
        if config.show_conf and p.score is not None:
            label += f"  {p.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), p.color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # RE-ID event: a brief "RE-ID <sim>" tag above the id label (fades after
        # ~1s, set by the tracker). Only when --conf is on.
        if config.show_conf and p.reid_notify and p.reid_sim is not None:
            rlabel = f"RE-ID {p.reid_sim:.2f}"
            (rw, rh), _ = cv2.getTextSize(rlabel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ry = y1 - th - 10  # sits just above the id label band
            cv2.rectangle(canvas, (x1, ry - rh - 4), (x1 + rw + 4, ry), p.color, -1)
            cv2.putText(canvas, rlabel, (x1 + 2, ry - 2), cv2.FONT_HERSHEY_SIMPLEX,
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


def _draw_zone(canvas: np.ndarray, zone: Zone, color: tuple[int, int, int]) -> None:
    """Fill (faint) and outline every contour of ``zone`` in ``color``."""
    for poly in zone.polygons:
        if len(poly) < 2:
            continue
        pts = np.array(poly, np.int32).reshape(-1, 1, 2)
        if len(poly) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
        cv2.polylines(canvas, [pts], True, color, 2)


def _draw_pos_panel(
    canvas: np.ndarray, people: list[TrackedPerson], result: QueueResult
) -> None:
    """Bottom panel listing each person at a checkout — Mashgin (green, POS) and
    non-Mashgin (red, REG) — with their serving time. Hidden when nobody serves."""
    rows = [(s, POS_COLOR, "POS") for s in result.statuses if s.serving]
    rows += [(s, ALT_COLOR, "REG") for s in result.statuses if s.serving_other]
    if not rows:
        return
    h, w = canvas.shape[:2]
    scale = max(1.0, w / 960.0)
    fs = 0.6 * scale
    thick = max(1, round(2 * scale))
    line_h = int(round(26 * scale))
    pad = int(round(8 * scale))
    panel_h = pad * 2 + line_h * len(rows)
    y0 = h - panel_h
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, (s, col, label) in enumerate(rows):
        y = y0 + pad + line_h * i + int(round(18 * scale))
        cv2.putText(canvas, f"AT {label}  -  ID {s.id}: {s.serving_seconds:0.1f}s",
                    (pad, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, thick, cv2.LINE_AA)


def annotate_queue(
    canvas: np.ndarray,
    people: list[TrackedPerson],
    result: QueueResult,
    zone: Zone | None,
    config: AppConfig,
    pos_zone: Zone | None = None,
    alt_zone: Zone | None = None,
) -> np.ndarray:
    """Overlay the line zone (orange), the Mashgin POS (green) and non-Mashgin
    checkout (red), a ``WAIT`` / ``POS`` / ``REG`` tag per person, and the live
    ``in line / at POS / at REG`` counts. Draws in place and returns ``canvas``."""
    if zone is not None:
        _draw_zone(canvas, zone, ZONE_COLOR)
    if pos_zone is not None:
        _draw_zone(canvas, pos_zone, POS_COLOR)
    if alt_zone is not None:
        _draw_zone(canvas, alt_zone, ALT_COLOR)

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
        elif s.serving or s.serving_other:
            # at a checkout: green (Mashgin) or red (non-Mashgin) fill + timer
            col = POS_COLOR if s.serving else ALT_COLOR
            label = "POS" if s.serving else "REG"
            txt = (0, 0, 0) if s.serving else (255, 255, 255)
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
            tag = f"{label} {s.serving_seconds:0.1f}s"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = min(y2 + th + 6, canvas.shape[0] - 2)
            cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty), col, -1)
            cv2.putText(canvas, tag, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, txt, 2, cv2.LINE_AA)
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

    header = f"in line: {result.count}   at POS: {result.serving_count}"
    if alt_zone is not None:
        header += f"   at REG: {result.serving_other_count}"
    cv2.putText(canvas, header, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ZONE_COLOR, 2, cv2.LINE_AA)
    _draw_pos_panel(canvas, people, result)
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
