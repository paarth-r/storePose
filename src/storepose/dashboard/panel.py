"""Render the live dashboard as a cv2 image and composite it beside the video.

The Chrome dashboard is HTML/JS; to bake it into a saved recording we redraw the
*same* numbers (the ``metrics`` summaries the page polls) as a dark side panel,
then place the annotated video on the left 3/4 and this panel on the right 1/4 of
one composite frame. No browser/headless dependency: it stays in-process, fast,
and frame-synced. Styling mirrors the dark Mashgin dashboard palette.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import metrics

# Dashboard palette (page.py CSS vars) as BGR for cv2.
_CANVAS = (23, 14, 10)     # --canvas  #0a0e17
_PAPER = (38, 24, 18)      # --paper   #121826
_SUNKEN = (33, 20, 14)     # --sunken  #0e1421
_INK = (247, 238, 234)     # --ink     #eaeef7
_BRAND = (255, 120, 88)    # --brand   #5878ff
_MUTED = (169, 148, 139)   # --muted   #8b94a9
_HAIR = (63, 44, 35)       # --hair    #232c3f
_LOW = (153, 211, 52)      # #34d399
_MED = (36, 191, 251)      # #fbbf24
_HIGH = (133, 113, 251)    # #fb7185

_BUSY_COLOR = {"Low": _LOW, "Medium": _MED, "High": _HIGH}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class PanelData:
    """The dashboard numbers a panel needs — the same metrics the page shows."""

    busy: tuple | None          # (t, level, value) or None
    summary: dict               # metrics.summary_stats output
    checkouts: dict             # metrics.checkout_stats output
    occ: list                   # raw occupancy samples (t, waiting, serving)
    show_alt: bool = False      # include the AT REG / staffed-checkout rows


def panel_data(dash_state, show_alt: bool = False) -> PanelData:
    """Snapshot a :class:`DashboardState` into the numbers a panel renders."""
    occ, visits = dash_state.snapshot()
    busy_current, _ = dash_state.busy_snapshot()
    return PanelData(
        busy=busy_current,
        summary=metrics.summary_stats(occ, visits),
        checkouts=metrics.checkout_stats(visits),
        occ=occ,
        show_alt=show_alt,
    )


def _put(img, text, org, scale, color, thick=1):
    cv2.putText(img, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


def _put_right(img, text, right_x, y, scale, color, thick=1):
    (tw, _), _ = cv2.getTextSize(text, _FONT, scale, thick)
    _put(img, text, (right_x - tw, y), scale, color, thick)


def _sparkline(img, occ, x0, y0, w, h):
    """Tiny waiting-count line chart in the box (x0, y0, w, h)."""
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), _SUNKEN, -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), _HAIR, 1)
    waiting = [s[1] for s in occ][-w:]            # at most one sample per column
    if len(waiting) < 2:
        return
    hi = max(max(waiting), 1)
    n = len(waiting)
    pts = []
    for i, v in enumerate(waiting):
        px = x0 + round(i * (w - 1) / (n - 1))
        py = y0 + h - 1 - round(v / hi * (h - 2))
        pts.append((px, py))
    cv2.polylines(img, [np.array(pts, np.int32)], False, _BRAND, 1, cv2.LINE_AA)


def render_panel(width: int, height: int, data: PanelData) -> np.ndarray:
    """Draw the dashboard side panel at ``(height, width)`` as a BGR image."""
    img = np.empty((height, width, 3), np.uint8)
    img[:] = _CANVAS
    s = width / 320.0                              # scale relative to a 320px panel
    m = round(18 * s)                              # left/right margin
    rx = width - m                                 # right edge for right-aligned text
    y = round(34 * s)

    # --- brand header ---
    _put(img, "storePose", (m, y), 0.72 * s, _INK, max(1, round(2 * s)))
    y += round(20 * s)
    _put(img, "LINE MONITOR", (m, y), 0.42 * s, _MUTED, 1)
    y += round(14 * s)
    cv2.line(img, (m, y), (rx, y), _HAIR, 1)
    y += round(30 * s)

    # --- busy badge ---
    level = data.busy[1] if data.busy else None
    value = data.busy[2] if data.busy else 0.0
    color = _BUSY_COLOR.get(level, _MUTED)
    bh = round(58 * s)
    cv2.rectangle(img, (m, y - round(26 * s)), (rx, y - round(26 * s) + bh), _PAPER, -1)
    cv2.rectangle(img, (m, y - round(26 * s)), (rx, y - round(26 * s) + bh), color, max(1, round(2 * s)))
    _put(img, "BUSY", (m + round(10 * s), y - round(8 * s)), 0.4 * s, _MUTED, 1)
    _put(img, (level or "--").upper(), (m + round(10 * s), y + round(16 * s)),
         0.9 * s, color, max(1, round(2 * s)))
    _put_right(img, f"~{value:.1f}", rx - round(10 * s), y + round(12 * s), 0.6 * s, _INK, 1)
    y += round(54 * s)

    # --- live counts ---
    sm = data.summary
    rows = [("IN LINE", str(sm["in_line"])), ("AT POS", str(sm["at_pos"]))]
    if data.show_alt:
        rows.append(("AT REG", str(data.checkouts["other_n"])))
    for label, val in rows:
        _put(img, label, (m, y), 0.46 * s, _MUTED, 1)
        _put_right(img, val, rx, y, 0.56 * s, _INK, max(1, round(2 * s)))
        y += round(26 * s)

    y += round(6 * s)
    cv2.line(img, (m, y - round(14 * s)), (rx, y - round(14 * s)), _HAIR, 1)

    # --- averages ---
    avgs = [
        ("AVG LINE", f"{sm['avg_line_s']:.1f}s"),
        ("AVG POS", f"{sm['avg_pos_s']:.1f}s"),
        ("AVG TOTAL", f"{sm['avg_total_s']:.1f}s"),
        ("SERVED", str(sm["served_count"])),
    ]
    for label, val in avgs:
        _put(img, label, (m, y), 0.46 * s, _MUTED, 1)
        _put_right(img, val, rx, y, 0.52 * s, _INK, 1)
        y += round(24 * s)

    # --- checkout comparison (Mashgin vs staffed) ---
    ck = data.checkouts
    if data.show_alt and (ck["mashgin_n"] or ck["other_n"]):
        y += round(8 * s)
        cv2.line(img, (m, y - round(14 * s)), (rx, y - round(14 * s)), _HAIR, 1)
        _put(img, "MASHGIN", (m, y), 0.44 * s, _BRAND, 1)
        _put_right(img, f"{ck['mashgin_avg']:.1f}s", rx, y, 0.5 * s, _INK, 1)
        y += round(22 * s)
        _put(img, "STAFFED", (m, y), 0.44 * s, _MUTED, 1)
        _put_right(img, f"{ck['other_avg']:.1f}s", rx, y, 0.5 * s, _INK, 1)
        y += round(22 * s)
        dcol = _LOW if ck["delta"] >= 0 else _HIGH
        _put(img, "SAVES", (m, y), 0.44 * s, _MUTED, 1)
        _put_right(img, f"{ck['delta']:+.1f}s", rx, y, 0.5 * s, dcol, 1)
        y += round(22 * s)

    # --- occupancy sparkline pinned near the bottom ---
    spark_h = round(46 * s)
    spark_y = height - spark_h - round(24 * s)
    if spark_y > y:
        _put(img, "WAITING (RECENT)", (m, spark_y - round(8 * s)), 0.4 * s, _MUTED, 1)
        _sparkline(img, data.occ, m, spark_y, rx - m, spark_h)

    return img


def panel_width(video_w: int) -> int:
    """Panel width = one third of the video (video -> 3/4 of the result).

    Forced so ``video_w + panel_width`` is even, which H.264 (avc1) requires.
    """
    pw = round(video_w / 3)
    if (video_w + pw) % 2:
        pw += 1
    return pw


def composite(video: np.ndarray, panel_data: PanelData) -> np.ndarray:
    """Place ``video`` on the left 3/4 and a rendered panel on the right 1/4.

    The video keeps its native size; the panel is one third of the video width
    (so the video is exactly 3/4 of the result).
    """
    vh, vw = video.shape[:2]
    panel = render_panel(panel_width(vw), vh, panel_data)
    return np.hstack((video, panel))
