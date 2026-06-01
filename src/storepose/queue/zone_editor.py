"""Interactive click-to-draw editor for a queue zone polygon."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .zone import Zone

_COLOR = (0, 180, 255)


def default_zone_path(source: int | str) -> str:
    """Default ``zones/<name>.json`` path for a source."""
    name = f"cam{source}" if isinstance(source, int) else Path(str(source)).stem
    return str(Path("zones") / f"{name}.json")


def define_zone(source: int | str, out_path: str | None = None) -> str:
    """Open a frame from ``source`` and let the user click a polygon.

    Keys: left-click adds a point, ``u`` undo, ``c`` clear, ``s`` save, ``q``
    quit. Returns the path the zone was saved to (or the intended path).
    """
    out_path = out_path or default_zone_path(source)
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from source {source!r}")

    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    win = "define zone"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = frame.copy()
        if points:
            arr = np.array(points, np.int32).reshape(-1, 1, 2)
            for p in points:
                cv2.circle(disp, p, 4, _COLOR, -1)
            if len(points) >= 2:
                cv2.polylines(disp, [arr], len(points) >= 3, _COLOR, 2)
        cv2.putText(disp, f"{len(points)} pts | u:undo  c:clear  s:save  q:quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLOR, 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u") and points:
            points.pop()
        elif key == ord("c"):
            points.clear()
        elif key == ord("s"):
            if len(points) >= 3:
                Zone(points).save(out_path)
                print(f"Saved zone ({len(points)} pts) to {out_path}")
                break
            print("Need at least 3 points to save.")
    cv2.destroyAllWindows()
    return out_path
