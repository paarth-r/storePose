"""Interactive click-to-draw editor for a queue zone polygon."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .zone import Zone

_COLOR = (0, 180, 255)
_POS_COLOR = (255, 200, 0)  # azure, matches drawing.POS_COLOR


def default_zone_path(source: int | str) -> str:
    """Default ``zones/<name>.json`` path for a source."""
    name = f"cam{source}" if isinstance(source, int) else Path(str(source)).stem
    return str(Path("zones") / f"{name}.json")


def default_pos_zone_path(source: int | str) -> str:
    """Default ``zones/<name>_pos.json`` path for a POS zone."""
    name = f"cam{source}" if isinstance(source, int) else Path(str(source)).stem
    return str(Path("zones") / f"{name}_pos.json")


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


def define_zones(
    source: int | str,
    line_path: str | None = None,
    pos_path: str | None = None,
    pos_only: bool = False,
) -> dict[str, str]:
    """Draw line (key ``1``) and POS (key ``2``) contours in one session.

    Keys: left-click adds a point; ``n`` finishes the current contour (>=3 pts)
    and starts a new one; ``u`` undoes a point; ``c`` clears all; ``s`` saves;
    ``q`` quits. Saves all line contours to ``line_path`` and all POS contours to
    ``pos_path`` (each only if it has >=1 valid contour). Returns the saved paths.
    """
    line_path = line_path or default_zone_path(source)
    pos_path = pos_path or default_pos_zone_path(source)
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from source {source!r}")

    line_contours: list[list[tuple[int, int]]] = []
    pos_contours: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    mode = "pos" if pos_only else "line"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

    def color_for(m):
        return _POS_COLOR if m == "pos" else _COLOR

    def commit_current(target_mode):
        if len(current) >= 3:
            (pos_contours if target_mode == "pos" else line_contours).append(list(current))
        current.clear()

    win = "define zones"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    saved: dict[str, str] = {}
    while True:
        disp = frame.copy()
        for poly in line_contours:
            cv2.polylines(disp, [np.array(poly, np.int32).reshape(-1, 1, 2)], True, _COLOR, 2)
        for poly in pos_contours:
            cv2.polylines(disp, [np.array(poly, np.int32).reshape(-1, 1, 2)], True, _POS_COLOR, 2)
        c = color_for(mode)
        for p in current:
            cv2.circle(disp, p, 4, c, -1)
        if len(current) >= 2:
            cv2.polylines(disp, [np.array(current, np.int32).reshape(-1, 1, 2)],
                          len(current) >= 3, c, 2)
        hint = (f"mode:{mode.upper()}  1:line 2:pos  n:new-contour  "
                f"u:undo c:clear s:save q:quit   line:{len(line_contours)} pos:{len(pos_contours)}")
        cv2.putText(disp, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("1") and not pos_only:
            commit_current(mode); mode = "line"
        elif key == ord("2"):
            commit_current(mode); mode = "pos"
        elif key == ord("n"):
            commit_current(mode)
        elif key == ord("u") and current:
            current.pop()
        elif key == ord("c"):
            current.clear(); line_contours.clear(); pos_contours.clear()
        elif key == ord("s"):
            commit_current(mode)
            if pos_contours:
                Zone.from_polygons(pos_contours).save(pos_path)
                saved["pos"] = pos_path
                print(f"Saved POS zone ({len(pos_contours)} contour(s)) to {pos_path}")
            if line_contours and not pos_only:
                Zone.from_polygons(line_contours).save(line_path)
                saved["line"] = line_path
                print(f"Saved line zone ({len(line_contours)} contour(s)) to {line_path}")
            if saved:
                break
            print("Draw at least one contour (>=3 points) before saving.")
    cv2.destroyAllWindows()
    return saved
