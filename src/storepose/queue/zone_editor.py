"""Interactive click-to-draw editor for a queue zone polygon."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .zone import Zone

_COLOR = (0, 180, 255)        # line zone — orange
_POS_COLOR = (70, 200, 60)    # Mashgin checkout — green (matches drawing.POS_COLOR)
_ALT_COLOR = (60, 60, 235)    # non-Mashgin checkout — red (matches drawing.ALT_COLOR)
_BLUR_COLOR = (255, 0, 255)   # censor/blur zone — magenta


def default_zone_path(source: int | str) -> str:
    """Default ``zones/<name>.json`` path for a source."""
    name = f"cam{source}" if isinstance(source, int) else Path(str(source)).stem
    return str(Path("zones") / f"{name}.json")


def default_pos_zone_path(source: int | str) -> str:
    """Default ``zones/<name>_pos.json`` path for a POS zone."""
    name = f"cam{source}" if isinstance(source, int) else Path(str(source)).stem
    return str(Path("zones") / f"{name}_pos.json")


def default_alt_zone_path(source: int | str) -> str:
    """Default ``zones/<name>_alt.json`` path for a non-Mashgin checkout zone."""
    name = f"cam{source}" if isinstance(source, int) else Path(str(source)).stem
    return str(Path("zones") / f"{name}_alt.json")


def default_blur_zone_path(source: int | str) -> str:
    """Default ``zones/<name>_blur.json`` path for a censor/blur zone."""
    name = f"cam{source}" if isinstance(source, int) else Path(str(source)).stem
    return str(Path("zones") / f"{name}_blur.json")


def default_ignore_zone_path(source: int | str) -> str:
    """Default ``zones/<name>_ignore.json`` path for a detection ignore zone."""
    name = f"cam{source}" if isinstance(source, int) else Path(str(source)).stem
    return str(Path("zones") / f"{name}_ignore.json")


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
    alt_path: str | None = None,
    blur_path: str | None = None,
    only: str | None = None,
) -> dict[str, str]:
    """Draw line (``1``), Mashgin POS (``2``), non-Mashgin (``3``) and censor/blur
    (``4``) contours.

    ``n`` finishes the current contour (>=3 pts) and starts a new one; ``u`` undo;
    ``c`` clear all; ``s`` save; ``q`` quit. ``only`` ("pos", "alt" or "blur")
    locks the editor to that single type. Saves each type that has >=1 contour to
    its path (line/pos/alt/blur) and returns the saved paths.
    """
    paths = {
        "line": line_path or default_zone_path(source),
        "pos": pos_path or default_pos_zone_path(source),
        "alt": alt_path or default_alt_zone_path(source),
        "blur": blur_path or default_blur_zone_path(source),
    }
    colors = {"line": _COLOR, "pos": _POS_COLOR, "alt": _ALT_COLOR, "blur": _BLUR_COLOR}
    allowed = {"line", "pos", "alt", "blur"} if only is None else {only}

    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from source {source!r}")

    contours: dict[str, list] = {"line": [], "pos": [], "alt": [], "blur": []}
    current: list[tuple[int, int]] = []
    mode = only or "line"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

    def commit():
        if len(current) >= 3:
            contours[mode].append(list(current))
        current.clear()

    win = "define zones"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    saved: dict[str, str] = {}
    while True:
        disp = frame.copy()
        for typ in ("line", "pos", "alt", "blur"):
            for poly in contours[typ]:
                cv2.polylines(disp, [np.array(poly, np.int32).reshape(-1, 1, 2)],
                              True, colors[typ], 2)
        c = colors[mode]
        for p in current:
            cv2.circle(disp, p, 4, c, -1)
        if len(current) >= 2:
            cv2.polylines(disp, [np.array(current, np.int32).reshape(-1, 1, 2)],
                          len(current) >= 3, c, 2)
        hint = (f"mode:{mode.upper()}  1:line 2:mashgin 3:non-mashgin 4:blur  n:new  "
                f"u:undo c:clear s:save q:quit   "
                f"L:{len(contours['line'])} M:{len(contours['pos'])} "
                f"N:{len(contours['alt'])} B:{len(contours['blur'])}")
        cv2.putText(disp, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("1") and "line" in allowed:
            commit(); mode = "line"
        elif key == ord("2") and "pos" in allowed:
            commit(); mode = "pos"
        elif key == ord("3") and "alt" in allowed:
            commit(); mode = "alt"
        elif key == ord("4") and "blur" in allowed:
            commit(); mode = "blur"
        elif key == ord("n"):
            commit()
        elif key == ord("u") and current:
            current.pop()
        elif key == ord("c"):
            current.clear()
            for typ in contours:
                contours[typ].clear()
        elif key == ord("s"):
            commit()
            for typ in ("line", "pos", "alt", "blur"):
                if typ in allowed and contours[typ]:
                    Zone.from_polygons(contours[typ]).save(paths[typ])
                    saved[typ] = paths[typ]
                    print(f"Saved {typ} zone ({len(contours[typ])} contour(s)) to {paths[typ]}")
            if saved:
                break
            print("Draw at least one contour (>=3 points) before saving.")
    cv2.destroyAllWindows()
    return saved
