"""Polygon zone (one or more contours) with point/coverage tests + JSON."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _is_nested(seq) -> bool:
    """True if ``seq`` is a list of contours (vs a flat list of points)."""
    return bool(seq) and isinstance(seq[0][0], (list, tuple, np.ndarray))


class Zone:
    """One or more polygon contours in image coordinates.

    ``contains`` and ``coverage`` test membership against the **union** of the
    contours. A contour with fewer than 3 points contributes nothing. Construct
    with a flat list of points for a single contour (back-compatible) or with
    :meth:`from_polygons` for several.
    """

    def __init__(self, points_or_polygons):
        polys = points_or_polygons or []
        if polys and not _is_nested(polys):
            polys = [polys]  # a flat list of points is one contour
        self.polygons: list[list[tuple[int, int]]] = [
            [(int(x), int(y)) for x, y in poly] for poly in polys
        ]
        self._polys = [
            np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            for poly in self.polygons
            if len(poly) >= 3
        ]

    @classmethod
    def from_polygons(cls, polygons) -> "Zone":
        return cls(list(polygons))

    @property
    def points(self) -> list[tuple[int, int]]:
        """The first contour (back-compat; ``[]`` when empty)."""
        return self.polygons[0] if self.polygons else []

    def contains(self, point: tuple[float, float]) -> bool:
        """True if ``point`` is inside or on any contour."""
        pt = (float(point[0]), float(point[1]))
        return any(cv2.pointPolygonTest(poly, pt, False) >= 0 for poly in self._polys)

    def coverage(self, box, grid: int = 7) -> float:
        """Fraction of ``box`` (xyxy) inside any contour, via a grid sample."""
        if not self._polys:
            return 0.0
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inside = 0
        for i in range(grid):
            px = x1 + (x2 - x1) * (i + 0.5) / grid
            for j in range(grid):
                py = y1 + (y2 - y1) * (j + 0.5) / grid
                if any(cv2.pointPolygonTest(poly, (px, py), False) >= 0
                       for poly in self._polys):
                    inside += 1
        return inside / (grid * grid)

    def to_dict(self) -> dict:
        return {"polygons": [[list(p) for p in poly] for poly in self.polygons]}

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        if "polygons" in data:
            return cls.from_polygons(
                [[tuple(p) for p in poly] for poly in data["polygons"]]
            )
        return cls([tuple(p) for p in data["points"]])  # legacy single-contour

    def save(self, path: str) -> None:
        p = Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str) -> "Zone":
        return cls.from_dict(json.loads(Path(path).read_text()))
