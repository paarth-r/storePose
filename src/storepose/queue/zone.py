"""Polygon queue zone with point-in-zone test and JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class Zone:
    """A polygon region of interest in image coordinates.

    ``contains`` reports whether a point falls inside (or on) the polygon. A
    zone with fewer than 3 points contains nothing.
    """

    def __init__(self, points: list[tuple[int, int]]):
        self.points: list[tuple[int, int]] = [(int(x), int(y)) for x, y in points]
        self._poly = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)

    def contains(self, point: tuple[float, float]) -> bool:
        """True if ``point`` is inside or on the polygon boundary."""
        if len(self.points) < 3:
            return False
        inside = cv2.pointPolygonTest(self._poly, (float(point[0]), float(point[1])), False)
        return inside >= 0

    def coverage(self, box, grid: int = 7) -> float:
        """Fraction of ``box`` (xyxy) inside the polygon, via a ``grid``x``grid``
        sample of points across the box. 1.0 = fully inside, 0.0 = fully out."""
        if len(self.points) < 3:
            return 0.0
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inside = 0
        for i in range(grid):
            px = x1 + (x2 - x1) * (i + 0.5) / grid
            for j in range(grid):
                py = y1 + (y2 - y1) * (j + 0.5) / grid
                if cv2.pointPolygonTest(self._poly, (px, py), False) >= 0:
                    inside += 1
        return inside / (grid * grid)

    def to_dict(self) -> dict:
        return {"points": [list(p) for p in self.points]}

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        return cls([tuple(p) for p in data["points"]])

    def save(self, path: str) -> None:
        p = Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str) -> "Zone":
        return cls.from_dict(json.loads(Path(path).read_text()))
