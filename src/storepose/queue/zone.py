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
