"""Convert CVAT-for-video point-track exports into occupancy ground truth.

CVAT annotates each person as a single *point* in *track* mode (keyframe +
interpolate). A track's presence is defined by its keyframes and ``outside``
flags, not by positional interpolation, so per-frame occupancy counts are
well-defined. The pure logic here is unit-tested; the CLI shell lives in
``busy_report.py``.
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GtShape:
    """One keyframe of a track: position, visibility, and attributes."""

    frame: int
    outside: bool
    x: float
    y: float
    attrs: dict[str, str]


@dataclass
class GtTrack:
    """A single person's point track. ``shapes`` are sorted by ``frame``."""

    id: int
    label: str
    shapes: list[GtShape]


def parse_cvat_xml(text: str) -> list[GtTrack]:
    """Parse a CVAT-for-video 1.1 XML export into a list of tracks."""
    root = ET.fromstring(text)
    tracks: list[GtTrack] = []
    for tr in root.findall("track"):
        shapes: list[GtShape] = []
        for pt in tr.findall("points"):
            coords = (pt.get("points") or "").split(";")[0]
            x_str, _, y_str = coords.partition(",")
            attrs = {
                a.get("name", ""): (a.text or "") for a in pt.findall("attribute")
            }
            shapes.append(
                GtShape(
                    frame=int(pt.get("frame", "0")),
                    outside=pt.get("outside") == "1",
                    x=float(x_str) if x_str else 0.0,
                    y=float(y_str) if y_str else 0.0,
                    attrs=attrs,
                )
            )
        shapes.sort(key=lambda s: s.frame)
        tracks.append(
            GtTrack(id=int(tr.get("id", "0")), label=tr.get("label", ""), shapes=shapes)
        )
    return tracks
