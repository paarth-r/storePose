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
    # attrs is a mutable dict; frozen=True does not make this hashable
    __hash__ = None  # type: ignore[assignment]


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


def _active_shape(track: GtTrack, frame: int) -> GtShape | None:
    """The last keyframe at or before ``frame`` (the one that governs state)."""
    active: GtShape | None = None
    for s in track.shapes:
        if s.frame <= frame:
            active = s
        else:
            break
    return active


def present_at(track: GtTrack, frame: int) -> bool:
    """True if the track is visible at ``frame`` (governing keyframe not outside)."""
    active = _active_shape(track, frame)
    return active is not None and not active.outside


def membership_at(track: GtTrack, frame: int) -> str | None:
    """The ``membership`` attribute while present, else ``None``."""
    active = _active_shape(track, frame)
    if active is None or active.outside:
        return None
    return active.attrs.get("membership")


def occupancy_gt_at(tracks: list[GtTrack], frame: int) -> int:
    """Number of present, in-line people at ``frame``."""
    return sum(1 for t in tracks if membership_at(t, frame) == "in_line")


def sample_occupancy_gt(
    tracks: list[GtTrack],
    fps: float,
    step: float = 1.0,
    t_start: float = 0.0,
    t_end: float | None = None,
) -> list[tuple[float, int]]:
    """Sample GT occupancy on ``[t_start, t_end)`` every ``step`` seconds.

    Frame numbers map to seconds via ``fps`` so the timeline aligns with the
    predicted timeline reconstructed from ``waits.csv`` (which is in seconds).
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if not tracks:
        return []
    max_frame = max((s.frame for t in tracks for s in t.shapes), default=0)
    if t_end is None:
        t_end = max_frame / fps
    out: list[tuple[float, int]] = []
    i = 0
    while True:
        t = t_start + i * step
        if t >= t_end:
            break
        frame = round(t * fps)
        out.append((t, occupancy_gt_at(tracks, frame)))
        i += 1
    return out


def write_occupancy_csv(path: str | Path, samples: list[tuple[float, int]]) -> None:
    """Write ``t_s,occupancy`` rows."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "occupancy"])
        for t, occ in samples:
            w.writerow([f"{t:.3f}", occ])


def read_occupancy_csv(path: str | Path) -> list[tuple[float, int]]:
    """Read a ``t_s,occupancy`` CSV back into ``(t, occ)`` pairs."""
    out: list[tuple[float, int]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append((float(row["t_s"]), int(row["occupancy"])))
    return out
