"""Serialize tracked-person boxes into a CVAT-for-video 1.1 XML pre-annotation.

This is the mirror of ``cvat_import``: the storePose pipeline detects and tracks
people, and this module writes those tracks as CVAT box tracks so a human can
review them (drag, delete false positives, fix ``membership``) instead of
annotating from scratch. Upload the result into a CVAT task via
*Menu -> Upload annotations* with format *"CVAT for video 1.1"*.

The task's ``person`` label must already exist with shape *rectangle* and a
``membership`` select attribute (values ``in_line``/``bystander``); CVAT matches
uploaded tracks to the task by label name and image size.

The serialization here is pure and round-trip tested against ``parse_cvat_xml``;
the pipeline driver that produces ``BoxTrack``s lives in ``busy_report.py``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class BoxShape:
    """One keyframe of a box track: ``xyxy`` extent, visibility, attributes."""

    frame: int
    outside: bool
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class BoxTrack:
    """A single person's box track across frames."""

    id: int
    label: str
    shapes: list[BoxShape]


def tracks_to_cvat_xml(
    tracks: list[BoxTrack],
    *,
    width: int,
    height: int,
) -> str:
    """Render ``tracks`` as a CVAT-for-video 1.1 annotation document.

    ``width``/``height`` are the source frame size, written into ``meta`` so CVAT
    aligns the upload to the task's images. Every shape is emitted as a keyframe
    (the pipeline produces a box per frame, so no interpolation is needed); mark
    a track's departure frame with ``outside=True``.
    """
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    size = ET.SubElement(task, "original_size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)

    for tr in tracks:
        t_el = ET.SubElement(root, "track", id=str(tr.id), label=tr.label)
        for s in sorted(tr.shapes, key=lambda s: s.frame):
            b_el = ET.SubElement(
                t_el,
                "box",
                frame=str(s.frame),
                outside="1" if s.outside else "0",
                occluded="0",
                keyframe="1",
                xtl=f"{s.xtl:.2f}",
                ytl=f"{s.ytl:.2f}",
                xbr=f"{s.xbr:.2f}",
                ybr=f"{s.ybr:.2f}",
            )
            for name, value in s.attrs.items():
                a_el = ET.SubElement(b_el, "attribute", name=name)
                a_el.text = value

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
        root, encoding="unicode"
    )
