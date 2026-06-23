from __future__ import annotations

import xml.etree.ElementTree as ET

from storepose.eval.cvat_export import BoxShape, BoxTrack, tracks_to_cvat_xml
from storepose.eval.cvat_import import parse_cvat_xml


def _track(tid: int, shapes: list[BoxShape], label: str = "person") -> BoxTrack:
    return BoxTrack(id=tid, label=label, shapes=shapes)


def test_export_roundtrips_through_the_importer():
    # Export box tracks, then parse them back: the importer should recover each
    # box's bottom-center ground point, outside flag, and membership attribute.
    tracks = [
        _track(0, [
            BoxShape(10, False, 80.0, 40.0, 120.0, 200.0, {"membership": "in_line"}),
            BoxShape(11, True, 90.0, 50.0, 130.0, 210.0, {"membership": "in_line"}),
        ]),
        _track(1, [
            BoxShape(12, False, 300.0, 100.0, 360.0, 400.0, {"membership": "bystander"}),
        ]),
    ]
    xml = tracks_to_cvat_xml(tracks, width=1920, height=1080)
    parsed = parse_cvat_xml(xml)

    assert [t.id for t in parsed] == [0, 1]
    assert parsed[0].label == "person"
    # bottom-center of (80,40,120,200) is (100, 200)
    assert (parsed[0].shapes[0].x, parsed[0].shapes[0].y) == (100.0, 200.0)
    assert parsed[0].shapes[0].attrs["membership"] == "in_line"
    assert parsed[0].shapes[1].outside is True
    assert parsed[1].shapes[0].attrs["membership"] == "bystander"


def test_export_emits_valid_cvat_for_video_header():
    xml = tracks_to_cvat_xml([], width=640, height=480)
    root = ET.fromstring(xml)
    assert root.tag == "annotations"
    assert root.findtext("version") == "1.1"
    # CVAT matches uploaded tracks to the task by image size in meta.
    assert root.find("./meta/task/original_size/width").text == "640"
    assert root.find("./meta/task/original_size/height").text == "480"
