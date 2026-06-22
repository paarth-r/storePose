from __future__ import annotations

from storepose.eval.cvat_import import GtShape, GtTrack, parse_cvat_xml

SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <track id="0" label="person">
    <points frame="10" outside="0" occluded="0" keyframe="1" points="100.0,200.0">
      <attribute name="membership">in_line</attribute>
    </points>
    <points frame="20" outside="0" occluded="0" keyframe="1" points="110.0,205.0">
      <attribute name="membership">in_line</attribute>
    </points>
    <points frame="30" outside="1" occluded="0" keyframe="1" points="120.0,210.0">
      <attribute name="membership">in_line</attribute>
    </points>
  </track>
  <track id="1" label="person">
    <points frame="12" outside="0" occluded="0" keyframe="1" points="300.0,400.0">
      <attribute name="membership">bystander</attribute>
    </points>
  </track>
</annotations>
"""


def test_parse_builds_tracks_and_shapes():
    tracks = parse_cvat_xml(SAMPLE_XML)
    assert [t.id for t in tracks] == [0, 1]
    assert tracks[0].label == "person"
    assert tracks[0].shapes[0] == GtShape(10, False, 100.0, 200.0, {"membership": "in_line"})
    assert tracks[0].shapes[2].outside is True
    assert tracks[1].shapes[0].attrs["membership"] == "bystander"


def test_parse_shapes_sorted_by_frame():
    xml = SAMPLE_XML.replace('frame="10"', 'frame="99"')  # out-of-order keyframe
    tracks = parse_cvat_xml(xml)
    frames = [s.frame for s in tracks[0].shapes]
    assert frames == sorted(frames)


def test_parse_multipoint_takes_first_point():
    xml = SAMPLE_XML.replace('points="100.0,200.0"', 'points="100.0,200.0;150.0,250.0"')
    tracks = parse_cvat_xml(xml)
    assert (tracks[0].shapes[0].x, tracks[0].shapes[0].y) == (100.0, 200.0)
