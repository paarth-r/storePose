from __future__ import annotations

import pytest

from storepose.eval.cvat_import import (
    GtShape,
    GtTrack,
    membership_at,
    occupancy_gt_at,
    parse_cvat_xml,
    present_at,
    read_occupancy_csv,
    sample_occupancy_gt,
    write_occupancy_csv,
)

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


def test_present_only_within_track_before_outside():
    tracks = parse_cvat_xml(SAMPLE_XML)
    track = tracks[0]
    assert present_at(track, 9) is False     # before first keyframe
    assert present_at(track, 10) is True      # at first keyframe
    assert present_at(track, 25) is True      # between non-outside keyframes
    assert present_at(track, 30) is False     # outside keyframe
    assert present_at(track, 99) is False     # after outside persists


def test_membership_is_none_when_absent():
    tracks = parse_cvat_xml(SAMPLE_XML)
    track = tracks[0]
    assert membership_at(track, 15) == "in_line"
    assert membership_at(track, 30) is None   # outside -> not a member


def test_occupancy_counts_only_in_line_and_present():
    tracks = parse_cvat_xml(SAMPLE_XML)
    # frame 15: track 0 in_line present, track 1 bystander present -> occ 1
    assert occupancy_gt_at(tracks, 15) == 1
    # frame 9: neither present -> 0
    assert occupancy_gt_at(tracks, 9) == 0
    # frame 30: track 0 outside, track 1 absent -> 0
    assert occupancy_gt_at(tracks, 30) == 0


def test_sample_occupancy_gt_grid_and_values():
    tracks = parse_cvat_xml(SAMPLE_XML)  # in_line track present frames [10,30)
    # fps=10 -> 1s == 10 frames; sample every 1s over [0, t_end)
    samples = sample_occupancy_gt(tracks, fps=10.0, step=1.0)
    by_t = dict(samples)
    assert by_t[0.0] == 0    # frame 0, before any keyframe
    assert by_t[1.0] == 1    # frame 10 = first in_line keyframe; round(1.0*10)=10


def test_sample_occupancy_gt_validates_inputs():
    tracks = parse_cvat_xml(SAMPLE_XML)
    with pytest.raises(ValueError):
        sample_occupancy_gt(tracks, fps=0.0)
    with pytest.raises(ValueError):
        sample_occupancy_gt(tracks, fps=10.0, step=0.0)


def test_occupancy_csv_roundtrip(tmp_path):
    samples = [(0.0, 0), (1.0, 2), (2.0, 1)]
    p = tmp_path / "gt.csv"
    write_occupancy_csv(p, samples)
    assert read_occupancy_csv(p) == samples
