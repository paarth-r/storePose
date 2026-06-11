import os

import numpy as np

from storepose.config import from_args, parse_source
from storepose.video_sink import VideoSink


def test_parse_source_numeric_is_camera_index():
    assert parse_source("0") == 0
    assert parse_source("2") == 2
    assert parse_source(1) == 1


def test_parse_source_path_stays_string():
    assert parse_source("videos/clip.mp4") == "videos/clip.mp4"


def test_from_args_accepts_path_and_save(tmp_path):
    out = str(tmp_path / "out.mp4")
    cfg = from_args(["--source", "videos/clip.mp4", "--save", out])
    assert cfg.source == "videos/clip.mp4"
    assert cfg.save == out


def test_from_args_numeric_source():
    assert from_args(["--source", "3"]).source == 3
    assert from_args([]).save is None


def test_video_sink_writes_nonempty_mp4(tmp_path):
    out = str(tmp_path / "out.mp4")
    frame = np.zeros((48, 64, 3), np.uint8)
    with VideoSink(out, fps=10) as sink:
        for _ in range(5):
            sink.write(frame)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_video_sink_creates_parent_dir(tmp_path):
    out = str(tmp_path / "nested" / "dir" / "out.mp4")
    with VideoSink(out, fps=10) as sink:
        sink.write(np.zeros((48, 64, 3), np.uint8))
    assert os.path.exists(out)


def test_default_pos_zone_path():
    from storepose.queue.zone_editor import default_pos_zone_path
    assert default_pos_zone_path(0) == "zones/cam0_pos.json"
    assert default_pos_zone_path("videos/clip.mp4") == "zones/clip_pos.json"
