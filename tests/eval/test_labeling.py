from storepose.busy.types import BusyLevel
from storepose.eval.labeling import (
    Window,
    enumerate_windows,
    level_for_key,
    unlabeled,
)


def test_enumerate_full_windows():
    wins = enumerate_windows(duration_s=1800.0, window_s=600.0)
    assert [w.index for w in wins] == [0, 1, 2]
    assert wins[0] == Window(0, 0.0, 600.0)
    assert wins[2].end_s == 1800.0


def test_enumerate_includes_partial_final_window():
    wins = enumerate_windows(duration_s=700.0, window_s=600.0)
    assert len(wins) == 2
    assert wins[1].start_s == 600.0
    assert wins[1].end_s == 700.0  # clipped to duration, not 1200


def test_short_clip_yields_one_window():
    wins = enumerate_windows(duration_s=42.0, window_s=600.0)
    assert len(wins) == 1
    assert wins[0] == Window(0, 0.0, 42.0)


def test_enumerate_empty_for_zero_duration():
    assert enumerate_windows(0.0, 600.0) == []


def test_unlabeled_skips_existing():
    wins = enumerate_windows(1800.0, 600.0)
    existing = {0: BusyLevel.LOW, 2: BusyLevel.HIGH}
    assert [w.index for w in unlabeled(wins, existing)] == [1]


def test_level_for_key():
    assert level_for_key("1") == BusyLevel.LOW
    assert level_for_key("m") == BusyLevel.MEDIUM
    assert level_for_key("H") == BusyLevel.HIGH
    assert level_for_key("x") is None
