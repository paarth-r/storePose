import numpy as np

from storepose.pipeline import FrameResult
from storepose.tracking.tracker import MultiObjectTracker


def make_result(boxes, kpts=None):
    boxes = np.array(boxes, float).reshape(-1, 4)
    n = len(boxes)
    if kpts is None:
        kpts = np.zeros((n, 17, 2), float)
    return FrameResult(boxes=boxes, keypoints=kpts, scores=np.ones((n, 17), float))


def test_track_confirms_after_min_hits():
    tr = MultiObjectTracker(max_age=10, min_hits=3, iou_thr=0.3, smooth=False)
    box = [10, 10, 50, 90]
    assert tr.update(make_result([box]), 1 / 30) == []
    assert tr.update(make_result([box]), 1 / 30) == []
    out = tr.update(make_result([box]), 1 / 30)
    assert len(out) == 1 and out[0].id == 0 and out[0].coasting is False


def test_stable_id_across_frames():
    tr = MultiObjectTracker(max_age=10, min_hits=1, iou_thr=0.3, smooth=False)
    box = [10, 10, 50, 90]
    a = tr.update(make_result([box]), 1 / 30)
    b = tr.update(make_result([box]), 1 / 30)
    assert a[0].id == b[0].id == 0


def test_two_detections_get_distinct_ids():
    tr = MultiObjectTracker(max_age=10, min_hits=1, iou_thr=0.3, smooth=False)
    out = tr.update(make_result([[0, 0, 20, 40], [100, 0, 120, 40]]), 1 / 30)
    assert {p.id for p in out} == {0, 1}


def test_coasts_then_dies():
    tr = MultiObjectTracker(max_age=3, min_hits=1, iou_thr=0.3, smooth=False)
    box = [10, 10, 50, 90]
    assert tr.update(make_result([box]), 1 / 30)[0].coasting is False
    o1 = tr.update(make_result([]), 1 / 30)
    assert len(o1) == 1 and o1[0].coasting is True and o1[0].keypoints is None
    tr.update(make_result([]), 1 / 30)
    assert len(tr.update(make_result([]), 1 / 30)) == 1  # t=3, still within max_age
    assert tr.update(make_result([]), 1 / 30) == []      # t=4, culled


def test_reentry_gets_new_id():
    tr = MultiObjectTracker(max_age=1, min_hits=1, iou_thr=0.3, smooth=False)
    box = [10, 10, 50, 90]
    tr.update(make_result([box]), 1 / 30)   # id 0
    tr.update(make_result([]), 1 / 30)      # coast t=1
    tr.update(make_result([]), 1 / 30)      # t=2 > max_age -> culled
    out = tr.update(make_result([box]), 1 / 30)
    assert len(out) == 1 and out[0].id == 1


def test_tentative_track_dropped_on_miss():
    tr = MultiObjectTracker(max_age=10, min_hits=3, iou_thr=0.3, smooth=False)
    tr.update(make_result([[10, 10, 50, 90]]), 1 / 30)  # tentative (hits=1)
    tr.update(make_result([]), 1 / 30)                  # missed before confirm
    # a fresh detection should now be id 1 (the tentative track is gone)
    out = tr.update(make_result([[10, 10, 50, 90]]), 1 / 30)
    tr.update(make_result([[10, 10, 50, 90]]), 1 / 30)
    out = tr.update(make_result([[10, 10, 50, 90]]), 1 / 30)
    assert out and out[0].id == 1


from storepose.tracking.tracker import suppress_coasting_duplicates


class _FakeTrack:
    def __init__(self, box, coasting, confirmed=True, hits=5):
        self.box = np.array(box, float)
        self.coasting = coasting
        self.confirmed = confirmed
        self.hits = hits


def test_suppresses_coasting_ghost_over_active():
    active = _FakeTrack([0, 0, 40, 80], coasting=False)
    ghost = _FakeTrack([3, 3, 43, 83], coasting=True)  # IoU ~0.86
    kept = suppress_coasting_duplicates([active, ghost], max_overlap=0.5)
    assert active in kept and ghost not in kept


def test_keeps_two_overlapping_active_tracks():
    a = _FakeTrack([0, 0, 40, 80], coasting=False)
    b = _FakeTrack([3, 3, 43, 83], coasting=False)  # both real detections
    kept = suppress_coasting_duplicates([a, b], max_overlap=0.5)
    assert len(kept) == 2  # never merge two genuinely-detected people


def test_keeps_coasting_track_that_overlaps_nobody():
    a = _FakeTrack([0, 0, 40, 80], coasting=False)
    c = _FakeTrack([100, 0, 140, 80], coasting=True)  # separate, still predicted
    kept = suppress_coasting_duplicates([a, c], max_overlap=0.5)
    assert len(kept) == 2
