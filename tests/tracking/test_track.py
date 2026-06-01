import numpy as np

from storepose.tracking.track import Track, color_for


def _box():
    return np.array([0, 0, 10, 20], float)


def _kpts():
    return np.full((17, 2), 5.0)


def test_color_is_stable_and_bgr_tuple():
    c = color_for(2)
    assert color_for(2) == c
    assert len(c) == 3


def test_new_track_with_min_hits_one_is_confirmed():
    t = Track(0, _box(), _kpts(), np.ones(17), dt=1 / 30,
              min_hits=1, smooth=False, min_cutoff=1.0, beta=0.0)
    assert t.confirmed is True
    assert t.coasting is False
    assert t.keypoints is not None


def test_track_confirms_after_min_hits():
    t = Track(0, _box(), _kpts(), np.ones(17), dt=1 / 30,
              min_hits=3, smooth=False, min_cutoff=1.0, beta=0.0)
    assert t.confirmed is False
    t.predict(); t.update(_box(), _kpts(), np.ones(17), 1 / 30)
    assert t.confirmed is False
    t.predict(); t.update(_box(), _kpts(), np.ones(17), 1 / 30)
    assert t.confirmed is True


def test_predict_marks_coasting():
    t = Track(0, _box(), _kpts(), np.ones(17), dt=1 / 30,
              min_hits=1, smooth=False, min_cutoff=1.0, beta=0.0)
    t.predict()
    assert t.coasting is True
    assert t.time_since_update == 1
    t.update(_box(), _kpts(), np.ones(17), 1 / 30)
    assert t.coasting is False
    assert t.time_since_update == 0
