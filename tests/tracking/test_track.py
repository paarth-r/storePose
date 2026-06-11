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


def _track(box, descriptor=None):
    return Track(0, np.array(box, float), None, None, 1 / 30,
                 min_hits=1, smooth=False, min_cutoff=1.0, beta=0.007,
                 descriptor=descriptor)


def test_descriptor_seeded_then_blended():
    t = _track([0, 0, 10, 20], descriptor=np.array([1.0, 0.0], np.float32))
    assert np.allclose(t.descriptor, [1.0, 0.0])
    t.update(np.array([0, 0, 10, 20], float), None, None, 1 / 30,
             descriptor=np.array([0.0, 1.0], np.float32))
    # EMA alpha=0.3 -> 0.7*[1,0] + 0.3*[0,1]
    assert np.allclose(t.descriptor, [0.7, 0.3])


def test_reactivate_reseats_motion_and_keeps_identity():
    t = _track([0, 0, 10, 20], descriptor=np.array([1.0], np.float32))
    t.confirmed = True
    for _ in range(5):
        t.predict()
    assert t.time_since_update == 5
    t.reactivate(np.array([100, 100, 110, 120], float), None, None, 1 / 30,
                 descriptor=np.array([1.0], np.float32))
    assert t.time_since_update == 0
    assert t.confirmed is True
    assert np.allclose(t.box, [100, 100, 110, 120], atol=1.0)
    assert t.hits == 2  # reactivation counts as a fresh match


def test_update_descriptor_none_keeps_previous():
    t = _track([0, 0, 10, 20], descriptor=np.array([1.0, 0.0], np.float32))
    t.update(np.array([0, 0, 10, 20], float), None, None, 1 / 30, descriptor=None)
    assert np.allclose(t.descriptor, [1.0, 0.0])
