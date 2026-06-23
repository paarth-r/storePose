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


def _moving_track(drift: bool) -> Track:
    # Establish steady rightward motion: detections at x=0,10,20 over 3 frames.
    t = Track(0, np.array([0, 0, 10, 20], float), None, None, 1 / 30,
              min_hits=1, smooth=False, min_cutoff=1.0, beta=0.007, drift=drift)
    t.predict(); t.update(np.array([10, 0, 20, 20], float), None, None, 1 / 30)
    t.predict(); t.update(np.array([20, 0, 30, 20], float), None, None, 1 / 30)
    return t


def test_new_track_has_no_reid_event():
    t = Track(0, _box(), _kpts(), np.ones(17), dt=1 / 30,
              min_hits=1, smooth=False, min_cutoff=1.0, beta=0.0)
    assert t.reid_sim is None
    assert t.reid_time_left == 0.0


def test_reactivate_records_reid_similarity_and_arms_notif():
    t = Track(0, _box(), _kpts(), np.ones(17), dt=1 / 30,
              min_hits=1, smooth=False, min_cutoff=1.0, beta=0.0)
    t.predict()  # lost it for a frame
    t.reactivate(_box(), _kpts(), np.ones(17), 1 / 30, reid_sim=0.78)
    assert t.reid_sim == 0.78
    assert t.reid_time_left > 0.0


def test_reid_notif_fades_after_about_one_second():
    t = Track(0, _box(), _kpts(), np.ones(17), dt=1 / 30,
              min_hits=1, smooth=False, min_cutoff=1.0, beta=0.0)
    t.reactivate(_box(), _kpts(), np.ones(17), 1 / 30, reid_sim=0.9)
    assert t.reid_time_left > 0.0
    for _ in range(40):  # >1s of frames at 30fps
        t.predict(1 / 30)
    assert t.reid_time_left == 0.0


def test_no_drift_freezes_coasting_box_at_last_detection():
    t = _moving_track(drift=False)
    last = t.last_box.copy()
    t.predict()  # coast: no detection this frame
    assert t.coasting is True
    np.testing.assert_allclose(t.box, last)  # box must not drift forward


def test_drift_advances_coasting_box_forward():
    t = _moving_track(drift=True)
    last_cx = (t.last_box[0] + t.last_box[2]) / 2.0
    t.predict()  # coast: Kalman extrapolates along the established velocity
    coast_cx = (t.box[0] + t.box[2]) / 2.0
    assert coast_cx > last_cx


def _track(box, appearance_mem=None):
    return Track(0, np.array(box, float), None, None, 1 / 30,
                 min_hits=1, smooth=False, min_cutoff=1.0, beta=0.007,
                 appearance_mem=appearance_mem)


def test_appearance_mem_stored_on_init():
    t = _track([0, 0, 10, 20], appearance_mem=["m0"])
    assert t.appearance_mem == ["m0"]


def test_update_replaces_mem_when_given():
    t = _track([0, 0, 10, 20], appearance_mem=["m0"])
    t.update(np.array([0, 0, 10, 20], float), None, None, 1 / 30, appearance_mem=["m1"])
    assert t.appearance_mem == ["m1"]


def test_update_none_mem_keeps_previous():
    t = _track([0, 0, 10, 20], appearance_mem=["m0"])
    t.update(np.array([0, 0, 10, 20], float), None, None, 1 / 30, appearance_mem=None)
    assert t.appearance_mem == ["m0"]


def test_reactivate_reseats_motion_and_keeps_identity():
    t = _track([0, 0, 10, 20], appearance_mem=["m0"])
    t.confirmed = True
    for _ in range(5):
        t.predict()
    assert t.time_since_update == 5
    t.reactivate(np.array([100, 100, 110, 120], float), None, None, 1 / 30,
                 appearance_mem=["m1"])
    assert t.time_since_update == 0
    assert t.confirmed is True
    assert np.allclose(t.box, [100, 100, 110, 120], atol=1.0)
    assert t.hits == 2  # reactivation counts as a fresh match
    assert t.appearance_mem == ["m1"]


def test_velocity_reads_recent_heading():
    # A track fed rightward detections has a +x velocity and ~0 y.
    t = Track(0, np.array([0, 0, 10, 20], float), None, None, 1 / 2,
              min_hits=1, smooth=False, min_cutoff=1.0, beta=0.0)
    for x in (10, 20, 30):
        t.predict(0.5)
        t.update(np.array([x, 0, x + 10, 20], float), None, None, 0.5)
    vx, vy = t.velocity()
    assert vx > 0 and abs(vy) < abs(vx)
    cx, cy = t.last_center()
    assert cx == 35.0 and cy == 10.0  # center of [30,0,40,20]


def test_velocity_zero_for_single_detection():
    t = Track(0, np.array([0, 0, 10, 20], float), None, None, 1 / 2,
              min_hits=1, smooth=False, min_cutoff=1.0, beta=0.0)
    assert t.velocity() == (0.0, 0.0)
