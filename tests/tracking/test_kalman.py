import numpy as np
import pytest

from storepose.tracking.kalman import KalmanBoxTracker


def _cx(box):
    return (box[0] + box[2]) / 2


def test_init_box_roundtrips():
    box = np.array([10, 20, 30, 60], float)
    t = KalmanBoxTracker(box)
    np.testing.assert_allclose(t.box, box, atol=1e-6)


def test_predict_static_keeps_center():
    t = KalmanBoxTracker(np.array([0, 0, 10, 10], float))
    t.predict()
    assert _cx(t.box) == pytest.approx(5.0, abs=1e-6)


def test_update_moves_toward_measurement():
    t = KalmanBoxTracker(np.array([0, 0, 10, 10], float))
    for _ in range(5):
        t.update(np.array([10, 0, 20, 10], float))  # shifted +10 in x
    assert _cx(t.box) > 5.0


def test_predict_advances_by_velocity():
    t = KalmanBoxTracker(np.array([0, 0, 10, 10], float))
    for i in range(1, 6):
        t.update(np.array([10 * i, 0, 10 * i + 10, 10], float))  # +10/frame
    before = _cx(t.box)
    t.predict()
    assert _cx(t.box) > before
