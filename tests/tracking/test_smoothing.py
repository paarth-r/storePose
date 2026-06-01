import numpy as np

from storepose.tracking.smoothing import KeypointSmoother, OneEuroFilter


def test_first_sample_passes_through():
    f = OneEuroFilter()
    assert f(5.0, dt=1 / 30) == 5.0


def test_constant_signal_stays_constant():
    f = OneEuroFilter()
    f(3.0, 1 / 30)
    for _ in range(20):
        out = f(3.0, 1 / 30)
    assert abs(out - 3.0) < 1e-6


def test_reduces_variance_on_noisy_constant():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    noisy = 10.0 + rng.normal(0, 1.0, size=200)
    out = np.array([f(float(v), 1 / 30) for v in noisy])
    assert out[50:].std() < noisy[50:].std()


def test_tracks_a_ramp():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.1)
    last = 0.0
    for i in range(100):
        last = f(float(i), 1 / 30)
    assert last > 80.0  # follows the ramp upward, some lag allowed


def test_keypoint_smoother_shape_and_smoothing():
    rng = np.random.default_rng(1)
    s = KeypointSmoother(num_keypoints=17, min_cutoff=0.5, beta=0.0)
    base = np.full((17, 2), 100.0)
    outs = []
    for _ in range(60):
        outs.append(s.update(base + rng.normal(0, 2.0, (17, 2)), 1 / 30))
    outs = np.array(outs)
    assert outs[0].shape == (17, 2)
    assert outs[20:, 0, 0].std() < 2.0  # smoothed below input noise std
    assert s.last is not None
