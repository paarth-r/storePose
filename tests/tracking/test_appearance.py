import numpy as np

from storepose.tracking.appearance import HsvHistogramAppearance


def _frame_with_rect(color, x1, y1, x2, y2, size=200):
    f = np.zeros((size, size, 3), np.uint8)
    f[y1:y2, x1:x2] = color  # BGR
    return f


def test_same_color_crop_high_similarity():
    app = HsvHistogramAppearance(kpt_thr=0.5)
    box = np.array([40, 40, 120, 160], float)
    fa = _frame_with_rect((0, 0, 220), 40, 40, 120, 160)   # red person
    fb = _frame_with_rect((0, 0, 220), 50, 45, 130, 165)   # same red, shifted
    a = app.extract(fa, box, None, None)
    b = app.extract(fb, np.array([50, 45, 130, 165], float), None, None)
    assert a is not None and b is not None
    assert app.similarity(a, b) > 0.8


def test_different_color_low_similarity():
    app = HsvHistogramAppearance(kpt_thr=0.5)
    box = np.array([40, 40, 120, 160], float)
    red = app.extract(_frame_with_rect((0, 0, 220), 40, 40, 120, 160), box, None, None)
    blue = app.extract(_frame_with_rect((220, 0, 0), 40, 40, 120, 160), box, None, None)
    assert red is not None and blue is not None
    assert app.similarity(red, blue) < 0.3


def test_keypoints_localize_torso_over_box():
    app = HsvHistogramAppearance(kpt_thr=0.5)
    # box spans a green background; only the torso quad is red
    f = np.zeros((200, 200, 3), np.uint8)
    f[20:180, 20:120] = (0, 200, 0)      # green box area
    f[60:120, 50:90] = (0, 0, 220)       # red torso
    box = np.array([20, 20, 120, 180], float)
    kpts = np.zeros((17, 2), float)
    kpts[5] = (50, 60); kpts[6] = (90, 60); kpts[11] = (50, 120); kpts[12] = (90, 120)
    scores = np.ones(17, float)
    desc = app.extract(f, box, kpts, scores)
    red_only = app.extract(_frame_with_rect((0, 0, 220), 50, 60, 90, 120), box, None, None)
    assert app.similarity(desc, red_only) > 0.8  # torso (red) not box (green)


def test_degenerate_crop_returns_none():
    app = HsvHistogramAppearance(kpt_thr=0.5)
    f = np.zeros((200, 200, 3), np.uint8)            # all black -> fully masked
    assert app.extract(f, np.array([10, 10, 60, 120], float), None, None) is None
    tiny = app.extract(f, np.array([10, 10, 11, 11], float), None, None)
    assert tiny is None
