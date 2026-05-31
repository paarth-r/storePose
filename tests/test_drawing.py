import numpy as np

from storepose.config import AppConfig
from storepose.drawing import annotate
from storepose.pipeline import FrameResult
from storepose.pose import NUM_KEYPOINTS


def _blank():
    return np.zeros((120, 160, 3), np.uint8)


def test_annotate_empty_returns_same_shape_and_does_not_mutate_input():
    frame = _blank()
    result = FrameResult(
        boxes=np.empty((0, 4), np.float32),
        keypoints=np.empty((0, NUM_KEYPOINTS, 2), np.float32),
        scores=np.empty((0, NUM_KEYPOINTS), np.float32),
    )
    out = annotate(frame, result, AppConfig(), fps=12.0)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)  # FPS/header text drawn
    assert np.array_equal(frame, _blank())  # original untouched


def test_annotate_draws_box_pixels_for_a_person():
    frame = _blank()
    boxes = np.array([[20, 20, 100, 100]], np.float32)
    kpts = np.full((1, NUM_KEYPOINTS, 2), 50.0, np.float32)
    scores = np.ones((1, NUM_KEYPOINTS), np.float32)
    result = FrameResult(boxes=boxes, keypoints=kpts, scores=scores)
    out = annotate(frame, result, AppConfig(), fps=30.0)
    # green box edge should appear somewhere on the frame
    assert (out[:, :, 1] > 0).sum() > 0
    assert out.shape == frame.shape
