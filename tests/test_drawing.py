import numpy as np

from storepose.config import AppConfig
from storepose.drawing import annotate, annotate_busy, annotate_tracked
from storepose.pipeline import FrameResult
from storepose.pose import NUM_KEYPOINTS
from storepose.tracking.types import TrackedPerson


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


def test_annotate_tracked_draws_box_and_skeleton():
    frame = _blank()
    p = TrackedPerson(
        id=2, box=np.array([20, 20, 100, 100], float),
        keypoints=np.full((NUM_KEYPOINTS, 2), 50.0), scores=np.ones(NUM_KEYPOINTS),
        coasting=False, color=(0, 255, 0),
    )
    out = annotate_tracked(frame, [p], AppConfig(), fps=30.0)
    assert out.shape == frame.shape
    assert (out[:, :, 1] > 0).sum() > 0
    assert np.array_equal(frame, _blank())  # input untouched


def test_annotate_tracked_coasting_has_no_pose_and_no_crash():
    p = TrackedPerson(
        id=1, box=np.array([10, 10, 50, 50], float),
        keypoints=None, scores=None, coasting=True, color=(255, 0, 0),
    )
    out = annotate_tracked(_blank(), [p], AppConfig(), fps=None)
    assert out.shape == (120, 160, 3)


def test_annotate_queue_draws_zone_and_count():
    from storepose.drawing import annotate_queue
    from storepose.queue.types import PersonStatus, QueueResult
    from storepose.queue.zone import Zone
    frame = _blank()
    person = TrackedPerson(id=1, box=np.array([20, 20, 60, 100], float),
                           keypoints=None, scores=None, coasting=False, color=(0, 255, 0))
    result = QueueResult(
        statuses=[PersonStatus(id=1, waiting=True, candidate=False, progress=1.0, wait_seconds=3.4)],
        count=1,
    )
    zone = Zone([(0, 0), (120, 0), (120, 110), (0, 110)])
    out = annotate_queue(frame.copy(), [person], result, zone, AppConfig())
    assert out.shape == frame.shape
    assert (out[:, :, 2] > 0).sum() > 0  # zone/tag drawn (orange has red channel)


def test_annotate_queue_safe_without_zone():
    from storepose.drawing import annotate_queue
    from storepose.queue.types import QueueResult
    out = annotate_queue(_blank(), [], QueueResult(statuses=[], count=0), None, AppConfig())
    assert out.shape == (120, 160, 3)


def test_annotate_queue_candidate_fill_draws():
    from storepose.drawing import annotate_queue
    from storepose.queue.types import PersonStatus, QueueResult
    from storepose.queue.zone import Zone
    frame = _blank()
    person = TrackedPerson(id=7, box=np.array([20, 20, 60, 100], float),
                           keypoints=None, scores=None, coasting=False, color=(255, 0, 0))
    result = QueueResult(
        statuses=[PersonStatus(id=7, waiting=False, candidate=True, progress=0.6, wait_seconds=0.0)],
        count=0,
    )
    zone = Zone([(0, 0), (120, 0), (120, 110), (0, 110)])
    out = annotate_queue(frame.copy(), [person], result, zone, AppConfig())
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)  # candidate fill drawn


def test_annotate_busy_draws_without_crash_and_keeps_shape():
    frame = _blank()
    out = annotate_busy(frame, "High", 4.0, window_remaining_s=42.0)
    assert out.shape == frame.shape
    assert out.any()  # the badge drew some non-zero pixels
