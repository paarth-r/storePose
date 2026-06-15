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
        det_scores=np.empty((0,), np.float32),
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
    result = FrameResult(boxes=boxes, keypoints=kpts, scores=scores,
                         det_scores=np.array([0.9], np.float32))
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


def test_conf_overlay_changes_tracked_label():
    frame = _blank()
    p = TrackedPerson(
        id=2, box=np.array([20, 20, 100, 100], float),
        keypoints=None, scores=None, coasting=False, color=(0, 255, 0),
        score=0.91,
    )
    without = annotate_tracked(frame, [p], AppConfig(show_conf=False), fps=None)
    with_conf = annotate_tracked(frame, [p], AppConfig(show_conf=True), fps=None)
    assert not np.array_equal(without, with_conf)  # confidence text drawn


def test_conf_overlay_skipped_when_score_none():
    # coasting person has score None -> --conf draws nothing extra, no crash
    frame = _blank()
    p = TrackedPerson(
        id=1, box=np.array([20, 20, 100, 100], float),
        keypoints=None, scores=None, coasting=True, color=(0, 255, 0), score=None,
    )
    a = annotate_tracked(frame, [p], AppConfig(show_conf=False), fps=None)
    b = annotate_tracked(frame, [p], AppConfig(show_conf=True), fps=None)
    assert np.array_equal(a, b)


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


def test_queue_waiting_fill_uses_person_color_not_green():
    from storepose.drawing import annotate_queue
    from storepose.queue.types import PersonStatus, QueueResult
    from storepose.queue.zone import Zone
    frame = _blank()
    blue = (255, 0, 0)  # BGR blue person color
    person = TrackedPerson(id=1, box=np.array([20, 20, 100, 110], float),
                           keypoints=None, scores=None, coasting=False, color=blue)
    result = QueueResult(
        statuses=[PersonStatus(id=1, waiting=True, candidate=False, progress=1.0, wait_seconds=3.4)],
        count=1,
    )
    zone = Zone([(0, 0), (160, 0), (160, 120), (0, 120)])
    out = annotate_queue(frame.copy(), [person], result, zone, AppConfig())
    cy, cx = 65, 60  # inside the box
    b, g, r = out[cy, cx]
    assert b > g and b > r  # tinted toward the person's blue, not green


def test_queue_candidate_fill_uses_person_color_not_orange():
    from storepose.drawing import annotate_queue
    from storepose.queue.types import PersonStatus, QueueResult
    from storepose.queue.zone import Zone
    frame = _blank()
    blue = (255, 0, 0)
    person = TrackedPerson(id=3, box=np.array([20, 20, 100, 110], float),
                           keypoints=None, scores=None, coasting=False, color=blue)
    result = QueueResult(
        statuses=[PersonStatus(id=3, waiting=False, candidate=True, progress=0.9, wait_seconds=0.0)],
        count=0,
    )
    zone = Zone([(0, 0), (160, 0), (160, 120), (0, 120)])
    out = annotate_queue(frame.copy(), [person], result, zone, AppConfig())
    cy, cx = 100, 60  # inside the rising fill near the box bottom
    b, g, r = out[cy, cx]
    assert b > r  # person blue, not orange (orange would have r >= b)


def test_palette_has_no_near_orange_color():
    from storepose.tracking.track import _PALETTE
    for b, g, r in _PALETTE:
        # an orange is low-blue, mid/high-green, high-red; ensure none collide
        assert not (b < 80 and g > 120 and r > 200)


def test_annotate_queue_draws_pos_zone_and_serving_tag():
    from storepose.drawing import annotate_queue, POS_COLOR
    from storepose.queue.types import PersonStatus, QueueResult
    from storepose.queue.zone import Zone
    frame = _blank()
    teal = (200, 200, 0)
    person = TrackedPerson(id=1, box=np.array([20, 20, 100, 110], float),
                           keypoints=None, scores=None, coasting=False, color=teal)
    result = QueueResult(
        statuses=[PersonStatus(id=1, waiting=False, candidate=False, progress=1.0,
                               wait_seconds=0.0, serving=True, serving_seconds=4.2)],
        count=0, serving_count=1,
    )
    line_zone = Zone([(0, 0), (160, 0), (160, 120), (0, 120)])
    pos_zone = Zone([(80, 0), (160, 0), (160, 120), (80, 120)])
    out = annotate_queue(frame.copy(), [person], result, line_zone, AppConfig(),
                         pos_zone=pos_zone)
    assert out.shape == frame.shape
    assert (out[:, :, 0] > 0).sum() > 0  # POS_COLOR has a blue channel


def test_annotate_queue_pos_zone_optional():
    from storepose.drawing import annotate_queue
    from storepose.queue.types import QueueResult
    out = annotate_queue(_blank(), [], QueueResult(statuses=[], count=0), None, AppConfig())
    assert out.shape == (120, 160, 3)


def test_annotate_queue_draws_all_contours():
    from storepose.drawing import annotate_queue
    from storepose.queue.types import QueueResult
    from storepose.queue.zone import Zone
    frame = _blank()  # 120x160
    zone = Zone.from_polygons([
        [(0, 0), (40, 0), (40, 40), (0, 40)],
        [(110, 70), (150, 70), (150, 110), (110, 110)],
    ])
    out = annotate_queue(frame.copy(), [], QueueResult(statuses=[], count=0),
                         zone, AppConfig())
    assert out[5, 5].sum() > 0
    assert out[90, 130].sum() > 0


def _pos_setup(serving):
    from storepose.queue.types import PersonStatus, QueueResult
    from storepose.queue.zone import Zone
    person = TrackedPerson(id=3, box=np.array([10, 10, 30, 38], float),
                           keypoints=None, scores=None, coasting=False, color=(255, 0, 0))
    if serving:
        st = PersonStatus(id=3, waiting=False, candidate=False, progress=1.0,
                          wait_seconds=0.0, serving=True, serving_seconds=4.2)
        res = QueueResult(statuses=[st], count=0, serving_count=1)
    else:
        st = PersonStatus(id=3, waiting=True, candidate=False, progress=1.0, wait_seconds=1.0)
        res = QueueResult(statuses=[st], count=1, serving_count=0)
    zone = Zone([(0, 0), (40, 0), (40, 40), (0, 40)])  # top-left only
    return person, res, zone


def test_pos_panel_drawn_when_serving():
    from storepose.drawing import annotate_queue
    person, res, zone = _pos_setup(serving=True)
    out = annotate_queue(_blank().copy(), [person], res, zone, AppConfig())
    assert out[95:120, :, :].sum() > 0  # panel text near the bottom


def test_pos_panel_absent_when_nobody_serving():
    from storepose.drawing import annotate_queue
    person, res, zone = _pos_setup(serving=False)
    out = annotate_queue(_blank().copy(), [person], res, zone, AppConfig())
    assert out[95:120, :, :].sum() == 0  # nothing at the bottom


def test_annotate_queue_draws_alt_zone_and_reg_tag():
    from storepose.drawing import annotate_queue, ALT_COLOR
    from storepose.queue.types import PersonStatus, QueueResult
    from storepose.queue.zone import Zone
    frame = _blank()
    person = TrackedPerson(id=1, box=np.array([20, 20, 100, 110], float),
                           keypoints=None, scores=None, coasting=False, color=(200, 200, 0))
    result = QueueResult(
        statuses=[PersonStatus(id=1, waiting=False, candidate=False, progress=1.0,
                               wait_seconds=0.0, serving=False, serving_seconds=4.0,
                               serving_other=True)],
        count=0, serving_count=0, serving_other_count=1,
    )
    line = Zone([(0, 0), (160, 0), (160, 120), (0, 120)])
    alt = Zone([(80, 0), (160, 0), (160, 120), (80, 120)])
    out = annotate_queue(frame.copy(), [person], result, line, AppConfig(), alt_zone=alt)
    assert out.shape == frame.shape
    # ALT_COLOR is red (high red channel) -> some red pixels from the zone + tag
    assert (out[:, :, 2] > 120).sum() > 0
