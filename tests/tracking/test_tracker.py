import numpy as np

from storepose.pipeline import FrameResult
from storepose.tracking.tracker import MultiObjectTracker


def make_result(boxes, kpts=None):
    boxes = np.array(boxes, float).reshape(-1, 4)
    n = len(boxes)
    if kpts is None:
        kpts = np.zeros((n, 17, 2), float)
    return FrameResult(boxes=boxes, keypoints=kpts, scores=np.ones((n, 17), float),
                       det_scores=np.full(n, 0.9, float))


def test_track_confirms_after_min_hits():
    tr = MultiObjectTracker(max_age=10, min_hits=3, iou_thr=0.3, smooth=False)
    box = [10, 10, 50, 90]
    assert tr.update(make_result([box]), 1 / 30) == []
    assert tr.update(make_result([box]), 1 / 30) == []
    out = tr.update(make_result([box]), 1 / 30)
    assert len(out) == 1 and out[0].id == 0 and out[0].coasting is False


def test_detector_score_propagates_and_clears_on_coast():
    tr = MultiObjectTracker(max_age=3, min_hits=1, iou_thr=0.3, smooth=False, coast=True)
    box = [10, 10, 50, 90]
    out = tr.update(make_result([box]), 1 / 30)
    assert abs(out[0].score - 0.9) < 1e-6     # detector score reaches the person
    coast = tr.update(make_result([]), 1 / 30)
    assert coast[0].coasting is True and coast[0].score is None  # cleared on coast


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
    tr = MultiObjectTracker(max_age=3, min_hits=1, iou_thr=0.3, smooth=False, coast=True)
    box = [10, 10, 50, 90]
    assert tr.update(make_result([box]), 1 / 30)[0].coasting is False
    o1 = tr.update(make_result([]), 1 / 30)
    assert len(o1) == 1 and o1[0].coasting is True and o1[0].keypoints is None
    tr.update(make_result([]), 1 / 30)
    assert len(tr.update(make_result([]), 1 / 30)) == 1  # t=3, still within max_age
    assert tr.update(make_result([]), 1 / 30) == []      # t=4, culled


def test_no_coast_default_omits_undetected_track():
    tr = MultiObjectTracker(max_age=5, min_hits=1, iou_thr=0.3, smooth=False)  # coast off
    box = [10, 10, 50, 90]
    assert tr.update(make_result([box]), 1 / 30)[0].coasting is False
    assert tr.update(make_result([]), 1 / 30) == []  # no detection -> not emitted


def test_coast_flag_emits_held_track():
    tr = MultiObjectTracker(max_age=5, min_hits=1, iou_thr=0.3, smooth=False, coast=True)
    box = [10, 10, 50, 90]
    tr.update(make_result([box]), 1 / 30)
    out = tr.update(make_result([]), 1 / 30)
    assert len(out) == 1 and out[0].coasting is True


def test_no_coast_keeps_id_when_detection_returns():
    # Suppressing coasting output must not lose identity: the track survives
    # internally, so a returning detection re-matches the same id by IoU.
    tr = MultiObjectTracker(max_age=5, min_hits=1, iou_thr=0.3, smooth=False)
    box = [10, 10, 50, 90]
    assert tr.update(make_result([box]), 1 / 30)[0].id == 0
    assert tr.update(make_result([]), 1 / 30) == []      # gap frame: not emitted
    out = tr.update(make_result([box]), 1 / 30)          # detection returns
    assert len(out) == 1 and out[0].id == 0              # same id, no new track


def test_stationary_track_suppressed_after_window():
    frame = np.zeros((400, 400, 3), np.uint8)
    tr = MultiObjectTracker(max_age=100, min_hits=1, iou_thr=0.3, smooth=False,
                            stationary_seconds=1.0, stationary_radius=0.05)
    box = [100, 100, 140, 180]
    first = tr.update(make_result([box]), 0.5, frame)
    assert len(first) == 1                      # emitted before enough history
    out = first
    for _ in range(5):                          # > 1s of no movement
        out = tr.update(make_result([box]), 0.5, frame)
    assert out == []                            # flagged as a static prop


def test_moving_track_not_suppressed_by_stationary_filter():
    frame = np.zeros((400, 400, 3), np.uint8)
    tr = MultiObjectTracker(max_age=100, min_hits=1, iou_thr=0.3, smooth=False,
                            stationary_seconds=1.0, stationary_radius=0.02)
    x, out = 50, None
    for _ in range(6):
        out = tr.update(make_result([[x, 100, x + 40, 180]]), 0.5, frame)
        x += 40                                 # keeps moving right
    assert len(out) == 1                        # a mover is never suppressed


def test_stationary_filter_off_by_default():
    frame = np.zeros((400, 400, 3), np.uint8)
    tr = MultiObjectTracker(max_age=100, min_hits=1, iou_thr=0.3, smooth=False)
    box = [100, 100, 140, 180]
    out = None
    for _ in range(8):
        out = tr.update(make_result([box]), 0.5, frame)
    assert len(out) == 1                        # no filter unless configured


def test_stationary_prop_excluded_from_reid():
    # A static prop must never be a re-id target: a moving detection elsewhere
    # with near-identical appearance (e.g. both dark) must NOT revive its id.
    frame = np.zeros((400, 400, 3), np.uint8)
    tr = MultiObjectTracker(max_age=2, min_hits=1, iou_thr=0.3, smooth=False,
                            appearance=_ConstScoreStub(0.99), reid=True,
                            reid_max_age=200, reid_thr=0.5,
                            stationary_seconds=1.0, stationary_radius=0.05)
    prop = [100, 100, 140, 180]
    for _ in range(5):                      # >1s stationary -> flagged as a prop
        tr.update(make_result([prop]), 0.5, frame)
    for _ in range(4):                      # age the prop out
        tr.update(make_result([]), 0.5, frame)
    out = tr.update(make_result([[300, 300, 340, 380]]), 0.5, frame)  # new, matching look
    assert out and all(p.id != 0 for p in out)   # prop's id 0 was not revived


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


class _ColorStub:
    """Appearance bound to the BGR pixel at a box center; gallery = list of colors.

    Lets tests bind identity to a painted color, exercising the real frame
    plumbing. ``score`` is exact-match against any gallery entry (1.0 / 0.0).
    """
    def extract(self, frame, box, keypoints, scores):
        cx = int((box[0] + box[2]) / 2.0); cy = int((box[1] + box[3]) / 2.0)
        h, w = frame.shape[:2]
        cx = min(max(cx, 0), w - 1); cy = min(max(cy, 0), h - 1)
        px = frame[cy, cx]
        return None if int(px.sum()) == 0 else px.astype(np.float32)

    def extract_batch(self, frame, boxes, keypoints, scores):
        return [self.extract(frame, b, None, None) for b in boxes]

    def new_memory(self, desc):
        return [] if desc is None else [desc]

    def update_memory(self, mem, desc):
        mem = list(mem or [])
        if desc is not None:
            mem.append(desc)
        return mem

    def score(self, mem, desc):
        if not mem or desc is None:
            return -1.0
        return 1.0 if any(np.array_equal(e, desc) for e in mem) else 0.0


class _ConstScoreStub:
    """Appearance whose score is a fixed value, to probe the gallery margin."""
    def __init__(self, value):
        self.value = value

    def extract(self, frame, box, keypoints, scores):
        return np.array([1.0], np.float32)

    def extract_batch(self, frame, boxes, keypoints, scores):
        return [np.array([1.0], np.float32) for _ in boxes]

    def new_memory(self, desc):
        return [desc]

    def update_memory(self, mem, desc):
        return list(mem or []) + ([] if desc is None else [desc])

    def score(self, mem, desc):
        return self.value


def _frame(boxes_colors, size=400):
    f = np.zeros((size, size, 3), np.uint8)
    for (x1, y1, x2, y2), color in boxes_colors:
        f[int(y1):int(y2), int(x1):int(x2)] = color
    return f


def _reid_tracker(max_age=3, reid_max_age=50, gallery_spatial_gate=True):
    return MultiObjectTracker(
        max_age=max_age, min_hits=1, iou_thr=0.3, smooth=False,
        appearance=_ColorStub(), reid=True, reid_max_age=reid_max_age, reid_thr=0.5,
        gallery_spatial_gate=gallery_spatial_gate,
    )


def test_reattach_same_appearance_revives_id():
    tr = _reid_tracker()
    box = [100, 100, 140, 180]
    red = (0, 0, 220)
    out = tr.update(make_result([box]), 1 / 30, _frame([(box, red)]))
    assert out[0].id == 0
    blank = np.zeros((400, 400, 3), np.uint8)
    for _ in range(5):                       # age out past max_age -> gallery
        tr.update(make_result([]), 1 / 30, blank)
    back = [110, 105, 150, 185]
    out2 = tr.update(make_result([back]), 1 / 30, _frame([(back, red)]))
    assert len(out2) == 1 and out2[0].id == 0   # same id revived


def test_reattach_emits_reid_notification_with_similarity():
    tr = MultiObjectTracker(max_age=2, min_hits=1, iou_thr=0.3, smooth=False,
                            appearance=_ConstScoreStub(0.78), reid=True,
                            reid_max_age=50, reid_thr=0.5)
    box = [100, 100, 140, 180]
    f = _frame([(box, (0, 0, 220))])
    out = tr.update(make_result([box]), 1 / 30, f)
    assert out[0].reid_notify is False        # fresh track: no re-id yet
    assert out[0].reid_sim is None
    blank = np.zeros((400, 400, 3), np.uint8)
    for _ in range(5):                        # age out to the gallery
        tr.update(make_result([]), 1 / 30, blank)
    back = [110, 105, 150, 185]
    out2 = tr.update(make_result([back]), 1 / 30, _frame([(back, (0, 0, 220))]))
    assert len(out2) == 1 and out2[0].id == 0
    assert out2[0].reid_notify is True        # re-attach armed the notification
    assert abs(out2[0].reid_sim - 0.78) < 1e-6


def test_reattach_different_appearance_gets_new_id():
    tr = _reid_tracker()
    box = [100, 100, 140, 180]
    tr.update(make_result([box]), 1 / 30, _frame([(box, (0, 0, 220))]))  # red, id 0
    blank = np.zeros((400, 400, 3), np.uint8)
    for _ in range(5):
        tr.update(make_result([]), 1 / 30, blank)
    back = [110, 105, 150, 185]
    out2 = tr.update(make_result([back]), 1 / 30, _frame([(back, (0, 220, 0))]))  # green
    assert out2[0].id == 1                       # different person -> new id


def test_reattach_past_ttl_gets_new_id():
    tr = _reid_tracker(max_age=2, reid_max_age=2)
    box = [100, 100, 140, 180]; red = (0, 0, 220)
    tr.update(make_result([box]), 1 / 30, _frame([(box, red)]))   # id 0
    blank = np.zeros((400, 400, 3), np.uint8)
    for _ in range(8):                            # past max_age AND reid TTL
        tr.update(make_result([]), 1 / 30, blank)
    back = [110, 105, 150, 185]
    out2 = tr.update(make_result([back]), 1 / 30, _frame([(back, red)]))
    assert out2[0].id == 1                        # gallery expired -> new id


def test_occluded_person_reattaches_without_swapping_neighbor():
    tr = _reid_tracker()
    a = [40, 100, 80, 180]; b = [300, 100, 340, 180]
    red, green = (0, 0, 220), (0, 220, 0)
    out = tr.update(make_result([a, b]), 1 / 30, _frame([(a, red), (b, green)]))
    assert len(out) == 2
    # A occluded for several frames; B stays
    for _ in range(5):
        tr.update(make_result([b]), 1 / 30, _frame([(b, green)]))
    a_back = [45, 105, 85, 185]
    out2 = tr.update(make_result([a_back, b]), 1 / 30, _frame([(a_back, red), (b, green)]))
    assert len(out2) == 2 and len({p.id for p in out2}) == 2   # two distinct ids, no merge


def test_reid_disabled_returns_new_id_on_reappearance():
    tr = MultiObjectTracker(max_age=2, min_hits=1, iou_thr=0.3, smooth=False)  # reid off
    box = [100, 100, 140, 180]
    tr.update(make_result([box]), 1 / 30)        # no frame arg
    for _ in range(4):
        tr.update(make_result([]), 1 / 30)
    out = tr.update(make_result([[110, 105, 150, 185]]), 1 / 30)
    assert out[0].id == 1


def test_reattach_anchors_to_last_seen_not_coasted_position():
    # A person moving, lost beyond max_age, then reappearing near where they were
    # LAST SEEN must keep their id. The spatial gate must anchor to the
    # last-detected position, not the Kalman-extrapolated (coasted) one, which
    # drifts far ahead for a mover and would otherwise reject the true return.
    # Uses realistic params (min_hits=3, smoothing on, max_age 12) so the gallery
    # path is exercised.
    tr = MultiObjectTracker(max_age=12, min_hits=3, iou_thr=0.3, smooth=True,
                            appearance=_ColorStub(), reid=True, reid_max_age=40, reid_thr=0.6)
    red = (0, 0, 220)
    x = 100
    out = None
    for _ in range(6):                       # confirm + build rightward velocity
        b = [x, 100, x + 60, 260]
        out = tr.update(make_result([b]), 1 / 8, _frame([(b, red)], size=600))
        x += 25
    assert out[0].id == 0
    last_seen_x = x - 25
    blank = np.zeros((600, 600, 3), np.uint8)
    for _ in range(20):                      # lost well past max_age -> gallery
        tr.update(make_result([]), 1 / 8, blank)
    back = [last_seen_x, 100, last_seen_x + 60, 260]   # return to the last-seen spot
    out2 = tr.update(make_result([back]), 1 / 8, _frame([(back, red)], size=600))
    assert len(out2) == 1 and out2[0].id == 0   # id preserved, not a fresh spawn


def test_gallery_reattach_is_location_gated_by_default():
    # A fully-lost (gallery) track may only be revived within a plausibility
    # radius around its last-seen center. A same-colored detection on the
    # opposite side of the frame is too far to be the same person -> new id,
    # not a false cross-frame revival (the over-merging failure mode).
    tr = _reid_tracker()  # max_age=3, reid_max_age=50, gate ON
    left = [20, 180, 60, 260]; red = (0, 0, 220)
    out = tr.update(make_result([left]), 1 / 30, _frame([(left, red)]))
    assert out[0].id == 0
    blank = np.zeros((400, 400, 3), np.uint8)
    for _ in range(6):                 # age out past max_age -> gallery
        tr.update(make_result([]), 1 / 30, blank)
    far = [340, 180, 380, 260]         # opposite side, beyond the spatial gate
    out2 = tr.update(make_result([far]), 1 / 30, _frame([(far, red)]))
    assert len(out2) == 1 and out2[0].id == 1   # too far -> new id, no teleport


def test_gallery_reattaches_across_full_frame_exit_when_gate_disabled():
    # --no-gallery-spatial-gate restores the old appearance-only cross-exit
    # re-entry: same color, opposite side, same id.
    tr = _reid_tracker(gallery_spatial_gate=False)
    left = [20, 180, 60, 260]; red = (0, 0, 220)
    out = tr.update(make_result([left]), 1 / 30, _frame([(left, red)]))
    assert out[0].id == 0
    blank = np.zeros((400, 400, 3), np.uint8)
    for _ in range(6):                 # age out past max_age -> gallery
        tr.update(make_result([]), 1 / 30, blank)
    far = [340, 180, 380, 260]         # opposite side, beyond the spatial gate
    out2 = tr.update(make_result([far]), 1 / 30, _frame([(far, red)]))
    assert len(out2) == 1 and out2[0].id == 0   # appearance-only -> revived


def test_gallery_threshold_is_stricter_than_active():
    # score 0.52 clears the active floor (0.5) but not the gallery floor (0.55)
    tr = MultiObjectTracker(
        max_age=2, min_hits=1, iou_thr=0.3, smooth=False,
        appearance=_ConstScoreStub(0.52), reid=True, reid_max_age=50, reid_thr=0.5,
    )
    box = [100, 100, 140, 180]
    f = _frame([(box, (0, 0, 220))])
    assert tr.update(make_result([box]), 1 / 30, f)[0].id == 0
    blank = np.zeros((400, 400, 3), np.uint8)
    for _ in range(4):                 # age out -> gallery
        tr.update(make_result([]), 1 / 30, blank)
    back = [110, 105, 150, 185]
    out = tr.update(make_result([back]), 1 / 30, _frame([(back, (0, 0, 220))]))
    assert out[0].id == 1              # gallery match rejected -> new id
