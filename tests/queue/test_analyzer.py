import numpy as np

from storepose.queue.analyzer import QueueAnalyzer
from storepose.queue.zone import Zone
from storepose.tracking.types import TrackedPerson

ZONE = Zone([(0, 0), (200, 0), (200, 200), (0, 200)])


def person(pid, box):
    return TrackedPerson(
        id=pid, box=np.array(box, float),
        keypoints=None, scores=None, coasting=False, color=(0, 255, 0),
    )


def test_stationary_in_zone_becomes_waiting():
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0)
    box = [40, 40, 60, 80]  # foot (50, 80) inside, height 40
    assert an.update([person(1, box)], 0.5).statuses[0].waiting is False
    r = an.update([person(1, box)], 0.5)  # in_streak reaches 1.0
    assert r.statuses[0].waiting is True
    assert r.count == 1


def test_moving_in_zone_still_counts_no_motion_gating():
    # No motion gating: a person moving around inside the zone (e.g. fetching
    # items / pushing a cart) still counts once in-zone for enter_frames.
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0)
    r = None
    for xc in (10, 40, 70, 100):  # moving, but all inside the zone
        r = an.update([person(1, [xc, 40, xc + 20, 80])], 0.5)
    assert r.statuses[0].waiting is True
    assert r.count == 1


def test_leaving_zone_ends_wait_and_emits_completed():
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0)
    box = [40, 40, 60, 80]
    an.update([person(1, box)], 0.5)
    an.update([person(1, box)], 0.5)  # waiting now
    out = [400, 40, 420, 80]  # foot (410, 80) outside zone
    an.update([person(1, out)], 0.5)  # out_streak 0.5
    r = an.update([person(1, out)], 0.5)  # out_streak 1.0 -> ends
    assert r.statuses[0].waiting is False
    assert len(r.completed) == 1
    assert r.completed[0].id == 1
    assert r.completed[0].wait_seconds > 0


def test_disappearance_finalizes_wait():
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=5.0)
    box = [40, 40, 60, 80]
    an.update([person(1, box)], 0.5)
    an.update([person(1, box)], 0.5)  # waiting
    r = an.update([], 0.5)  # track id 1 vanished
    assert r.count == 0
    assert len(r.completed) == 1 and r.completed[0].id == 1


def test_reid_grace_resumes_timer_after_brief_disappearance():
    # A waiting person whose track vanishes (lost into the re-id gallery) and is
    # re-identified within the grace window must RESUME their saved wait timer,
    # not reset it.
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=5.0, reid_grace_seconds=3.0)
    box = [40, 40, 60, 80]
    an.update([person(1, box)], 0.5)
    an.update([person(1, box)], 0.5)        # waiting
    r = an.update([person(1, box)], 0.5)    # accruing wait
    before = r.statuses[0].wait_seconds
    assert before > 0
    a1 = an.update([], 0.5)                  # vanished (within grace) -> paused
    a2 = an.update([], 0.5)
    assert a1.completed == [] and a2.completed == []
    assert a1.count == 0                     # not shown while gone
    r2 = an.update([person(1, box)], 0.5)    # re-id revives same id -> resume
    assert len(r2.completed) == 0
    assert r2.statuses[0].waiting is True
    assert r2.statuses[0].wait_seconds >= before          # resumed, not reset
    r3 = an.update([person(1, box)], 0.5)
    assert r3.statuses[0].wait_seconds > r2.statuses[0].wait_seconds  # keeps climbing


def test_reid_grace_finalizes_if_gone_past_window():
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=5.0, reid_grace_seconds=1.0)
    box = [40, 40, 60, 80]
    an.update([person(1, box)], 0.5)
    an.update([person(1, box)], 0.5)        # waiting
    r1 = an.update([], 0.5)                  # absent 0.5 < 1.0 -> paused
    assert r1.completed == []
    r2 = an.update([], 0.5)                  # absent 1.0 >= 1.0 -> finalize
    assert len(r2.completed) == 1 and r2.completed[0].id == 1


# A POS zone occupying the right half; the line ZONE is the full (0,0)-(200,200).
POS = Zone([(120, 0), (200, 0), (200, 200), (120, 200)])


def pos_person(pid, x):
    # foot center at (x, 80); x>=120 is "at POS", x<120 is "waiting region"
    return person(pid, [x - 10, 40, x + 10, 80])


def test_waiting_then_serving_then_served():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=1.0,
                       pos_enter_frames=1)
    an.update([pos_person(1, 40)], 0.5)
    r = an.update([pos_person(1, 40)], 0.5)
    assert r.statuses[0].waiting is True and r.statuses[0].serving is False
    an.update([pos_person(1, 40)], 0.5)              # accruing waiting
    r2 = an.update([pos_person(1, 160)], 0.5)        # step into POS -> SERVING
    assert r2.statuses[0].serving is True
    assert r2.serving_count == 1 and r2.count == 0
    an.update([pos_person(1, 160)], 0.5)             # accruing serving
    an.update([pos_person(1, 400)], 0.5)             # leave POS
    r3 = an.update([pos_person(1, 400)], 0.5)        # >= exit_seconds -> SERVED
    assert len(r3.completed) == 1
    c = r3.completed[0]
    assert c.outcome == "served" and c.wait_seconds > 0 and c.serving_seconds > 0


def test_waiting_then_abandoned_before_pos():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=1.0)
    an.update([pos_person(1, 40)], 0.5)
    an.update([pos_person(1, 40)], 0.5)              # WAITING
    an.update([pos_person(1, 400)], 0.5)             # out of line
    r = an.update([pos_person(1, 400)], 0.5)         # out >= exit_seconds -> ABANDONED
    assert len(r.completed) == 1
    assert r.completed[0].outcome == "abandoned"
    assert r.completed[0].serving_seconds == 0.0


def test_walkup_straight_to_pos_has_no_waiting():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=1.0)
    an.update([pos_person(1, 160)], 0.5)             # directly at POS
    r = an.update([pos_person(1, 160)], 0.5)         # 2 frames -> SERVING
    assert r.statuses[0].serving is True
    an.update([pos_person(1, 400)], 0.5)
    r2 = an.update([pos_person(1, 400)], 0.5)        # leave -> SERVED
    assert r2.completed[0].outcome == "served"
    assert r2.completed[0].wait_seconds == 0.0


def test_gap_while_serving_adds_to_serving_seconds():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=5.0,
                       reid_grace_seconds=3.0)
    an.update([pos_person(1, 160)], 0.5)
    r = an.update([pos_person(1, 160)], 0.5)         # SERVING
    before = r.statuses[0].serving_seconds
    an.update([], 0.5)                               # vanished (within grace)
    an.update([], 0.5)
    r2 = an.update([pos_person(1, 160)], 0.5)        # re-id -> resume serving
    assert r2.statuses[0].serving is True
    assert r2.statuses[0].serving_seconds >= before + 1.0


def test_gap_while_waiting_attributed_to_waiting_even_if_returns_at_pos():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=5.0,
                       reid_grace_seconds=3.0, pos_enter_frames=1)
    an.update([pos_person(1, 40)], 0.5)
    r = an.update([pos_person(1, 40)], 0.5)          # WAITING
    wait_before = r.statuses[0].wait_seconds
    an.update([], 0.5)                               # vanished while waiting
    an.update([], 0.5)
    r2 = an.update([pos_person(1, 160)], 0.5)        # returns at POS
    assert r2.statuses[0].wait_seconds >= wait_before + 1.0
    assert r2.statuses[0].serving is True


def test_gap_past_grace_finalizes_with_outcome():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=5.0,
                       reid_grace_seconds=1.0)
    an.update([pos_person(1, 160)], 0.5)
    an.update([pos_person(1, 160)], 0.5)             # SERVING (reached POS)
    an.update([], 0.5)                               # absent 0.5 < 1.0
    r = an.update([], 0.5)                           # absent 1.0 >= grace -> finalize
    assert len(r.completed) == 1 and r.completed[0].outcome == "served"


def test_pos_entry_debounced():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=5.0,
                       pos_enter_frames=3)
    an.update([pos_person(1, 40)], 0.5)
    an.update([pos_person(1, 40)], 0.5)              # WAITING
    an.update([pos_person(1, 160)], 0.5)             # in POS frame 1
    r = an.update([pos_person(1, 160)], 0.5)         # frame 2 (< 3): not serving yet
    # box is in POS, so they've left the line count immediately (debouncing -> neither)
    assert r.statuses[0].waiting is False and r.statuses[0].serving is False
    assert r.count == 0
    r2 = an.update([pos_person(1, 160)], 0.5)        # frame 3 -> SERVING
    assert r2.statuses[0].serving is True


def test_pos_graze_resets_and_stays_waiting():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=5.0,
                       pos_enter_frames=3)
    an.update([pos_person(1, 40)], 0.5)
    an.update([pos_person(1, 40)], 0.5)              # WAITING
    an.update([pos_person(1, 160)], 0.5)             # 1 frame grazing POS
    rb = an.update([pos_person(1, 40)], 0.5)         # back in line -> pos_frames reset
    assert rb.statuses[0].waiting is True            # back in the line count
    an.update([pos_person(1, 160)], 0.5)             # 1
    r = an.update([pos_person(1, 160)], 0.5)         # 2 (< 3) -> graze never reaches serving
    assert r.statuses[0].serving is False


def test_pos_priority_when_in_both_zones():
    # Ankle inside the overlap of the line zone and POS zone: the person must be
    # counted only as serving (POS), never simultaneously as waiting.
    import numpy as np
    pos = Zone([(100, 0), (200, 0), (200, 200), (100, 200)])
    an = QueueAnalyzer(ZONE, pos_zone=pos, enter_frames=2, exit_seconds=2.0, pos_enter_frames=1)
    k = np.zeros((17, 2)); k[15] = (150, 80); k[16] = (150, 80)  # ankle in both zones
    p = TrackedPerson(id=1, box=np.array([140, 40, 160, 100], float),
                      keypoints=k, scores=np.ones(17), coasting=False, color=(0, 255, 0))
    an.update([p], 0.5)
    r = an.update([p], 0.5)
    assert r.statuses[0].serving is True
    assert r.statuses[0].waiting is False
    assert r.serving_count == 1 and r.count == 0


def test_single_zone_completed_wait_is_served_not_abandoned():
    # No POS zone -> no abandonment concept -> completed waits are "served",
    # so single-zone --busy keeps receiving them.
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0)
    box = [40, 40, 60, 80]
    an.update([person(1, box)], 0.5)
    an.update([person(1, box)], 0.5)  # waiting
    out = [400, 40, 420, 80]
    an.update([person(1, out)], 0.5)
    r = an.update([person(1, out)], 0.5)  # leaves -> completed
    assert len(r.completed) == 1
    assert r.completed[0].outcome == "served"


def test_candidate_progress_then_inclusion():
    an = QueueAnalyzer(ZONE, enter_frames=5, exit_seconds=1.0)
    box = [40, 40, 60, 80]
    an.update([person(1, box)], 0.1)  # frame 1
    r2 = an.update([person(1, box)], 0.1)  # frame 2 -> 2/5
    s = r2.statuses[0]
    assert s.waiting is False and s.candidate is True
    assert abs(s.progress - 0.4) < 1e-6
    an.update([person(1, box)], 0.1)  # 3
    an.update([person(1, box)], 0.1)  # 4
    r5 = an.update([person(1, box)], 0.1)  # 5 -> waiting
    assert r5.statuses[0].waiting is True
    assert r5.statuses[0].candidate is False
    assert r5.statuses[0].progress == 1.0


def _kpts_with_ankles(lx, ly, rx, ry, score=0.9):
    k = np.zeros((17, 2), float)
    s = np.zeros(17, float)
    k[15] = (lx, ly); k[16] = (rx, ry)
    s[15] = s[16] = score
    return k, s


def person_pose(pid, box, kpts, scores):
    return TrackedPerson(id=pid, box=np.array(box, float),
                         keypoints=kpts, scores=scores, coasting=False, color=(0, 255, 0))


def test_ankle_inside_counts_even_if_box_mostly_outside():
    # box is outside the zone, but the ankles are inside -> in zone via ankles
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0, kpt_thr=0.5)
    k, s = _kpts_with_ankles(50, 50, 60, 50)  # well inside ZONE (0..200)
    box = [300, 300, 360, 400]  # outside zone
    an.update([person_pose(1, box, k, s)], 0.5)
    r = an.update([person_pose(1, box, k, s)], 0.5)
    assert r.statuses[0].waiting is True


def test_occluded_ankles_fall_back_to_coverage():
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0,
                       kpt_thr=0.5, coverage_thr=0.5)
    k, s = _kpts_with_ankles(50, 50, 60, 50, score=0.1)  # ankles low confidence
    box = [40, 40, 120, 160]  # mostly inside ZONE -> coverage high
    an.update([person_pose(1, box, k, s)], 0.5)
    r = an.update([person_pose(1, box, k, s)], 0.5)
    assert r.statuses[0].waiting is True


def test_occluded_ankles_box_outside_not_waiting():
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0,
                       kpt_thr=0.5, coverage_thr=0.5)
    k, s = _kpts_with_ankles(50, 50, 60, 50, score=0.1)
    box = [400, 400, 480, 560]  # outside zone
    an.update([person_pose(1, box, k, s)], 0.5)
    r = an.update([person_pose(1, box, k, s)], 0.5)
    assert r.statuses[0].waiting is False


def test_ankle_outside_but_box_covered_stays_in_zone():
    # Ankle keypoints are confident but OUTSIDE the zone, yet the box is mostly
    # inside -> OR keeps the person in-zone (timer must not reset).
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0,
                       kpt_thr=0.5, coverage_thr=0.5)
    k, s = _kpts_with_ankles(500, 500, 510, 500, score=0.9)  # outside ZONE (0..200)
    box = [40, 40, 120, 160]  # mostly inside ZONE -> coverage high
    an.update([person_pose(1, box, k, s)], 0.5)
    r = an.update([person_pose(1, box, k, s)], 0.5)
    assert r.statuses[0].waiting is True


def test_wait_not_reset_when_ankle_leaves_but_box_covered():
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0,
                       kpt_thr=0.5, coverage_thr=0.5)
    box = [40, 40, 120, 160]  # inside zone
    k_in, s_in = _kpts_with_ankles(80, 150, 90, 150, score=0.9)
    an.update([person_pose(1, box, k_in, s_in)], 0.5)
    an.update([person_pose(1, box, k_in, s_in)], 0.5)  # waiting now
    # ankle now confidently OUTSIDE the zone, but box still covered
    k_out, s_out = _kpts_with_ankles(500, 500, 510, 500, score=0.9)
    r = an.update([person_pose(1, box, k_out, s_out)], 0.5)
    assert r.statuses[0].waiting is True  # not reset
    assert r.statuses[0].wait_seconds > 0
    assert len(r.completed) == 0


def test_candidate_progress_not_reset_by_brief_ankle_dropout():
    # Box sits OUTSIDE the coverage zone, so in-zone depends only on the ankle.
    # The ankle flashes out for one frame; progress must hold (grace), so the
    # person still reaches inclusion.
    an = QueueAnalyzer(ZONE, enter_frames=3, exit_seconds=1.0,
                       kpt_thr=0.5, coverage_thr=0.5)
    box = [400, 400, 480, 560]  # outside ZONE -> coverage 0
    k_in, s_in = _kpts_with_ankles(50, 50, 60, 50, score=0.9)   # ankle inside zone
    k_out, s_out = _kpts_with_ankles(50, 50, 60, 50, score=0.1)  # ankle not visible

    an.update([person_pose(1, box, k_in, s_in)], 0.1)   # in_frames 1
    an.update([person_pose(1, box, k_in, s_in)], 0.1)   # in_frames 2
    an.update([person_pose(1, box, k_out, s_out)], 0.1)  # dropout -> hold at 2
    r = an.update([person_pose(1, box, k_in, s_in)], 0.1)  # 3 -> waiting
    assert r.statuses[0].waiting is True


def test_candidate_resets_after_sustained_loss():
    an = QueueAnalyzer(ZONE, enter_frames=3, exit_seconds=0.3,
                       kpt_thr=0.5, coverage_thr=0.5)
    box = [400, 400, 480, 560]  # outside zone
    k_in, s_in = _kpts_with_ankles(50, 50, 60, 50, score=0.9)
    k_out, s_out = _kpts_with_ankles(50, 50, 60, 50, score=0.1)
    an.update([person_pose(1, box, k_in, s_in)], 0.1)   # in_frames 1
    an.update([person_pose(1, box, k_in, s_in)], 0.1)   # in_frames 2
    # sustained loss > exit_seconds (0.3): 4 * 0.1 = 0.4 -> reset
    for _ in range(4):
        an.update([person_pose(1, box, k_out, s_out)], 0.1)
    r = an.update([person_pose(1, box, k_in, s_in)], 0.1)  # only in_frames 1 again
    assert r.statuses[0].waiting is False


def test_standing_person_triggers_via_foot_region_coverage():
    # Floor strip zone (bottom of frame). A standing person's box extends well
    # ABOVE the strip, so whole-box coverage is < 0.5, but the foot region is
    # inside -> should count as in-zone with NO ankle keypoints.
    floor = Zone([(0, 120), (200, 120), (200, 200), (0, 200)])
    an = QueueAnalyzer(floor, enter_frames=2, exit_seconds=1.0,
                       kpt_thr=0.5, coverage_thr=0.5, foot_band=0.3)
    box = [40, 20, 120, 180]  # head at y=20 (above strip), feet at y=180 (inside)
    assert floor.coverage(box) < 0.5  # whole-box coverage would NOT trigger
    an.update([person(1, box)], 0.5)
    r = an.update([person(1, box)], 0.5)
    assert r.statuses[0].waiting is True


def test_min_dwell_rejects_passerby():
    # enter_frames satisfied quickly, but dwell gate (2s) not met -> bystander
    # who passes through is NOT counted as waiting.
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0, min_dwell_seconds=2.0)
    box = [40, 40, 60, 80]  # inside zone
    r1 = an.update([person(1, box)], 0.1)   # in_frames 1, dwell 0.1
    r2 = an.update([person(1, box)], 0.1)   # in_frames 2 (>=2) but dwell 0.2 < 2.0
    assert r2.statuses[0].waiting is False
    assert r2.statuses[0].candidate is True
    assert r1.statuses[0].progress < 1.0


def test_min_dwell_admits_lingerer():
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0, min_dwell_seconds=2.0)
    box = [40, 40, 60, 80]
    r = None
    for _ in range(25):                      # 25 * 0.1 = 2.5s of dwell
        r = an.update([person(1, box)], 0.1)
    assert r.statuses[0].waiting is True


def test_min_dwell_progress_tracks_slower_gate():
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0, min_dwell_seconds=1.0)
    box = [40, 40, 60, 80]
    an.update([person(1, box)], 0.1)         # frames 1/2=0.5, dwell 0.1/1.0=0.1
    r = an.update([person(1, box)], 0.1)     # frames 2/2=1.0, dwell 0.2/1.0=0.2
    # progress is the slower (dwell) gate
    assert abs(r.statuses[0].progress - 0.2) < 1e-6


def test_min_dwell_default_off_preserves_behavior():
    an = QueueAnalyzer(ZONE, enter_frames=2, exit_seconds=1.0)  # default dwell 0
    box = [40, 40, 60, 80]
    an.update([person(1, box)], 0.01)
    r = an.update([person(1, box)], 0.01)    # tiny dt, but dwell gate off
    assert r.statuses[0].waiting is True
