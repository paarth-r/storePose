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
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_frames=2, exit_seconds=1.0)
    box = [40, 40, 60, 80]  # foot (50, 80) inside, height 40
    assert an.update([person(1, box)], 0.5).statuses[0].waiting is False
    r = an.update([person(1, box)], 0.5)  # in_streak reaches 1.0
    assert r.statuses[0].waiting is True
    assert r.count == 1


def test_walkthrough_does_not_wait():
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_frames=2, exit_seconds=1.0)
    r = None
    for xc in (10, 40, 70, 100):  # +30 px/frame -> fast
        r = an.update([person(1, [xc, 40, xc + 20, 80])], 0.5)
    assert r.statuses[0].waiting is False
    assert r.count == 0


def test_leaving_zone_ends_wait_and_emits_completed():
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_frames=2, exit_seconds=1.0)
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
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_frames=2, exit_seconds=5.0)
    box = [40, 40, 60, 80]
    an.update([person(1, box)], 0.5)
    an.update([person(1, box)], 0.5)  # waiting
    r = an.update([], 0.5)  # track id 1 vanished
    assert r.count == 0
    assert len(r.completed) == 1 and r.completed[0].id == 1


def test_candidate_progress_then_inclusion():
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_frames=5, exit_seconds=1.0)
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
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_frames=2, exit_seconds=1.0, kpt_thr=0.5)
    k, s = _kpts_with_ankles(50, 50, 60, 50)  # well inside ZONE (0..200)
    box = [300, 300, 360, 400]  # outside zone
    an.update([person_pose(1, box, k, s)], 0.5)
    r = an.update([person_pose(1, box, k, s)], 0.5)
    assert r.statuses[0].waiting is True


def test_occluded_ankles_fall_back_to_coverage():
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_frames=2, exit_seconds=1.0,
                       kpt_thr=0.5, coverage_thr=0.5)
    k, s = _kpts_with_ankles(50, 50, 60, 50, score=0.1)  # ankles low confidence
    box = [40, 40, 120, 160]  # mostly inside ZONE -> coverage high
    an.update([person_pose(1, box, k, s)], 0.5)
    r = an.update([person_pose(1, box, k, s)], 0.5)
    assert r.statuses[0].waiting is True


def test_occluded_ankles_box_outside_not_waiting():
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_frames=2, exit_seconds=1.0,
                       kpt_thr=0.5, coverage_thr=0.5)
    k, s = _kpts_with_ankles(50, 50, 60, 50, score=0.1)
    box = [400, 400, 480, 560]  # outside zone
    an.update([person_pose(1, box, k, s)], 0.5)
    r = an.update([person_pose(1, box, k, s)], 0.5)
    assert r.statuses[0].waiting is False


def test_ankle_outside_but_box_covered_stays_in_zone():
    # Ankle keypoints are confident but OUTSIDE the zone, yet the box is mostly
    # inside -> OR keeps the person in-zone (timer must not reset).
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_frames=2, exit_seconds=1.0,
                       kpt_thr=0.5, coverage_thr=0.5)
    k, s = _kpts_with_ankles(500, 500, 510, 500, score=0.9)  # outside ZONE (0..200)
    box = [40, 40, 120, 160]  # mostly inside ZONE -> coverage high
    an.update([person_pose(1, box, k, s)], 0.5)
    r = an.update([person_pose(1, box, k, s)], 0.5)
    assert r.statuses[0].waiting is True


def test_wait_not_reset_when_ankle_leaves_but_box_covered():
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_frames=2, exit_seconds=1.0,
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
