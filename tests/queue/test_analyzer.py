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
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_seconds=1.0, exit_seconds=1.0)
    box = [40, 40, 60, 80]  # foot (50, 80) inside, height 40
    assert an.update([person(1, box)], 0.5).statuses[0].waiting is False
    r = an.update([person(1, box)], 0.5)  # in_streak reaches 1.0
    assert r.statuses[0].waiting is True
    assert r.count == 1


def test_walkthrough_does_not_wait():
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_seconds=1.0, exit_seconds=1.0)
    r = None
    for xc in (10, 40, 70, 100):  # +30 px/frame -> fast
        r = an.update([person(1, [xc, 40, xc + 20, 80])], 0.5)
    assert r.statuses[0].waiting is False
    assert r.count == 0


def test_leaving_zone_ends_wait_and_emits_completed():
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_seconds=1.0, exit_seconds=1.0)
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
    an = QueueAnalyzer(ZONE, wait_speed=0.15, enter_seconds=1.0, exit_seconds=5.0)
    box = [40, 40, 60, 80]
    an.update([person(1, box)], 0.5)
    an.update([person(1, box)], 0.5)  # waiting
    r = an.update([], 0.5)  # track id 1 vanished
    assert r.count == 0
    assert len(r.completed) == 1 and r.completed[0].id == 1
