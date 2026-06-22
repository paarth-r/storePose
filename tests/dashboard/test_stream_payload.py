import numpy as np

from storepose.dashboard.stream_payload import build_overlay
from storepose.queue.types import PersonStatus
from storepose.queue.zone import Zone
from storepose.tracking.types import TrackedPerson


def _person(pid, box, kpts=None, scores=None):
    return TrackedPerson(
        id=pid,
        box=np.array(box, dtype=float),
        keypoints=None if kpts is None else np.array(kpts, dtype=float),
        scores=None if scores is None else np.array(scores, dtype=float),
        coasting=False,
        color=(0, 255, 0),
    )


def test_dimensions_passed_through():
    o = build_overlay([], [], {}, None, 1280, 720)
    assert o["w"] == 1280 and o["h"] == 720
    assert o["people"] == []
    assert o["zones"] == {}
    assert o["busy"] is None


def test_person_box_and_keypoints():
    kpts = [[float(i), float(i + 1)] for i in range(17)]
    scores = [0.9] * 17
    o = build_overlay([_person(5, [10, 20, 110, 220], kpts, scores)], [], {}, None, 640, 480)
    p = o["people"][0]
    assert p["id"] == 5
    assert p["box"] == [10.0, 20.0, 110.0, 220.0]
    assert len(p["kpts"]) == 17
    assert p["kpts"][0] == [0.0, 1.0, 0.9]


def test_coasting_person_has_no_keypoints():
    o = build_overlay([_person(1, [0, 0, 10, 10])], [], {}, None, 640, 480)
    assert o["people"][0]["kpts"] == []


def test_state_derived_from_status():
    people = [_person(1, [0, 0, 1, 1]), _person(2, [0, 0, 1, 1]),
              _person(3, [0, 0, 1, 1]), _person(4, [0, 0, 1, 1])]
    statuses = [
        PersonStatus(id=1, waiting=True, candidate=False, progress=1.0, wait_seconds=12.0),
        PersonStatus(id=2, waiting=False, candidate=False, progress=1.0, wait_seconds=3.0,
                     serving=True, serving_seconds=8.0),
        PersonStatus(id=3, waiting=False, candidate=False, progress=1.0, wait_seconds=1.0,
                     serving_other=True, serving_seconds=4.0),
        PersonStatus(id=4, waiting=False, candidate=True, progress=0.4, wait_seconds=0.0),
    ]
    o = build_overlay(people, statuses, {}, None, 640, 480)
    by_id = {p["id"]: p for p in o["people"]}
    assert by_id[1]["state"] == "waiting" and by_id[1]["wait"] == 12.0
    assert by_id[2]["state"] == "serving" and by_id[2]["serve"] == 8.0
    assert by_id[3]["state"] == "serving_other"
    assert by_id[4]["state"] == "candidate" and by_id[4]["progress"] == 0.4


def test_person_without_status_is_tracked():
    o = build_overlay([_person(9, [0, 0, 1, 1])], [], {}, None, 640, 480)
    assert o["people"][0]["state"] == "tracked"


def test_zones_serialized_as_polygons():
    line = Zone([(0, 0), (10, 0), (10, 10)])
    pos = Zone([(20, 20), (30, 20), (30, 30)])
    o = build_overlay([], [], {"line": line, "pos": pos, "alt": None}, None, 640, 480)
    assert o["zones"]["line"] == [[[0, 0], [10, 0], [10, 10]]]
    assert o["zones"]["pos"] == [[[20, 20], [30, 20], [30, 30]]]
    assert "alt" not in o["zones"]


def test_busy_serialized():
    o = build_overlay([], [], {}, ("High", 2.5), 640, 480)
    assert o["busy"] == {"level": "High", "value": 2.5}
