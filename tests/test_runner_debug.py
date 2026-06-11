"""Pure-helper tests for the frame-by-frame debug view (runner.py)."""

from storepose.queue.types import PersonStatus
from storepose.runner import _debug_rows, _person_state, _scrub


def _status(**kw):
    base = dict(id=1, waiting=False, candidate=False, progress=0.0, wait_seconds=0.0,
                serving=False, serving_seconds=0.0, serving_other=False, debug=None)
    base.update(kw)
    return PersonStatus(**base)


def test_person_state_priority():
    assert _person_state(_status(serving=True)) == "serving-Mashgin"
    assert _person_state(_status(serving_other=True)) == "serving-REG"
    assert _person_state(_status(waiting=True)) == "waiting"
    assert _person_state(_status(candidate=True, progress=0.4)) == "candidate 40%"
    assert _person_state(_status()) == "out"


def test_debug_rows_flatten_debug_dict():
    dbg = {"speed": 0.27, "transit": True, "line": False, "pos": True, "reg": False}
    s = _status(id=9, waiting=True, wait_seconds=3.14159, serving_seconds=1.2, debug=dbg)
    (row,) = _debug_rows([s])
    assert row == {
        "id": 9, "state": "waiting", "wait": 3.14, "serve": 1.2,
        "speed": 0.27, "line": False, "pos": True, "reg": False, "transit": True,
    }


def test_debug_rows_tolerates_missing_debug():
    (row,) = _debug_rows([_status(id=2)])
    assert row["speed"] == 0.0 and row["line"] is False and row["transit"] is False


def test_scrub_clamps_to_bounds():
    assert _scrub(0, +1, 5) == 1          # review older
    assert _scrub(0, -1, 5) == 0          # already newest, stays
    assert _scrub(4, +1, 5) == 4          # oldest in a 5-frame buffer, stays
    assert _scrub(2, -1, 5) == 1          # step toward newest
    assert _scrub(3, 0, 2) == 1           # buffer shrank: clamp into range
    assert _scrub(0, +1, 0) == 0          # empty buffer
