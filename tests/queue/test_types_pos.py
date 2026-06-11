from storepose.queue.types import PersonStatus, CompletedWait, QueueResult


def test_person_status_has_serving_fields_with_defaults():
    s = PersonStatus(id=1, waiting=True, candidate=False, progress=1.0, wait_seconds=2.0)
    assert s.serving is False
    assert s.serving_seconds == 0.0
    s2 = PersonStatus(id=2, waiting=False, candidate=False, progress=1.0,
                      wait_seconds=0.0, serving=True, serving_seconds=3.5)
    assert s2.serving is True and s2.serving_seconds == 3.5


def test_completed_wait_has_serving_and_outcome_defaults():
    c = CompletedWait(id=1, entered_s=0.0, exited_s=10.0, wait_seconds=8.0)
    assert c.serving_seconds == 0.0
    assert c.outcome == "served"
    c2 = CompletedWait(1, 0.0, 12.0, 8.0, 4.0, "served")
    assert c2.serving_seconds == 4.0 and c2.outcome == "served"


def test_queue_result_has_serving_count_default():
    r = QueueResult(statuses=[], count=0)
    assert r.serving_count == 0


def test_serving_other_defaults():
    s = PersonStatus(id=1, waiting=False, candidate=False, progress=1.0, wait_seconds=0.0)
    assert s.serving_other is False
    r = QueueResult(statuses=[], count=0)
    assert r.serving_other_count == 0
