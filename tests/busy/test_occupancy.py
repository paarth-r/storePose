from storepose.busy.occupancy import occupancy_at, sample_occupancy
from storepose.queue.types import CompletedWait


def w(entered, exited):
    return CompletedWait(id=0, entered_s=entered, exited_s=exited,
                         wait_seconds=exited - entered)


def test_occupancy_counts_overlapping_intervals():
    waits = [w(0, 10), w(5, 15), w(12, 20)]
    assert occupancy_at(waits, 1) == 1
    assert occupancy_at(waits, 6) == 2     # first two overlap
    assert occupancy_at(waits, 13) == 2    # second and third overlap
    assert occupancy_at(waits, 18) == 1
    assert occupancy_at(waits, 25) == 0


def test_interval_half_open():
    waits = [w(0, 10)]
    assert occupancy_at(waits, 0) == 1     # entered inclusive
    assert occupancy_at(waits, 10) == 0    # exited exclusive


def test_sample_occupancy_empty():
    assert sample_occupancy([]) == []


def test_sample_occupancy_steps_and_extent():
    waits = [w(0, 4)]
    samples = sample_occupancy(waits, step=1.0)
    assert samples == [(0.0, 1), (1.0, 1), (2.0, 1), (3.0, 1)]
