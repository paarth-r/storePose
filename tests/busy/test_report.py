from storepose.busy.aggregator import BusyAggregator
from storepose.busy.report import read_busy_levels, read_waits, write_busy
from storepose.busy.types import BusyLevel


def test_round_trip_waits(tmp_path):
    p = tmp_path / "waits.csv"
    p.write_text(
        "id,entered_s,exited_s,wait_seconds\n"
        "5,1.83,2.84,0.90\n"
        "4,1.94,4.81,2.75\n"
    )
    waits = read_waits(p)
    assert len(waits) == 2
    assert waits[0].id == 5
    assert waits[1].wait_seconds == 2.75


def test_write_then_read_busy_levels(tmp_path):
    agg = BusyAggregator(window_seconds=10.0)
    for t in range(10):
        agg.observe(float(t), occupancy=6, dt=1.0)   # window 0 -> HIGH
    for t in range(10, 20):
        agg.observe(float(t), occupancy=0, dt=1.0)   # window 1 -> LOW
    wins = agg.windows()
    out = tmp_path / "busy.csv"
    write_busy(out, wins)
    levels = read_busy_levels(out)
    assert levels[0] == BusyLevel.HIGH
    assert levels[1] == BusyLevel.LOW


def test_read_waits_parses_serving_and_outcome(tmp_path):
    from storepose.busy.report import read_waits
    p = tmp_path / "w.csv"
    p.write_text(
        "id,entered_s,exited_s,wait_seconds,serving_seconds,outcome\n"
        "1,0.00,10.00,7.00,3.00,served\n"
    )
    waits = read_waits(p)
    assert len(waits) == 1
    assert waits[0].wait_seconds == 7.0
    assert waits[0].serving_seconds == 3.0
    assert waits[0].outcome == "served"


def test_read_waits_back_compat_without_new_columns(tmp_path):
    from storepose.busy.report import read_waits
    p = tmp_path / "old.csv"
    p.write_text("id,entered_s,exited_s,wait_seconds\n1,0.00,10.00,8.00\n")
    waits = read_waits(p)
    assert waits[0].serving_seconds == 0.0 and waits[0].outcome == "served"
