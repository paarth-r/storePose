from storepose.queue.outliers import OutlierFilter
from storepose.queue.types import CompletedWait


def _wait(dur, outcome="served", pid=1):
    return CompletedWait(id=pid, entered_s=0.0, exited_s=dur, wait_seconds=dur,
                         outcome=outcome)


def test_floor_only_before_warmup():
    f = OutlierFilter(floor=2.0, frac=0.25, warmup=10)
    # no samples yet -> only the floor applies, regardless of how short
    assert f.judge(_wait(1.0)).rejected is True       # below floor
    assert f.judge(_wait(3.0)).rejected is False      # above floor, accepted


def test_relative_rule_after_warmup():
    f = OutlierFilter(floor=2.0, frac=0.25, warmup=10)
    for _ in range(10):
        assert f.judge(_wait(20.0)).rejected is False  # warm up the median at ~20s
    # threshold now max(2, 0.25*20) = 5
    assert f.judge(_wait(2.0)).rejected is True        # 2 < 5 -> rejected
    assert f.judge(_wait(6.0)).rejected is False       # 6 > 5 -> accepted


def test_floor_dominates_when_median_low():
    f = OutlierFilter(floor=2.0, frac=0.25, warmup=3)
    for _ in range(3):
        f.judge(_wait(4.0))           # median 4 -> relative term 1.0, floor 2 wins
    assert f.judge(_wait(1.5)).rejected is True   # below floor
    assert f.judge(_wait(2.5)).rejected is False  # above floor


def test_per_outcome_independence():
    f = OutlierFilter(floor=0.0, frac=0.5, warmup=3)
    for _ in range(3):
        f.judge(_wait(20.0, outcome="served"))     # served median ~20 -> thr 10
    # abandoned has no samples -> floor 0 -> a short abandoned is accepted
    assert f.judge(_wait(1.0, outcome="abandoned")).rejected is False
    # but a short served is rejected against its own median
    assert f.judge(_wait(1.0, outcome="served")).rejected is True


def test_rejected_does_not_update_median():
    f = OutlierFilter(floor=2.0, frac=0.25, warmup=3)
    for _ in range(3):
        f.judge(_wait(20.0))                       # median 20 -> thr 5
    for _ in range(5):
        assert f.judge(_wait(1.0)).rejected is True  # phantoms, all rejected
    # median must still be 20 (thr 5), so a 6s visit is still accepted
    assert f.threshold("served") == 5.0
    assert f.judge(_wait(6.0)).rejected is False


def test_judge_does_not_mutate_input():
    f = OutlierFilter(floor=5.0, frac=0.25, warmup=0)
    w = _wait(1.0)
    out = f.judge(w)
    assert out is not w and out.rejected is True
    assert w.rejected is False  # original untouched
