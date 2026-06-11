from storepose.dashboard.state import DashboardState, Visit


def test_observe_downsamples_to_interval():
    s = DashboardState(occ_interval=1.0)
    s.observe(0.0, 2, 1)
    s.observe(0.5, 3, 0)   # within interval -> dropped
    s.observe(1.0, 4, 2)   # >= interval -> kept
    occ, _ = s.snapshot()
    assert occ == [(0.0, 2, 1), (1.0, 4, 2)]


def test_add_visit_records():
    s = DashboardState()
    s.add_visit(5.0, 7.0, 3.0, "served")
    _, visits = s.snapshot()
    assert visits == [Visit(5.0, 7.0, 3.0, "served")]


def test_snapshot_is_independent_copy():
    s = DashboardState(occ_interval=0.0)
    s.observe(0.0, 1, 0)
    occ, visits = s.snapshot()
    occ.append((9.0, 9, 9))
    occ2, _ = s.snapshot()
    assert len(occ2) == 1   # mutating the snapshot didn't touch state


def test_buffers_are_bounded():
    s = DashboardState(occ_interval=0.0, max_samples=3, max_visits=2)
    for i in range(10):
        s.observe(float(i), i, 0)
        s.add_visit(float(i), 1.0, 1.0, "served")
    occ, visits = s.snapshot()
    assert len(occ) == 3 and len(visits) == 2


def test_busy_set_and_snapshot():
    s = DashboardState()
    assert s.busy_snapshot() == (None, [])
    s.set_busy(5.0, "High", 4.2)
    cur, hist = s.busy_snapshot()
    assert cur == (5.0, "High", 4.2)
    assert hist == [(5.0, "High", 4.2)]
