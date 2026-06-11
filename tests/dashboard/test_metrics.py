from storepose.dashboard.metrics import (
    moving_average, occupancy_series, wait_serve_series, throughput_series,
    summary_stats, build_payload,
)
from storepose.dashboard.state import Visit


def test_moving_average_trailing_window():
    t = [0.0, 1.0, 2.0, 3.0]
    v = [0.0, 2.0, 4.0, 6.0]
    ma = moving_average(t, v, 1.5)
    assert ma[0] == 0.0
    assert ma[-1] == 5.0   # window (1.5, 3] -> [4, 6] -> 5


def test_occupancy_series_shape():
    occ = [(0.0, 2, 1), (1.0, 3, 0)]
    s = occupancy_series(occ, ma_window=10.0)
    assert s["t"] == [0.0, 1.0]
    assert s["waiting"] == [2, 3] and s["serving"] == [1, 0]
    assert len(s["waiting_ma"]) == 2 and len(s["serving_ma"]) == 2


def test_wait_serve_series_only_served():
    visits = [Visit(1.0, 4.0, 2.0, "served"), Visit(2.0, 6.0, 4.0, "abandoned")]
    s = wait_serve_series(visits, window=100.0)
    assert s["t"] == [1.0]
    assert s["wait_ma"] == [4.0] and s["serve_ma"] == [2.0]


def test_throughput_per_minute_buckets():
    visits = [Visit(t, 1.0, 1.0, "served") for t in (0.0, 10.0, 50.0, 70.0)]
    s = throughput_series(visits, bucket=60.0)
    assert s["served_per_min"] == [3.0, 1.0]


def test_build_payload_keys():
    occ = [(0.0, 1, 0)]
    visits = [Visit(0.0, 2.0, 1.0, "served")]
    p = build_payload((occ, visits))
    assert set(p) >= {"occupancy", "wait_serve", "throughput", "summary"}


def test_summary_stats():
    from storepose.dashboard.metrics import summary_stats
    occ = [(0.0, 2, 1), (1.0, 3, 2)]
    visits = [Visit(1.0, 10.0, 4.0, "served"),
              Visit(2.0, 6.0, 2.0, "served"),
              Visit(3.0, 5.0, 0.0, "abandoned")]
    s = summary_stats(occ, visits)
    assert s["in_line"] == 3 and s["at_pos"] == 2
    assert s["avg_line_s"] == 8.0    # (10 + 6) / 2
    assert s["avg_pos_s"] == 3.0     # (4 + 2) / 2
    assert s["avg_total_s"] == 11.0  # (14 + 8) / 2
    assert s["served_count"] == 2


def test_summary_stats_empty():
    s = summary_stats([], [])
    assert s == {"in_line": 0, "at_pos": 0, "avg_line_s": 0.0,
                 "avg_pos_s": 0.0, "avg_total_s": 0.0, "served_count": 0}


def test_busy_series():
    from storepose.dashboard.metrics import busy_series
    cur = (20.0, "High", 4.2)
    hist = [(0.0, "Low", 0.5), (10.0, "Medium", 2.0), (20.0, "High", 4.2)]
    b = busy_series(cur, hist)
    assert b["current"] == {"level": "High", "value": 4.2}
    assert b["t"] == [0.0, 10.0, 20.0]
    assert b["level_idx"] == [0, 1, 2]
    assert b["value"] == [0.5, 2.0, 4.2]


def test_busy_series_empty_and_payload():
    from storepose.dashboard.metrics import busy_series
    assert busy_series(None, [])["current"]["level"] is None
    p = build_payload(([(0.0, 1, 0)], []), ((0.0, "Low", 0.5), [(0.0, "Low", 0.5)]))
    assert p["busy"]["current"]["level"] == "Low"


def test_checkout_stats_and_payload():
    from storepose.dashboard.metrics import checkout_stats
    visits = [Visit(1.0, 5.0, 10.0, "served"), Visit(2.0, 4.0, 20.0, "served"),
              Visit(3.0, 6.0, 40.0, "served_other"), Visit(4.0, 2.0, 0.0, "abandoned")]
    s = checkout_stats(visits)
    assert s["mashgin_avg"] == 15.0 and s["mashgin_n"] == 2   # (10 + 20) / 2
    assert s["other_avg"] == 40.0 and s["other_n"] == 1
    assert s["delta"] == 25.0                                  # 40 - 15
    p = build_payload(([(0.0, 1, 0)], visits))
    assert p["checkouts"]["mashgin_n"] == 2 and "series" in p["checkouts"]


def test_build_payload_debug_block():
    rows = [{"id": 1, "state": "waiting", "speed": 0.1}]
    p = build_payload(([(0.0, 1, 0)], []), debug=(7, rows))
    assert p["debug"] == {"frame": 7, "rows": rows}
    # default is an empty block, never missing
    assert build_payload(([], []))["debug"] == {"frame": None, "rows": []}
