from storepose.dashboard.metrics import (
    moving_average, occupancy_series, wait_serve_series, throughput_series, build_payload,
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
    assert set(p) >= {"occupancy", "wait_serve", "throughput"}
