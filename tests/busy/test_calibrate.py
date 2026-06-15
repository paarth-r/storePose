import json

import pytest

from storepose.busy.aggregator import BusyAggregator
from storepose.busy.calibrate import (
    PEAK_FRACS,
    compute_strategies,
    load_calib,
    pick_default_strategy,
    thresholds_from_calib,
    write_calib,
)
from storepose.busy.types import BusyLevel, BusyThresholds


# --- compute_strategies --------------------------------------------------

def test_compute_strategies_known_distribution():
    # 12 sub-window values, sorted: 0 0 0 1 1 1 1 2 2 3 5 6 (peak 6)
    vals = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 5, 6]
    s = compute_strategies(vals)
    # nearest-rank: p60 -> idx ceil(.6*12)-1 = 7 -> 2 ; p85 -> idx ceil(.85*12)-1 = 10 -> 5
    assert s["skewed"] == {"low_max": 2.0, "medium_max": 5.0}
    # p33 -> idx ceil(.33*12)-1 = 3 -> 1 ; p66 -> idx ceil(.66*12)-1 = 7 -> 2
    assert s["thirds"] == {"low_max": 1.0, "medium_max": 2.0}
    # peak fractions of 6
    assert s["peak"] == {"low_max": round(PEAK_FRACS[0] * 6, 4),
                         "medium_max": round(PEAK_FRACS[1] * 6, 4)}


def test_compute_strategies_orders_bands():
    for vals in ([0, 0, 0, 0, 9], [4], [0, 5], list(range(50))):
        for band in compute_strategies(vals).values():
            assert band["medium_max"] >= band["low_max"]


def test_compute_strategies_empty_is_all_zero():
    s = compute_strategies([])
    for band in s.values():
        assert band == {"low_max": 0.0, "medium_max": 0.0}


# --- pick_default_strategy -----------------------------------------------

def test_pick_default_skewed_when_line_empties():
    # mostly empty, occasional spike -> median/peak low -> skewed
    vals = [0, 0, 0, 0, 0, 0, 1, 1, 2, 8]
    strat, ratio = pick_default_strategy(vals)
    assert strat == "skewed"
    assert ratio < 0.5


def test_pick_default_peak_when_line_sits_full():
    # rarely empties, sits near the top -> median/peak high -> peak
    vals = [4, 5, 5, 6, 6, 6, 7, 7, 7, 8]
    strat, ratio = pick_default_strategy(vals)
    assert strat == "peak"
    assert ratio >= 0.5


def test_pick_default_empty_or_flat_falls_back():
    assert pick_default_strategy([])[0] == "skewed"
    assert pick_default_strategy([0, 0, 0])[0] == "skewed"  # peak 0 -> fallback


# --- BusyAggregator.subwindow_values -------------------------------------

def _agg(sub=30.0, metric="occupancy_p90", window=600.0):
    return BusyAggregator(BusyThresholds(metric=metric),
                          window_seconds=window, sub_window_seconds=sub)


def test_subwindow_values_buckets_across_windows():
    # occupancy = sub-window index, samples spanning two 600s windows
    agg = _agg(sub=30.0, window=600.0)
    for t in range(0, 1200):  # 40 sub-windows of 30s
        agg.observe(float(t), t // 30, 1.0)
    vals = agg.subwindow_values()
    assert vals == [float(i) for i in range(40)]


def test_subwindow_values_requires_subwindow():
    agg = BusyAggregator(BusyThresholds(), window_seconds=600.0)
    agg.observe(0.0, 1, 1.0)
    with pytest.raises(ValueError):
        agg.subwindow_values()


def test_subwindow_values_rejects_mean_wait():
    agg = _agg(metric="mean_wait")
    agg.observe(0.0, 1, 1.0)
    with pytest.raises(ValueError):
        agg.subwindow_values()


# --- BusyAggregator.estimate_recent --------------------------------------

def test_estimate_recent_uses_only_trailing_samples():
    th = BusyThresholds(metric="occupancy_max", low_max=1.0, medium_max=3.0)
    agg = BusyAggregator(th, window_seconds=600.0)
    # busy early, calm recently
    for t in range(0, 50):
        agg.observe(float(t), 8, 1.0)
    for t in range(50, 100):
        agg.observe(float(t), 0, 1.0)
    level, value = agg.estimate_recent(99.0, lookback=30.0)
    assert value == 0.0
    assert level == BusyLevel.LOW


def test_estimate_recent_empty_window_is_low():
    agg = BusyAggregator(BusyThresholds(metric="occupancy_max"), window_seconds=600.0)
    for t in range(0, 50):
        agg.observe(float(t), 9, 1.0)
    level, value = agg.estimate_recent(200.0, lookback=30.0)  # nothing in [170,200]
    assert value == 0.0
    assert level == BusyLevel.LOW


def test_estimate_recent_nonpositive_lookback_falls_back():
    agg = _agg(metric="occupancy_max")
    for t in range(0, 50):
        agg.observe(float(t), 5, 1.0)
    assert agg.estimate_recent(49.0, 0.0) == agg.estimate(49.0)


# --- calib I/O + thresholds_from_calib -----------------------------------

def test_calib_round_trip_and_thresholds(tmp_path):
    payload = {
        "stem": "demo", "metric": "occupancy_p90", "subwindow_seconds": 30.0,
        "strategies": {
            "skewed": {"low_max": 5.0, "medium_max": 6.0},
            "thirds": {"low_max": 4.0, "medium_max": 5.0},
            "peak": {"low_max": 2.1, "medium_max": 4.9},
        },
    }
    p = tmp_path / "demo.json"
    write_calib(p, payload)
    loaded = load_calib(p)
    assert loaded == payload
    th = thresholds_from_calib(loaded, "peak", hysteresis=0.5)
    assert (th.metric, th.low_max, th.medium_max, th.hysteresis) == (
        "occupancy_p90", 2.1, 4.9, 0.5)


def test_thresholds_from_calib_unknown_strategy(tmp_path):
    calib = {"metric": "occupancy_p90", "strategies": {"skewed": {"low_max": 1, "medium_max": 2}}}
    with pytest.raises(ValueError):
        thresholds_from_calib(calib, "peak")
