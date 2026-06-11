from storepose.busy.aggregator import (
    BusyAggregator,
    classify,
    weighted_percentile,
)
from storepose.busy.types import BusyLevel, BusyThresholds
from storepose.queue.types import CompletedWait


# --- weighted_percentile -------------------------------------------------

def test_weighted_percentile_empty_is_zero():
    assert weighted_percentile([], [], 0.9) == 0.0


def test_weighted_percentile_uniform_weights_median():
    assert weighted_percentile([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], 0.5) == 2


def test_weighted_percentile_downweights_brief_spike():
    # occupancy is 1 for 100s, then spikes to 9 for 1s. p90 should ignore spike.
    vals = [1.0, 9.0]
    wts = [100.0, 1.0]
    assert weighted_percentile(vals, wts, 0.9) == 1.0
    assert weighted_percentile(vals, wts, 0.999) == 9.0


# --- classify / hysteresis ----------------------------------------------

TH = BusyThresholds(metric="occupancy_p90", low_max=1.0, medium_max=3.0)


def test_classify_bands_inclusive():
    assert classify(0.0, TH) == BusyLevel.LOW
    assert classify(1.0, TH) == BusyLevel.LOW       # inclusive upper bound
    assert classify(1.5, TH) == BusyLevel.MEDIUM
    assert classify(3.0, TH) == BusyLevel.MEDIUM
    assert classify(3.1, TH) == BusyLevel.HIGH


def test_hysteresis_holds_level_near_boundary():
    th = BusyThresholds(low_max=1.0, medium_max=3.0, hysteresis=0.5)
    # was MEDIUM; value 0.8 is nominally LOW but within the deadband -> stays
    assert classify(0.8, th, prev=BusyLevel.MEDIUM) == BusyLevel.MEDIUM
    # drops clearly below boundary - hysteresis -> changes
    assert classify(0.4, th, prev=BusyLevel.MEDIUM) == BusyLevel.LOW


def test_hysteresis_requires_clearing_boundary_to_rise():
    th = BusyThresholds(low_max=1.0, medium_max=3.0, hysteresis=0.5)
    assert classify(1.3, th, prev=BusyLevel.LOW) == BusyLevel.LOW    # within band
    assert classify(1.6, th, prev=BusyLevel.LOW) == BusyLevel.MEDIUM  # cleared


# --- BusyAggregator ------------------------------------------------------

def test_windows_bucketed_by_window_seconds():
    agg = BusyAggregator(window_seconds=10.0)
    agg.observe(t=1.0, occupancy=0, dt=1.0)
    agg.observe(t=5.0, occupancy=0, dt=1.0)
    agg.observe(t=12.0, occupancy=5, dt=1.0)  # second window
    wins = agg.windows()
    assert [w.index for w in wins] == [0, 1]
    assert wins[0].start_s == 0.0 and wins[0].end_s == 10.0
    assert wins[1].start_s == 10.0


def test_label_from_robust_occupancy():
    th = BusyThresholds(metric="occupancy_p90", low_max=1.0, medium_max=3.0)
    agg = BusyAggregator(th, window_seconds=100.0)
    # mostly empty with one brief crowd spike -> robust p90 stays LOW
    for t in range(0, 95):
        agg.observe(float(t), occupancy=0, dt=1.0)
    for t in range(95, 100):
        agg.observe(float(t), occupancy=8, dt=1.0)
    win = agg.windows()[0]
    assert win.level == BusyLevel.LOW
    assert win.features.max_occupancy == 8.0  # spike still recorded as a feature


def test_high_when_sustained_crowd():
    th = BusyThresholds(metric="occupancy_p90", low_max=1.0, medium_max=3.0)
    agg = BusyAggregator(th, window_seconds=100.0)
    for t in range(0, 100):
        agg.observe(float(t), occupancy=6, dt=1.0)
    assert agg.windows()[0].level == BusyLevel.HIGH


def test_completed_waits_attributed_to_windows():
    agg = BusyAggregator(window_seconds=10.0)
    agg.observe(1.0, 1, 1.0)
    agg.observe(11.0, 1, 1.0)
    # entered in window 0, exited in window 1
    agg.add_wait(CompletedWait(id=1, entered_s=2.0, exited_s=13.0, wait_seconds=11.0))
    wins = {w.index: w for w in agg.windows()}
    assert wins[0].features.arrivals == 1
    assert wins[0].features.throughput == 0
    assert wins[1].features.throughput == 1
    assert wins[1].features.mean_wait_seconds == 11.0


def test_zero_dt_ignored():
    agg = BusyAggregator(window_seconds=10.0)
    agg.observe(1.0, 5, dt=0.0)
    agg.observe(2.0, 1, dt=1.0)
    assert agg.windows()[0].features.mean_occupancy == 1.0


def test_sub_window_smoothing_damps_single_busy_burst():
    # 10-min window: 9 quiet minutes (occ 0) + 1 busy minute (occ 9).
    # Without smoothing, p90 over the whole window could be pulled up; with
    # per-minute sub-windows the median minute is quiet -> LOW.
    th = BusyThresholds(metric="occupancy_p90", low_max=1.0, medium_max=3.0)
    agg = BusyAggregator(th, window_seconds=600.0, sub_window_seconds=60.0)
    for t in range(0, 540):          # 9 quiet minutes
        agg.observe(float(t), occupancy=0, dt=1.0)
    for t in range(540, 600):        # 1 busy minute
        agg.observe(float(t), occupancy=9, dt=1.0)
    win = agg.windows()[0]
    assert win.level == BusyLevel.LOW
    assert win.features.max_occupancy == 9.0  # feature still records the burst


def test_sub_window_high_when_most_minutes_busy():
    th = BusyThresholds(metric="occupancy_p90", low_max=1.0, medium_max=3.0)
    agg = BusyAggregator(th, window_seconds=600.0, sub_window_seconds=60.0)
    for t in range(0, 600):
        agg.observe(float(t), occupancy=6, dt=1.0)  # every minute busy
    assert agg.windows()[0].level == BusyLevel.HIGH


def test_sub_window_rejects_nonpositive():
    import pytest
    with pytest.raises(ValueError):
        BusyAggregator(window_seconds=600.0, sub_window_seconds=0.0)
