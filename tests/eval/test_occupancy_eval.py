from __future__ import annotations

import math

from storepose.eval.occupancy_eval import OccupancyReport, occupancy_eval


def test_perfect_match():
    gt = [(0.0, 1), (1.0, 2), (2.0, 3)]
    rep = occupancy_eval(gt, list(gt))
    assert rep.n == 3
    assert rep.mae == 0.0
    assert rep.bias == 0.0
    assert math.isclose(rep.corr, 1.0, rel_tol=1e-9)


def test_mae_and_bias_signs():
    gt = [(0.0, 0), (1.0, 0), (2.0, 0)]
    pred = [(0.0, 1), (1.0, 2), (2.0, 0)]  # over-counts by 1, 2, 0
    rep = occupancy_eval(gt, pred)
    assert rep.mae == 1.0          # (1+2+0)/3
    assert rep.bias == 1.0         # pred - gt, positive = over-count
    assert rep.pred_mean == 1.0


def test_only_overlapping_timestamps_scored():
    gt = [(0.0, 1), (1.0, 1)]
    pred = [(1.0, 1), (2.0, 5)]    # only t=1.0 overlaps
    rep = occupancy_eval(gt, pred)
    assert rep.n == 1
    assert rep.mae == 0.0


def test_no_overlap_is_empty_report():
    rep = occupancy_eval([(0.0, 1)], [(9.0, 1)])
    assert rep.n == 0
    assert rep.corr == 0.0


def test_constant_series_correlation_zero():
    gt = [(0.0, 2), (1.0, 2), (2.0, 2)]   # zero variance
    pred = [(0.0, 1), (1.0, 3), (2.0, 2)]
    rep = occupancy_eval(gt, pred)
    assert rep.corr == 0.0
