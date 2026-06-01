import numpy as np

from storepose.tracking.assignment import iou, iou_matrix, match


def test_iou_identical_is_one():
    b = np.array([0, 0, 10, 10], float)
    assert iou(b, b) == 1.0


def test_iou_disjoint_is_zero():
    assert iou(np.array([0, 0, 10, 10], float),
               np.array([20, 20, 30, 30], float)) == 0.0


def test_iou_half_overlap():
    a = np.array([0, 0, 10, 10], float)
    b = np.array([5, 0, 15, 10], float)  # overlap 50x100? inter=50, union=150
    assert iou(a, b) == 0.5 / 1.5  # inter 50, union 150 -> 1/3


def test_iou_matrix_shape_and_values():
    dets = [np.array([0, 0, 10, 10], float), np.array([100, 100, 110, 110], float)]
    trks = [np.array([0, 0, 10, 10], float)]
    m = iou_matrix(dets, trks)
    assert m.shape == (2, 1)
    assert m[0, 0] == 1.0
    assert m[1, 0] == 0.0


def test_match_pairs_overlapping():
    dets = [np.array([0, 0, 10, 10], float), np.array([100, 0, 110, 10], float)]
    trks = [np.array([101, 0, 111, 10], float), np.array([1, 1, 11, 11], float)]
    matches, ud, ut = match(dets, trks, iou_thr=0.3)
    # det0 -> trk1, det1 -> trk0
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert ud == [] and ut == []


def test_match_gates_low_iou():
    dets = [np.array([0, 0, 10, 10], float)]
    trks = [np.array([50, 50, 60, 60], float)]
    matches, ud, ut = match(dets, trks, iou_thr=0.3)
    assert matches == []
    assert ud == [0] and ut == [0]


def test_match_empty_inputs():
    assert match([], [], 0.3) == ([], [], [])
    assert match([np.array([0, 0, 1, 1], float)], [], 0.3) == ([], [0], [])
