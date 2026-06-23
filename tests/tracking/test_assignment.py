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


def test_appearance_breaks_iou_tie():
    # Two tracks both overlap the detection by IoU; appearance picks the one the
    # detection actually looks like (track 1 here).
    det = np.array([10, 10, 50, 90], float)
    trks = [np.array([12, 12, 52, 92], float), np.array([8, 8, 48, 88], float)]
    appsim = np.array([[0.2, 0.95]], float)  # det looks like track 1, not track 0
    matches, ud, ut = match([det], trks, iou_thr=0.3, appsim=appsim,
                            app_weight=0.5, app_floor=0.0)
    assert matches == [(0, 1)]


def test_appearance_veto_rejects_dissimilar_overlap():
    # The detection overlaps the track by IoU but looks nothing like it (a person
    # passing in front of a prop) -> not matched, left for a fresh id.
    det = np.array([10, 10, 50, 90], float)
    trk = np.array([10, 10, 50, 90], float)  # perfect IoU
    appsim = np.array([[0.1]], float)
    matches, ud, ut = match([det], [trk], iou_thr=0.3, appsim=appsim,
                            app_weight=0.5, app_floor=0.4)
    assert matches == [] and ud == [0] and ut == [0]


def test_appearance_none_is_iou_only():
    det = np.array([10, 10, 50, 90], float)
    trk = np.array([10, 10, 50, 90], float)
    assert match([det], [trk], iou_thr=0.3) == match([det], [trk], iou_thr=0.3, appsim=None)
    assert match([det], [trk], iou_thr=0.3)[0] == [(0, 0)]


def test_appearance_nan_pair_falls_back_to_iou():
    # An unknown appearance (NaN, e.g. a track with no embedding yet) must not
    # veto a good IoU match.
    det = np.array([10, 10, 50, 90], float)
    trk = np.array([10, 10, 50, 90], float)
    appsim = np.array([[np.nan]], float)
    matches, _, _ = match([det], [trk], iou_thr=0.3, appsim=appsim,
                          app_weight=0.5, app_floor=0.4)
    assert matches == [(0, 0)]


def test_motion_direction_breaks_tie_iou_cannot():
    # Two tracks with identical boxes (equal IoU to the detection); motion picks
    # the one whose recent velocity points toward the detection. This is the
    # crossing case: who-went-where decides when geometry/appearance tie.
    det = np.array([10, 10, 50, 90], float)
    trk = np.array([10, 10, 50, 90], float)
    motsim = np.array([[-0.9, 0.9]], float)  # det is along track 1's heading, against track 0's
    matches, _, _ = match([det], [trk, trk], iou_thr=0.3, motsim=motsim, mot_weight=0.5)
    assert matches == [(0, 1)]


def test_motion_nan_pair_is_neutral():
    det = np.array([10, 10, 50, 90], float)
    trk = np.array([10, 10, 50, 90], float)
    motsim = np.array([[np.nan]], float)  # stationary track: no heading -> neutral
    matches, _, _ = match([det], [trk], iou_thr=0.3, motsim=motsim, mot_weight=0.5)
    assert matches == [(0, 0)]
