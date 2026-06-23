import numpy as np

from storepose.detector import (
    _contained_keep_indices,
    filter_confident,
    suppress_contained_boxes,
)


def test_filter_confident_drops_boxes_below_threshold():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 30, 30]], float)
    scores = np.array([0.92, 0.55, 0.81], np.float32)
    b, s = filter_confident(boxes, scores, 0.8)
    assert len(b) == 2
    assert np.allclose(s, [0.92, 0.81])
    assert np.array_equal(b[0], [0, 0, 10, 10]) and np.array_equal(b[1], [0, 0, 30, 30])


def test_filter_confident_empty_passes_through():
    b, s = filter_confident(np.empty((0, 4), float), np.empty((0,), np.float32), 0.8)
    assert len(b) == 0 and len(s) == 0


def test_drops_box_mostly_contained_in_a_larger_one():
    boxes = np.array([
        [0, 0, 100, 200],     # full person
        [10, 20, 90, 180],    # duplicate sub-box, ~80%+ inside the first
    ], float)
    out = suppress_contained_boxes(boxes, 0.8)
    assert len(out) == 1
    assert np.array_equal(out[0], [0, 0, 100, 200])  # the larger box is kept


def test_keeps_two_side_by_side_people():
    # Overlap but neither box is mostly contained in the other -> both kept.
    boxes = np.array([
        [0, 0, 60, 100],
        [45, 0, 105, 100],
    ], float)
    out = suppress_contained_boxes(boxes, 0.8)
    assert len(out) == 2


def test_keeps_disjoint_boxes():
    boxes = np.array([[0, 0, 50, 50], [200, 200, 260, 300]], float)
    out = suppress_contained_boxes(boxes, 0.8)
    assert len(out) == 2


def test_removes_exact_duplicate():
    boxes = np.array([[0, 0, 100, 100], [0, 0, 100, 100]], float)
    out = suppress_contained_boxes(boxes, 0.8)
    assert len(out) == 1


def test_empty_and_single_pass_through():
    assert len(suppress_contained_boxes(np.empty((0, 4), float), 0.8)) == 0
    one = np.array([[1, 2, 3, 4]], float)
    assert np.array_equal(suppress_contained_boxes(one, 0.8), one)


def test_keep_indices_align_boxes_and_scores():
    # the contained sub-box (index 1) is dropped; scores must filter in tandem
    boxes = np.array([
        [0, 0, 100, 200],     # full person -> kept
        [10, 20, 90, 180],    # duplicate sub-box -> dropped
        [300, 0, 360, 100],   # disjoint person -> kept
    ], float)
    scores = np.array([0.95, 0.40, 0.80], np.float32)
    idx = _contained_keep_indices(boxes, 0.8)
    assert idx == [0, 2]
    assert np.allclose(scores[idx], [0.95, 0.80])  # dropped box's score gone


def test_preserves_input_order_of_kept_boxes():
    boxes = np.array([
        [0, 0, 40, 40],        # smaller, comes first
        [100, 100, 300, 300],  # larger, comes second
    ], float)
    out = suppress_contained_boxes(boxes, 0.8)
    assert len(out) == 2
    assert np.array_equal(out[0], [0, 0, 40, 40])  # original order kept
