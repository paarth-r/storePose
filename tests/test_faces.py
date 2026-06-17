import numpy as np

from storepose.faces import FACE_KEYPOINTS, blur_faces, face_region


def _kpts_with_face(cx, cy, spread=10.0):
    """A (17,2) keypoint array with the 5 face points clustered around (cx, cy)."""
    kpts = np.zeros((17, 2), dtype=np.float32)
    offsets = [(0, 0), (-spread, -spread), (spread, -spread),
               (-2 * spread, 0), (2 * spread, 0)]
    for idx, (dx, dy) in zip(FACE_KEYPOINTS, offsets):
        kpts[idx] = (cx + dx, cy + dy)
    return kpts


def test_face_region_from_keypoints_centers_on_face():
    box = np.array([0, 0, 200, 400], dtype=np.float32)
    kpts = _kpts_with_face(100, 60, spread=10.0)
    scores = np.ones(17, dtype=np.float32)
    region = face_region(box, kpts, scores, kpt_thr=0.5, frame_w=200, frame_h=400)
    x1, y1, x2, y2 = region
    # the face keypoints span x in [80, 120], y in [50, 60]; region must cover them
    assert x1 <= 80 and x2 >= 120
    assert y1 <= 50 and y2 >= 60
    # and it must be padded out beyond the raw keypoint extent
    assert x1 < 80 and x2 > 120 and y1 < 50


def test_face_region_falls_back_to_top_quarter_without_face_keypoints():
    box = np.array([10, 20, 110, 220], dtype=np.float32)  # 100x200
    scores = np.zeros(17, dtype=np.float32)  # nothing above threshold
    kpts = np.zeros((17, 2), dtype=np.float32)
    region = face_region(box, kpts, scores, kpt_thr=0.5, frame_w=640, frame_h=480)
    assert region == (10, 20, 110, 70)  # top 1/4 of height: 20 + 200*0.25 = 70


def test_face_region_falls_back_when_keypoints_none():
    box = np.array([0, 0, 80, 160], dtype=np.float32)
    region = face_region(box, None, None, kpt_thr=0.5, frame_w=640, frame_h=480)
    assert region == (0, 0, 80, 40)


def test_face_region_clamps_to_frame_bounds():
    box = np.array([-50, 10, 50, 210], dtype=np.float32)
    region = face_region(box, None, None, kpt_thr=0.5, frame_w=640, frame_h=480)
    x1, y1, x2, y2 = region
    assert x1 == 0  # left edge clamped from -50
    assert region == (0, 10, 50, 60)  # top quarter, clamped


def test_face_region_none_for_degenerate_box():
    box = np.array([100, 100, 100, 100], dtype=np.float32)  # zero area
    assert face_region(box, None, None, kpt_thr=0.5, frame_w=640, frame_h=480) is None


def test_face_region_single_visible_keypoint_uses_fallback():
    box = np.array([0, 0, 100, 200], dtype=np.float32)
    kpts = _kpts_with_face(50, 30)
    scores = np.zeros(17, dtype=np.float32)
    scores[0] = 1.0  # only the nose is visible -> not enough to size a region
    region = face_region(box, kpts, scores, kpt_thr=0.5, frame_w=640, frame_h=480)
    assert region == (0, 0, 100, 50)  # top quarter fallback


def test_blur_faces_modifies_face_region_only():
    canvas = np.full((400, 200, 3), 200, dtype=np.uint8)
    # paint a recognizable gradient in the top quarter so blur changes it
    canvas[0:100, :, 0] = np.tile(np.arange(200, dtype=np.uint8), (100, 1))
    box = np.array([0, 0, 200, 400], dtype=np.float32)
    before = canvas.copy()
    blur_faces(canvas, [(box, None, None)], kpt_thr=0.5)
    # top quarter changed
    assert not np.array_equal(canvas[0:100], before[0:100])
    # bottom untouched
    assert np.array_equal(canvas[100:], before[100:])


def test_blur_faces_returns_canvas_and_handles_empty():
    canvas = np.zeros((40, 40, 3), dtype=np.uint8)
    out = blur_faces(canvas, [], kpt_thr=0.5)
    assert out is canvas
