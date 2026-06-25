"""Floor geometry: homography matches ray-plane intersection; views compose."""

import numpy as np
import pytest

from storepose.scene.geometry import (
    CameraView,
    StoreFloor,
    floor_homography,
    intrinsics_matrix,
    plane_basis,
)


def _ref_floor_xy(u, v, normal, d, K):
    """Independent reference: explicit ray-plane intersection -> floor 2-D."""
    n = normal / np.linalg.norm(normal)
    e1, e2 = plane_basis(n)
    r = np.linalg.inv(K) @ np.array([u, v, 1.0])
    t = -d / (n.dot(r))
    X = t * r
    return np.array([e1.dot(X), e2.dot(X)])


def test_homography_matches_ray_plane_intersection():
    K = intrinsics_matrix(943.0, 960.0, 540.0)
    normal = np.array([0.0, -0.703, -0.711])
    d = 2.69
    H = floor_homography(normal, d, K)
    for u, v in [(960, 800), (400, 700), (1500, 950), (700, 600)]:
        q = H @ np.array([u, v, 1.0])
        got = np.array([q[0] / q[2], q[1] / q[2]])
        ref = _ref_floor_xy(u, v, normal, d, K)
        assert np.allclose(got, ref, atol=1e-6)


def test_basis_orthonormal_and_perpendicular():
    n = np.array([0.0, -0.7, -0.71])
    n = n / np.linalg.norm(n)
    e1, e2 = plane_basis(n)
    assert np.isclose(np.linalg.norm(e1), 1.0)
    assert np.isclose(np.linalg.norm(e2), 1.0)
    assert np.isclose(e1.dot(n), 0.0, atol=1e-9)
    assert np.isclose(e2.dot(n), 0.0, atol=1e-9)
    assert np.isclose(e1.dot(e2), 0.0, atol=1e-9)


def test_camera_view_identity_transform_is_floor_frame():
    K = intrinsics_matrix(900.0, 960.0, 540.0)
    normal = np.array([0.0, -0.7, -0.71])
    view = CameraView("cam0", floor_homography(normal, 2.5, K), normal)
    x, y = view.apply(960, 800)
    q = view.H @ np.array([960, 800, 1.0])
    assert np.allclose([x, y], [q[0] / q[2], q[1] / q[2]])


def test_camera_view_applies_store_transform():
    K = intrinsics_matrix(900.0, 960.0, 540.0)
    normal = np.array([0.0, -0.7, -0.71])
    H = floor_homography(normal, 2.5, K)
    base = CameraView("cam0", H, normal)
    shifted = CameraView(
        "cam0", H, normal,
        T_cam_to_store=np.array([[1, 0, 5.0], [0, 1, -3.0], [0, 0, 1]]),
    )
    bx, by = base.apply(960, 800)
    sx, sy = shifted.apply(960, 800)
    assert np.allclose([sx, sy], [bx + 5.0, by - 3.0])


def test_store_floor_get_and_missing():
    K = intrinsics_matrix(900.0, 960.0, 540.0)
    n = np.array([0.0, -0.7, -0.71])
    sf = StoreFloor()
    sf.add(CameraView("camA", floor_homography(n, 2.5, K), n))
    assert sf.get("camA").camera_id == "camA"
    with pytest.raises(KeyError):
        sf.get("camB")
