"""Pure plane-fit helpers (no model): back-projection and RANSAC recovery."""

import numpy as np

from storepose.scene.depthpro import (
    backproject,
    fit_floor_plane,
    floor_region_mask,
)
from storepose.scene.geometry import plane_basis


def test_backproject_recovers_metric_xyz():
    focal, cx, cy = 1000.0, 320.0, 240.0
    depth = np.full((480, 640), 3.0, dtype=np.float32)
    pts = backproject(depth, focal, cx, cy)
    # principal-point pixel projects straight ahead at depth z
    assert np.allclose(pts[240, 320], [0.0, 0.0, 3.0], atol=1e-4)
    # 160 px right of centre -> x = (160/focal)*z = 0.48 m
    assert np.allclose(pts[240, 480], [0.48, 0.0, 3.0], atol=1e-3)


def test_fit_floor_plane_recovers_known_plane():
    rng = np.random.default_rng(0)
    true_n = np.array([0.05, -0.70, -0.712])
    true_n /= np.linalg.norm(true_n)
    true_d = 2.69
    e1, e2 = plane_basis(true_n)
    origin = -true_d * true_n
    # 40k floor points + 5% off-plane outliers
    a = rng.uniform(-3, 3, 40000)
    b = rng.uniform(0, 6, 40000)
    noise = rng.normal(0, 0.005, 40000)
    X = origin + a[:, None] * e1 + b[:, None] * e2 + noise[:, None] * true_n
    n_out = 2000
    X[:n_out] += rng.normal(0, 1.0, (n_out, 3))
    pts = X.reshape(-1, 1, 3).astype(np.float32)
    mask = np.ones((X.shape[0], 1), dtype=bool)
    n_fit, d_fit = fit_floor_plane(pts, mask, seed=1)
    # normal aligns (up to sign already handled), d close
    assert abs(abs(n_fit.dot(true_n)) - 1.0) < 1e-3
    assert abs(abs(d_fit) - true_d) < 0.02


def test_fit_floor_plane_orients_normal_toward_camera():
    rng = np.random.default_rng(2)
    true_n = np.array([0.0, -0.7, -0.714])
    true_n /= np.linalg.norm(true_n)
    e1, e2 = plane_basis(true_n)
    origin = -2.5 * true_n
    a = rng.uniform(-2, 2, 20000)
    b = rng.uniform(0, 4, 20000)
    X = origin + a[:, None] * e1 + b[:, None] * e2
    n_fit, _ = fit_floor_plane(X.reshape(-1, 1, 3).astype(np.float32),
                               np.ones((X.shape[0], 1), bool), seed=3)
    assert n_fit[2] < 0  # points toward the camera


def test_floor_region_mask_excludes_upper_image_and_bad_depth():
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    depth[10, 10] = np.inf
    depth[80, 80] = 0.0
    mask = floor_region_mask(depth)
    assert not mask[10, 10]      # upper image + inf
    assert not mask[80, 80]      # zero depth
    assert mask[80, 50]          # lower image, valid depth
    assert not mask[:45].any()   # top 45% excluded
