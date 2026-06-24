"""Depth Pro scene-calibration gate: run once per fixed camera, cache the result.

This is the required first step before the pipeline runs for a camera. It infers
a metric depth map with Apple Depth Pro, back-projects to a metric point cloud
using the self-estimated focal length, RANSAC-fits the dominant floor plane, and
writes a :class:`~storepose.scene.cache.SceneCache`.

Validated on real footage 2026-06-23: clean metric depth, focal ~943 px, floor
plane at 2.69 m below a 44.7 deg-tilted cam, inliers on the actual floor. Depth
Pro OOM-kills on MPS, so inference runs on **CPU** by default (~60 s/frame, paid
once per camera and cached).

Only :func:`infer_depth` needs torch/depth_pro; the plane-fitting helpers are
pure numpy and unit-tested without the model.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .cache import DEFAULT_CACHE_DIR, SceneCache

# RANSAC defaults (tuned on the Counter Overview de-risk frame).
_PLANE_THRESH_M = 0.04
_RANSAC_ITERS = 1000
_RANSAC_MAX_PTS = 200_000
_FLOOR_REGION_TOP = 0.45  # ignore the upper 45% of the image (walls/shelves)
_DEPTH_MIN_M, _DEPTH_MAX_M = 0.2, 30.0


def infer_depth(frame_path: str, device: str = "cpu") -> tuple[np.ndarray, float]:
    """Run Depth Pro on one frame. Returns ``(depth_metres HxW, focal_px)``.

    Defaults to CPU because the model OOM-kills on MPS. torch and depth_pro are
    imported lazily so the rest of the module (and the live pipeline) never pull
    them.
    """
    try:
        import torch
        import depth_pro
    except ImportError as e:  # pragma: no cover - environment guard
        raise ImportError(
            "Depth Pro calibration needs torch + depth_pro (core deps). "
            "Install with: uv pip install torch torchvision "
            "'depth_pro @ git+https://github.com/apple/ml-depth-pro.git'"
        ) from e

    dev = torch.device(device)
    model, transform = depth_pro.create_model_and_transforms(
        device=dev, precision=torch.float32
    )
    model.eval()
    image, _, f_px = depth_pro.load_rgb(frame_path)
    with torch.no_grad():
        pred = model.infer(transform(image), f_px=f_px)
    depth = pred["depth"].detach().cpu().numpy().astype(np.float32)
    focal = float(pred["focallength_px"].detach().cpu().item())
    return depth, focal


def backproject(depth: np.ndarray, focal: float, cx: float, cy: float) -> np.ndarray:
    """Back-project a depth map to a camera-frame point cloud ``(H, W, 3)``."""
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (us - cx) * z / focal
    y = (vs - cy) * z / focal
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def floor_region_mask(depth: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that plausibly belong to the floor.

    Lower part of the image with finite, sensible depth. Excludes the upper
    image (walls, shelves, ceiling) where the floor cannot be.
    """
    h, w = depth.shape
    region = np.zeros((h, w), dtype=bool)
    region[int(h * _FLOOR_REGION_TOP):, :] = True
    region &= np.isfinite(depth) & (depth > _DEPTH_MIN_M) & (depth < _DEPTH_MAX_M)
    return region


def fit_floor_plane(
    points: np.ndarray,
    mask: np.ndarray,
    thresh: float = _PLANE_THRESH_M,
    iters: int = _RANSAC_ITERS,
    max_pts: int = _RANSAC_MAX_PTS,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """RANSAC the dominant plane through ``points[mask]``.

    Returns ``(normal, d)`` for ``normal . X + d = 0`` with unit ``normal``
    oriented toward the camera (``normal_z < 0``), so it is the scene vertical.
    Subsamples to ``max_pts`` for speed/memory (the unguarded full-resolution fit
    OOM-kills).
    """
    rng = np.random.default_rng(seed)
    P = points.reshape(-1, 3)[mask.reshape(-1)]
    if len(P) > max_pts:
        P = P[rng.choice(len(P), size=max_pts, replace=False)]
    n = len(P)
    if n < 3:
        raise ValueError("not enough floor-region points to fit a plane")
    best_cnt, best = -1, None
    for _ in range(iters):
        a, b, c = P[rng.integers(0, n, size=3)]
        nrm = np.cross(b - a, c - a)
        norm = np.linalg.norm(nrm)
        if norm < 1e-6:
            continue
        nrm = nrm / norm
        d = -nrm.dot(a)
        cnt = int((np.abs(P @ nrm + d) < thresh).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, (nrm, d)
    nrm, d = best
    # refit on inliers via SVD for a stable normal
    inl = np.abs(P @ nrm + d) < thresh
    Q = P[inl]
    centroid = Q.mean(axis=0)
    _, _, vt = np.linalg.svd(Q - centroid)
    nrm = vt[-1] / np.linalg.norm(vt[-1])
    d = -nrm.dot(centroid)
    if nrm[2] > 0:  # orient toward camera
        nrm, d = -nrm, -d
    return nrm, float(d)


def calibrate(
    frame_path: str,
    camera_id: str,
    device: str = "cpu",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    store_depth: bool = True,
) -> SceneCache:
    """Full gate: infer depth, fit the floor plane, write and return the cache."""
    depth, focal = infer_depth(frame_path, device=device)
    h, w = depth.shape
    cx, cy = w / 2.0, h / 2.0
    points = backproject(depth, focal, cx, cy)
    normal, d = fit_floor_plane(points, floor_region_mask(depth))
    cache = SceneCache(
        camera_id=camera_id,
        image_size=(w, h),
        focal_px=focal,
        cx=cx,
        cy=cy,
        plane_normal=normal,
        plane_d=d,
        source=os.fspath(frame_path),
    )
    cache.save(cache_dir, depth=depth if store_depth else None)
    return cache
