"""Foundational scene geometry from depth: the calibration gate and floor frame.

Depth Pro runs once per fixed camera (``depthpro.calibrate``) and writes a
:class:`~storepose.scene.cache.SceneCache`. Downstream code builds a
:class:`~storepose.scene.geometry.CameraView` from that cache to project image
pixels onto a metric floor frame. Shared infrastructure for the heatmap and
future features (gaze, 3-D analytics).
"""

from __future__ import annotations

from .cache import SceneCache
from .geometry import CameraView, StoreFloor, floor_homography, intrinsics_matrix

__all__ = [
    "SceneCache",
    "CameraView",
    "StoreFloor",
    "floor_homography",
    "intrinsics_matrix",
]
