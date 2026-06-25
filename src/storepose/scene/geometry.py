"""Floor-plane geometry: project image pixels onto a metric floor frame.

The scene-calibration gate (``scene/depthpro.py``) recovers, per fixed camera, a
floor plane ``n . X + d = 0`` in camera coordinates plus the pinhole intrinsics
(focal, principal point). This module turns that into the projection a heatmap
needs: image pixel ``(u, v)`` -> floor position ``(X, Y)`` in metres.

The map pixel -> floor-2D is an exact homography, derived analytically here so a
``CameraView`` is just two 3x3 matrices: ``H`` (pixel -> floor metres, camera
frame) and ``T_cam_to_store`` (floor metres -> shared store frame; identity for a
single camera). A ``StoreFloor`` is several ``CameraView``s sharing one store
frame -- the multi-camera stitching hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def intrinsics_matrix(focal_px: float, cx: float, cy: float) -> np.ndarray:
    """Pinhole camera matrix ``K`` for a square-pixel camera."""
    return np.array(
        [[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes ``(e1, e2)`` spanning a plane's tangent.

    ``e2 = normal x e1`` so ``(e1, e2, normal)`` is right-handed. The choice of
    ``e1`` is arbitrary (the floor frame has no intrinsic orientation for one
    camera); cross-camera alignment is handled later by ``T_cam_to_store``.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - seed.dot(n) * n
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def floor_homography(
    normal: np.ndarray, d: float, K: np.ndarray
) -> np.ndarray:
    """Exact homography ``H`` mapping image pixels to floor-plane metres.

    For pixel ``p = [u, v, 1]`` the camera ray is ``r = K^-1 p``; it meets the
    plane ``n . X + d = 0`` at ``X = t r`` with ``t = -d / (n . r)``. The floor
    coordinate frame has origin at the foot of the perpendicular from the camera
    (``O = -d n``) and axes ``(e1, e2)`` from :func:`plane_basis`, so the floor
    2-D coordinate is ``(e1 . X, e2 . X)`` (the ``O`` offset drops out since
    ``e_i . O = 0``). Each component is linear in ``p`` after clearing the
    ``n . r`` denominator, giving the 3x3 ``H`` returned here. Apply with
    :meth:`CameraView.apply`.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    e1, e2 = plane_basis(n)
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    KinvT = Kinv.T
    g1 = KinvT @ e1
    g2 = KinvT @ e2
    m = KinvT @ n
    H = np.stack([-d * g1, -d * g2, m], axis=0)
    return H


@dataclass
class CameraView:
    """One fixed camera registered into a shared store floor frame.

    Attributes:
        camera_id: Stable identifier (matches the scene-cache filename).
        H: ``(3, 3)`` homography, image pixel -> floor metres in camera frame.
        normal: ``(3,)`` floor-plane normal in camera coords (the scene
            vertical), oriented toward the camera. Used by the anchor module to
            drop occluded bodies to the floor.
        T_cam_to_store: ``(3, 3)`` 2-D transform from this camera's floor frame
            into the shared store frame. Identity until cross-camera alignment
            is solved.
    """

    camera_id: str
    H: np.ndarray
    normal: np.ndarray
    T_cam_to_store: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )

    def apply(self, u: float, v: float) -> tuple[float, float]:
        """Project image pixel ``(u, v)`` to ``(X, Y)`` in the store frame."""
        p = np.array([u, v, 1.0])
        q = self.H @ p
        cam = np.array([q[0] / q[2], q[1] / q[2], 1.0])
        s = self.T_cam_to_store @ cam
        return float(s[0] / s[2]), float(s[1] / s[2])


@dataclass
class StoreFloor:
    """Several cameras sharing one floor coordinate system."""

    cameras: list[CameraView] = field(default_factory=list)

    def add(self, view: CameraView) -> None:
        self.cameras.append(view)

    def get(self, camera_id: str) -> CameraView:
        for c in self.cameras:
            if c.camera_id == camera_id:
                return c
        raise KeyError(f"no camera {camera_id!r} in store floor")
