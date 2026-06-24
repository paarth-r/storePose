"""Persist and load the per-camera scene-geometry cache.

The scene-calibration gate runs Depth Pro once per fixed camera and writes the
result here so the live pipeline never re-runs the model. The cache is the full
scene geometry -- intrinsics, floor plane, floor normal, and (optionally) the
dense depth map and point cloud -- so future features (gaze, 3-D analytics)
reuse the same calibration rather than re-deriving it.

Layout (under ``calib/scene/``):
    <camera_id>.json   intrinsics, plane, normal, image size, units, source
    <camera_id>.npz    depth map (+ point cloud) -- large arrays, optional

The cache holds while the camera is fixed; re-run only if a camera moves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = Path("calib/scene")


@dataclass
class SceneCache:
    """Recovered geometry for one fixed camera.

    Attributes:
        camera_id: Stable id; the cache filename stem.
        image_size: ``(width, height)`` in pixels.
        focal_px: Self-estimated focal length (square pixels).
        cx, cy: Principal point (defaults to image centre).
        plane_normal: ``(3,)`` floor normal in camera coords, oriented toward
            the camera (the scene vertical).
        plane_d: Plane offset ``d`` in ``n . X + d = 0`` (metres); ``|d|`` is the
            camera height above the floor.
        source: Frame/video the calibration came from (provenance).
        depth_path: Filename of the sibling ``.npz`` (depth/point cloud), or None.
    """

    camera_id: str
    image_size: tuple[int, int]
    focal_px: float
    cx: float
    cy: float
    plane_normal: np.ndarray
    plane_d: float
    source: str = ""
    depth_path: str | None = None

    def intrinsics(self) -> np.ndarray:
        from .geometry import intrinsics_matrix

        return intrinsics_matrix(self.focal_px, self.cx, self.cy)

    def camera_view(self):
        """Build the :class:`~storepose.scene.geometry.CameraView` for this cache."""
        from .geometry import CameraView, floor_homography

        H = floor_homography(self.plane_normal, self.plane_d, self.intrinsics())
        return CameraView(camera_id=self.camera_id, H=H, normal=np.asarray(self.plane_normal))

    def save(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        depth: np.ndarray | None = None,
        points: np.ndarray | None = None,
    ) -> Path:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta = asdict(self)
        meta["plane_normal"] = [float(x) for x in np.asarray(self.plane_normal)]
        meta["image_size"] = [int(self.image_size[0]), int(self.image_size[1])]
        if depth is not None or points is not None:
            npz = cache_dir / f"{self.camera_id}.npz"
            arrays = {}
            if depth is not None:
                arrays["depth"] = depth.astype(np.float32)
            if points is not None:
                arrays["points"] = points.astype(np.float32)
            np.savez_compressed(npz, **arrays)
            meta["depth_path"] = npz.name
        json_path = cache_dir / f"{self.camera_id}.json"
        json_path.write_text(json.dumps(meta, indent=2))
        return json_path

    @classmethod
    def load(cls, camera_id: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> "SceneCache":
        cache_dir = Path(cache_dir)
        json_path = cache_dir / f"{camera_id}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"no scene calibration for camera {camera_id!r} at {json_path}. "
                f"Run the Depth Pro calibration gate first."
            )
        meta = json.loads(json_path.read_text())
        meta["image_size"] = tuple(meta["image_size"])
        meta["plane_normal"] = np.asarray(meta["plane_normal"], dtype=np.float64)
        return cls(**meta)

    def load_depth(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> np.ndarray | None:
        if not self.depth_path:
            return None
        npz = np.load(Path(cache_dir) / self.depth_path)
        return npz["depth"] if "depth" in npz else None
