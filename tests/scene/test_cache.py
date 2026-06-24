"""Scene cache round-trips and fails loudly when absent."""

import numpy as np
import pytest

from storepose.scene.cache import SceneCache


def _cache():
    return SceneCache(
        camera_id="cam0",
        image_size=(1920, 1080),
        focal_px=943.0,
        cx=960.0,
        cy=540.0,
        plane_normal=np.array([0.0, -0.703, -0.711]),
        plane_d=2.69,
        source="videos/x.mp4",
    )


def test_save_load_round_trip(tmp_path):
    _cache().save(tmp_path)
    loaded = SceneCache.load("cam0", tmp_path)
    assert loaded.camera_id == "cam0"
    assert loaded.image_size == (1920, 1080)
    assert np.isclose(loaded.focal_px, 943.0)
    assert np.allclose(loaded.plane_normal, [0.0, -0.703, -0.711])
    assert np.isclose(loaded.plane_d, 2.69)


def test_missing_cache_raises_with_guidance(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run the Depth Pro calibration"):
        SceneCache.load("nope", tmp_path)


def test_depth_round_trip(tmp_path):
    depth = np.random.default_rng(0).uniform(0.5, 8.0, (64, 96)).astype(np.float32)
    _cache().save(tmp_path, depth=depth)
    loaded = SceneCache.load("cam0", tmp_path)
    assert loaded.depth_path == "cam0.npz"
    back = loaded.load_depth(tmp_path)
    assert np.allclose(back, depth)


def test_camera_view_built_from_cache(tmp_path):
    view = _cache().camera_view()
    assert view.camera_id == "cam0"
    assert view.H.shape == (3, 3)
    # a lower-image pixel projects to a finite floor point
    x, y = view.apply(960, 850)
    assert np.isfinite(x) and np.isfinite(y)
