import numpy as np
import pytest

import storepose.tracking.osnet as osnet_mod
from storepose.tracking.osnet import OsnetAppearance, _GALLERY_K


class _FakeSession:
    """Stands in for onnxruntime.InferenceSession: returns fixed rows per crop.

    ``batch_dim`` controls the reported input shape's batch axis: a string for a
    dynamic batch, an int (e.g. 1) for a fixed one.
    """
    batch_dim = "batch"

    def __init__(self, *args, **kwargs):
        self.calls = []

    def get_inputs(self):
        shape = [type(self).batch_dim, 3, 256, 128]

        class _I:
            name = "input"

        _I.shape = shape
        return [_I()]

    def run(self, _outputs, feeds):
        batch = feeds["input"]
        n = batch.shape[0]
        self.calls.append(n)
        # deterministic, distinct, non-normalized rows so we can check L2-norm
        emb = np.zeros((n, 512), np.float32)
        for i in range(n):
            emb[i, i % 512] = float(i + 1) * 3.0
        return [emb]


class _FixedBatchSession(_FakeSession):
    """Fake whose ONNX input fixes the batch dim to 1 (issue #585 shape)."""
    batch_dim = 1


class _Fixed16Session(_FakeSession):
    """Fake whose ONNX input fixes the batch dim to 16 (the anriha export shape)."""
    batch_dim = 16


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(osnet_mod.ort, "InferenceSession", _FakeSession)
    return OsnetAppearance("unused.onnx", device="cpu")


def test_extract_batch_l2_normalized(app):
    frame = np.full((300, 300, 3), 127, np.uint8)
    boxes = [np.array([10, 10, 80, 200], float), np.array([100, 10, 170, 200], float)]
    descs = app.extract_batch(frame, boxes, [None, None], [None, None])
    assert len(descs) == 2
    for d in descs:
        assert d is not None
        assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-5


def test_degenerate_crop_returns_none(app):
    frame = np.full((300, 300, 3), 127, np.uint8)
    descs = app.extract_batch(frame, [np.array([10, 10, 12, 12], float)], [None], [None])
    assert descs == [None]


def test_score_is_max_cosine_over_gallery(app):
    frame = np.full((300, 300, 3), 127, np.uint8)
    # both in one batch so the fake session gives them distinct (orthogonal) rows
    boxes = [np.array([10, 10, 80, 200], float), np.array([100, 10, 170, 200], float)]
    a, b = app.extract_batch(frame, boxes, [None, None], [None, None])
    mem = app.new_memory(a)
    assert app.score(mem, a) == pytest.approx(1.0, abs=1e-5)   # identical -> cosine 1
    assert app.score(mem, b) == pytest.approx(0.0, abs=1e-5)   # orthogonal -> 0
    mem = app.update_memory(mem, b)
    assert app.score(mem, b) == pytest.approx(1.0, abs=1e-5)   # now b is in the gallery


def test_gallery_caps_at_k(app):
    frame = np.full((300, 300, 3), 127, np.uint8)
    d = app.extract_batch(frame, [np.array([10, 10, 80, 200], float)], [None], [None])[0]
    mem = app.new_memory(d)
    for _ in range(_GALLERY_K + 5):
        mem = app.update_memory(mem, d)
    assert len(mem) == _GALLERY_K


def test_score_empty_or_none(app):
    assert app.score(app.new_memory(None), None) == -1.0
    assert app.score(None, np.ones(512, np.float32)) == -1.0


def test_dynamic_batch_runs_once(app):
    frame = np.full((300, 300, 3), 127, np.uint8)
    boxes = [np.array([10, 10, 80, 200], float), np.array([100, 10, 170, 200], float)]
    app.extract_batch(frame, boxes, [None, None], [None, None])
    assert app._session.calls == [2]   # dynamic batch -> one batched forward pass


def test_fixed_batch_runs_per_crop(monkeypatch):
    monkeypatch.setattr(osnet_mod.ort, "InferenceSession", _FixedBatchSession)
    app = OsnetAppearance("unused.onnx", device="cpu")
    frame = np.full((300, 300, 3), 127, np.uint8)
    boxes = [np.array([10, 10, 80, 200], float), np.array([100, 10, 170, 200], float)]
    descs = app.extract_batch(frame, boxes, [None, None], [None, None])
    assert len(descs) == 2 and all(d is not None for d in descs)
    # fixed batch dim -> one run per crop (issue #585 safety), never a 2-row batch
    assert app._session.calls == [1, 1]


def test_fixed_batch_16_pads_partial_group(monkeypatch):
    monkeypatch.setattr(osnet_mod.ort, "InferenceSession", _Fixed16Session)
    app = OsnetAppearance("unused.onnx", device="cpu")
    frame = np.full((300, 300, 3), 127, np.uint8)
    boxes = [np.array([10, 10, 80, 200], float) for _ in range(3)]
    descs = app.extract_batch(frame, boxes, [None] * 3, [None] * 3)
    # fixed batch 16: pad the 3 crops up to 16, run once, slice back to 3
    assert len(descs) == 3 and all(d is not None for d in descs)
    assert app._session.calls == [16]
