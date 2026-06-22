# Re-id OSNet Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the weak HSV-histogram re-id descriptor with a learned OSNet ONNX embedding (default-on, histogram fallback) and a per-track feature gallery, so a returning person reliably keeps their id across crossings, similar clothing, lighting shifts, and full frame exits.

**Architecture:** The `AppearanceModel` seam grows from `extract`/`similarity` to `extract`/`extract_batch`/`new_memory`/`update_memory`/`score`, moving per-track appearance memory out of `Track` and behind the model. `HsvHistogramAppearance` keeps single-vector EMA memory; a new `OsnetAppearance` keeps a capped feature gallery scored by max cosine. The tracker re-attaches gallery (lost) candidates on appearance alone (no spatial gate, slightly stricter threshold) to enable cross-exit re-entry, while in-frame candidates keep the spatial gate. A launcher TUI cycle-column selects `osnet-x1 / osnet-x025 / histogram / off` per run.

**Tech Stack:** Python 3.12, numpy, OpenCV (`cv2`), onnxruntime (already a dep via rtmlib), pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-06-22-reid-osnet-embedding-design.md`

## Global Constraints

- Single-line shell commands only — never use `\` line-continuations.
- No emojis in any code, comment, commit message, or doc.
- `device` is only `"cpu"` or `"mps"` (CoreML); map `"mps"` to `CoreMLExecutionProvider` with a CPU fallback provider.
- Precision-biased: when a re-attach is uncertain, spawn a new id rather than risk a false merge.
- Re-id stays off when `--no-track` / `config.reid` is false (appearance model is `None`).
- OSNet is default-on: a missing/broken model must warn and fall back to the histogram, never crash.
- Backends: `REID_BACKENDS = ("osnet-x1", "osnet-x025", "histogram")`, default `"osnet-x1"`.
- Per-backend `reid_thr` default: OSNet (either size) -> `0.5`, histogram -> `0.6`. `--reid-thr` overrides.
- Gallery size `K = 10`; gallery threshold margin `0.05` (internal constants, not flags).

---

## File Structure

- Modify: `src/storepose/tracking/appearance.py` — protocol gains the 4 new seam methods; `HsvHistogramAppearance` implements them (single-vector EMA memory).
- Create: `src/storepose/tracking/osnet.py` — `OsnetAppearance` (crop -> ONNX -> L2-normalized embedding; feature-gallery memory; max-cosine score).
- Create: `src/storepose/tracking/reid_zoo.py` — pinned OSNet ONNX URL+sha resolver with download+cache.
- Modify: `src/storepose/tracking/track.py` — replace `descriptor` field/EMA with one opaque `appearance_mem` slot.
- Modify: `src/storepose/tracking/tracker.py` — `extract_batch`, memory via the seam, split spatial gate, stricter gallery threshold.
- Modify: `src/storepose/config.py` — `reid_backend`, `reid_weights`, `reid_thr` (None sentinel), `reid_thr_for()`, CLI.
- Modify: `src/storepose/runner.py` — `build_tracker` selects/constructs the backend with fallback and resolves the threshold.
- Modify: `src/storepose/launcher_core.py` — `Column.REID`, `REID_CYCLE`, `ColumnState.reid`, `toggle`, `build_run`.
- Modify: `README.md`, `docs/usage.md`.
- Create: `tests/tracking/test_osnet.py`. Modify: `tests/tracking/test_appearance.py`, `tests/tracking/test_track.py`, `tests/tracking/test_tracker.py`, `tests/test_config.py`, `tests/test_launcher_core.py`.

---

## Task 1: Appearance seam — add memory methods to the protocol + histogram

**Files:**
- Modify: `src/storepose/tracking/appearance.py`
- Test: `tests/tracking/test_appearance.py`

**Interfaces:**
- Produces: `AppearanceModel` protocol with `extract(frame, box, keypoints, scores) -> np.ndarray | None`, `extract_batch(frame, boxes, keypoints, scores) -> list[np.ndarray | None]`, `new_memory(desc) -> object`, `update_memory(mem, desc) -> object`, `score(mem, desc) -> float`. `HsvHistogramAppearance` memory = a single histogram ndarray (or `None`); `update_memory` EMA-blends at `alpha=0.3`; `score` = `HISTCMP_CORREL`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/tracking/test_appearance.py
def test_histogram_memory_seed_update_score():
    app = HsvHistogramAppearance(kpt_thr=0.5)
    box = np.array([40, 40, 120, 160], float)
    red = app.extract(_frame_with_rect((0, 0, 220), 40, 40, 120, 160), box, None, None)
    blue = app.extract(_frame_with_rect((220, 0, 0), 40, 40, 120, 160), box, None, None)
    mem = app.new_memory(red)
    assert app.score(mem, red) > 0.8
    assert app.score(mem, blue) < 0.3
    # update_memory blends toward the new observation but keeps it a single hist
    mem2 = app.update_memory(mem, blue)
    assert mem2.shape == red.shape


def test_histogram_memory_handles_none():
    app = HsvHistogramAppearance(kpt_thr=0.5)
    assert app.new_memory(None) is None
    assert app.score(None, None) == -1.0
    desc = app.extract(_frame_with_rect((0, 0, 220), 40, 40, 120, 160),
                       np.array([40, 40, 120, 160], float), None, None)
    assert app.update_memory(None, desc) is not None        # seeds from None
    assert np.array_equal(app.update_memory(desc, None), desc)  # None obs keeps mem


def test_histogram_extract_batch_aligns():
    app = HsvHistogramAppearance(kpt_thr=0.5)
    f = _frame_with_rect((0, 0, 220), 40, 40, 120, 160)
    descs = app.extract_batch(f, [np.array([40, 40, 120, 160], float)], [None], [None])
    assert len(descs) == 1 and descs[0] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tracking/test_appearance.py -k "memory or extract_batch" -v`
Expected: FAIL — `AttributeError: 'HsvHistogramAppearance' object has no attribute 'new_memory'`.

- [ ] **Step 3: Write minimal implementation**

In `src/storepose/tracking/appearance.py`, add the EMA constant after the bin constants (line ~17):

```python
_DESC_ALPHA = 0.3  # histogram EMA weight for a new observation
```

Replace the `AppearanceModel` protocol body with the new contract:

```python
class AppearanceModel(Protocol):
    """Extracts per-person descriptors and owns per-track appearance memory."""

    def extract(self, frame, box, keypoints, scores) -> np.ndarray | None: ...
    def extract_batch(self, frame, boxes, keypoints, scores) -> list[np.ndarray | None]: ...
    def new_memory(self, desc) -> object: ...
    def update_memory(self, mem, desc) -> object: ...
    def score(self, mem, desc) -> float: ...
```

In `HsvHistogramAppearance`, keep `extract` and `similarity` as-is and add:

```python
    def extract_batch(self, frame, boxes, keypoints, scores) -> list[np.ndarray | None]:
        return [
            self.extract(frame, boxes[i], keypoints[i], scores[i])
            for i in range(len(boxes))
        ]

    def new_memory(self, desc):
        return None if desc is None else np.array(desc, dtype=np.float32)

    def update_memory(self, mem, desc):
        if desc is None:
            return mem
        desc = np.array(desc, dtype=np.float32)
        if mem is None:
            return desc
        return ((1.0 - _DESC_ALPHA) * mem + _DESC_ALPHA * desc).astype(np.float32)

    def score(self, mem, desc) -> float:
        if mem is None or desc is None:
            return -1.0
        return float(cv2.compareHist(mem, desc, cv2.HISTCMP_CORREL))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tracking/test_appearance.py -v`
Expected: PASS (existing extract tests + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/storepose/tracking/appearance.py tests/tracking/test_appearance.py
git commit -m "feat: appearance seam gains per-track memory methods (histogram)"
```

---

## Task 2: OSNet embedding model + weight resolver

**Files:**
- Create: `src/storepose/tracking/reid_zoo.py`
- Create: `src/storepose/tracking/osnet.py`
- Test: `tests/tracking/test_osnet.py`

**Interfaces:**
- Consumes: the `AppearanceModel` seam from Task 1.
- Produces: `OsnetAppearance(weights_path: str, device: str = "cpu")` implementing the seam; memory = a `collections.deque(maxlen=10)` of L2-normalized `(512,)` float32 embeddings; `score(mem, desc)` = max dot product. `reid_zoo.resolve(backend: str) -> pathlib.Path` returns a cached, checksum-verified ONNX path, downloading on first use; `reid_zoo.SPECS: dict[str, ReidSpec]` holds the pinned URL+sha per backend.

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_osnet.py
import numpy as np
import pytest

import storepose.tracking.osnet as osnet_mod
from storepose.tracking.osnet import OsnetAppearance, _GALLERY_K


class _FakeSession:
    """Stands in for onnxruntime.InferenceSession: returns fixed rows per crop."""
    def __init__(self, *args, **kwargs):
        pass

    def get_inputs(self):
        class _I:
            name = "input"
        return [_I()]

    def run(self, _outputs, feeds):
        batch = feeds["input"]
        n = batch.shape[0]
        # deterministic, distinct, non-normalized rows so we can check L2-norm
        emb = np.zeros((n, 512), np.float32)
        for i in range(n):
            emb[i, i % 512] = float(i + 1) * 3.0
        return [emb]


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
    a = app.extract_batch(frame, [np.array([10, 10, 80, 200], float)], [None], [None])[0]
    b = app.extract_batch(frame, [np.array([100, 10, 170, 200], float)], [None], [None])[0]
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tracking/test_osnet.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'storepose.tracking.osnet'`.

- [ ] **Step 3: Write the resolver**

```python
# src/storepose/tracking/reid_zoo.py
"""Resolve and cache OSNet ReID ONNX weights (the ReID analogue of model_zoo).

URLs and checksums are pinned (see the plan's "obtain weights" step). Weights
download once into ~/.cache/storepose/reid/ and are checksum-verified on use.
"""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_CACHE_DIR = Path.home() / ".cache" / "storepose" / "reid"


@dataclass(frozen=True)
class ReidSpec:
    """A downloadable OSNet ONNX model."""
    url: str
    sha256: str
    filename: str


# Pinned in the "obtain weights" step below. Both 512-d embedding models.
SPECS: dict[str, ReidSpec] = {
    "osnet-x1": ReidSpec(url="", sha256="", filename="osnet_x1_0.onnx"),
    "osnet-x025": ReidSpec(url="", sha256="", filename="osnet_x0_25.onnx"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(backend: str) -> Path:
    """Return a local path to ``backend``'s ONNX weights, downloading if needed."""
    try:
        spec = SPECS[backend]
    except KeyError as exc:
        raise ValueError(f"no ReID weights for backend {backend!r}") from exc
    if not spec.url:
        raise RuntimeError(f"ReID weights URL for {backend!r} is not pinned yet")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CACHE_DIR / spec.filename
    if not dest.is_file():
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(spec.url, tmp)
        tmp.replace(dest)
    actual = _sha256(dest)
    if spec.sha256 and actual != spec.sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"ReID weights {spec.filename} checksum mismatch "
            f"(expected {spec.sha256}, got {actual})"
        )
    return dest
```

- [ ] **Step 4: Write the OSNet model**

```python
# src/storepose/tracking/osnet.py
"""OSNet ReID embedding appearance model (learned, replaces the histogram).

Implements the AppearanceModel seam with a per-track feature gallery scored by
max cosine similarity (min cosine distance) -- the DeepSORT/StrongSORT standard.
Weights are auto-downloaded ONNX (see reid_zoo); --reid-weights overrides.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

_INPUT_H, _INPUT_W = 256, 128
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
_GALLERY_K = 10  # embeddings retained per track
_MIN_CROP = 4    # reject crops smaller than this (px) on either side


def _providers(device: str) -> list[str]:
    if device == "mps":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OsnetAppearance:
    """Full-body crop -> OSNet ONNX -> L2-normalized embedding; gallery memory."""

    def __init__(self, weights_path: str, device: str = "cpu"):
        self._session = ort.InferenceSession(
            weights_path, providers=_providers(device)
        )
        self._input = self._session.get_inputs()[0].name

    def _crop(self, frame, box):
        h, w = frame.shape[:2]
        x1 = max(0, min(int(box[0]), w - 1))
        x2 = max(0, min(int(box[2]), w))
        y1 = max(0, min(int(box[1]), h - 1))
        y2 = max(0, min(int(box[3]), h))
        if x2 - x1 < _MIN_CROP or y2 - y1 < _MIN_CROP:
            return None
        return frame[y1:y2, x1:x2]

    def _preprocess(self, crop):
        img = cv2.resize(crop, (_INPUT_W, _INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return (img - _MEAN) / _STD

    def extract(self, frame, box, keypoints, scores):
        return self.extract_batch(frame, [box], [keypoints], [scores])[0]

    def extract_batch(self, frame, boxes, keypoints, scores):
        out: list[np.ndarray | None] = [None] * len(boxes)
        crops, idx = [], []
        for i, box in enumerate(boxes):
            c = self._crop(frame, box)
            if c is not None:
                crops.append(self._preprocess(c))
                idx.append(i)
        if not crops:
            return out
        batch = np.stack(crops).astype(np.float32)
        emb = np.asarray(self._session.run(None, {self._input: batch})[0], np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        for j, i in enumerate(idx):
            out[i] = emb[j]
        return out

    def new_memory(self, desc):
        g: deque = deque(maxlen=_GALLERY_K)
        if desc is not None:
            g.append(np.asarray(desc, np.float32))
        return g

    def update_memory(self, mem, desc):
        if mem is None:
            mem = deque(maxlen=_GALLERY_K)
        if desc is not None:
            mem.append(np.asarray(desc, np.float32))
        return mem

    def score(self, mem, desc) -> float:
        if mem is None or len(mem) == 0 or desc is None:
            return -1.0
        d = np.asarray(desc, np.float32)
        return float(max(float(np.dot(e, d)) for e in mem))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/tracking/test_osnet.py -v`
Expected: PASS (5 tests; the fake session makes them weight-free and offline).

- [ ] **Step 6: Obtain and pin the real weights**

This is the one value that must be discovered at implementation time. Acquire an
`osnet_x1_0` and `osnet_x0_25` ONNX export (e.g. export from
`KaiyangZhou/deep-person-reid` with `torch.onnx.export`, or fetch a community
ONNX from the `mikel-brostrom/boxmot` release assets). For each file, host it at
a stable URL (or use a GitHub release asset URL you control), then compute its
checksum:

Run: `python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" /path/to/osnet_x1_0.onnx`

Paste the URL and the printed sha256 into `SPECS` in `reid_zoo.py` for each
backend. Then verify a real download end-to-end (delete the cache first):

Run: `rm -rf ~/.cache/storepose/reid && uv run python -c "from storepose.tracking.reid_zoo import resolve; print(resolve('osnet-x1')); print(resolve('osnet-x025'))"`
Expected: prints two cached `.onnx` paths with no checksum error.

Confirm the model loads and emits a 512-d embedding:

Run: `uv run python -c "import numpy as np; from storepose.tracking.osnet import OsnetAppearance; from storepose.tracking.reid_zoo import resolve; a=OsnetAppearance(str(resolve('osnet-x1')),'cpu'); f=np.full((400,300,3),127,np.uint8); d=a.extract(f,np.array([20,20,120,360],float),None,None); print(d.shape, float(np.linalg.norm(d)))"`
Expected: `(512,) 1.0` (or `(512,)` with norm ~1.0).

If the embedding dim is not 512, update the test's `_FakeSession` width and any
doc references to match the real export.

- [ ] **Step 7: Commit**

```bash
git add src/storepose/tracking/osnet.py src/storepose/tracking/reid_zoo.py tests/tracking/test_osnet.py
git commit -m "feat: OSNet ReID embedding model + cached ONNX weight resolver"
```

---

## Task 3: Migrate Track + tracker to opaque appearance memory; split the gate

**Files:**
- Modify: `src/storepose/tracking/track.py`
- Modify: `src/storepose/tracking/tracker.py`
- Test: `tests/tracking/test_track.py`, `tests/tracking/test_tracker.py`

**Interfaces:**
- Consumes: the `AppearanceModel` seam (`extract_batch`, `new_memory`, `update_memory`, `score`).
- Produces: `Track` exposes `appearance_mem` (opaque, set by the tracker); `Track.__init__`/`update`/`reactivate` accept `appearance_mem=None` (assigned only when not `None`). The tracker uses `extract_batch` once per frame, updates memory through the model, gates gallery candidates on appearance alone at `reid_thr + 0.05`, and gates active candidates spatially at `reid_thr`.

- [ ] **Step 1: Write the failing Track test**

Replace the existing `descriptor`-based tests in `tests/tracking/test_track.py`
(the `test_descriptor_*` / `test_reactivate_*` / `test_update_descriptor_*`
functions and the `descriptor=` arg in their `_track` helper) with:

```python
def _track(box, appearance_mem=None):
    return Track(0, np.array(box, float), None, None, 1 / 30,
                 min_hits=1, smooth=False, min_cutoff=1.0, beta=0.007,
                 appearance_mem=appearance_mem)


def test_appearance_mem_stored_on_init():
    t = _track([0, 0, 10, 20], appearance_mem=["m0"])
    assert t.appearance_mem == ["m0"]


def test_update_replaces_mem_when_given():
    t = _track([0, 0, 10, 20], appearance_mem=["m0"])
    t.update(np.array([0, 0, 10, 20], float), None, None, 1 / 30, appearance_mem=["m1"])
    assert t.appearance_mem == ["m1"]


def test_update_none_mem_keeps_previous():
    t = _track([0, 0, 10, 20], appearance_mem=["m0"])
    t.update(np.array([0, 0, 10, 20], float), None, None, 1 / 30, appearance_mem=None)
    assert t.appearance_mem == ["m0"]


def test_reactivate_reseats_motion_and_keeps_identity():
    t = _track([0, 0, 10, 20], appearance_mem=["m0"])
    t.confirmed = True
    for _ in range(5):
        t.predict()
    assert t.time_since_update == 5
    t.reactivate(np.array([100, 100, 110, 120], float), None, None, 1 / 30,
                 appearance_mem=["m1"])
    assert t.time_since_update == 0
    assert t.confirmed is True
    assert np.allclose(t.box, [100, 100, 110, 120], atol=1.0)
    assert t.appearance_mem == ["m1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tracking/test_track.py -k "appearance_mem or reactivate" -v`
Expected: FAIL — `Track.__init__() got an unexpected keyword argument 'appearance_mem'`.

- [ ] **Step 3: Update Track**

In `src/storepose/tracking/track.py`: delete the `_DESC_ALPHA` constant and the
`_update_descriptor` method. Change the `descriptor` parameter/attribute to
`appearance_mem` in `__init__`, `update`, and `reactivate`.

`__init__`: replace the `descriptor` param and the `self.descriptor = (...)`
block with:

```python
        appearance_mem: object | None = None,
        det_score: float | None = None,
    ):
        self.id = track_id
        self.det_score = det_score  # detector confidence of the last detection
        self.kalman = KalmanBoxTracker(box)
        self.last_box = np.asarray(box, float)  # last *detected* box (not coasted)
        self.hits = 1
        self.time_since_update = 0
        self.min_hits = min_hits
        self.confirmed = min_hits <= 1
        self.color = color_for(track_id)
        self._smoother = (
            KeypointSmoother(min_cutoff=min_cutoff, beta=beta) if smooth else None
        )
        self.keypoints: np.ndarray | None = None
        self.scores: np.ndarray | None = None
        self.appearance_mem = appearance_mem  # opaque; owned by the AppearanceModel
        self._ingest_pose(keypoints, scores, dt)
```

`update`:

```python
    def update(self, box, keypoints, scores, dt, appearance_mem=None, det_score=None) -> None:
        """Correct with a matched detection."""
        self.kalman.update(box)
        self.last_box = np.asarray(box, float)
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self.min_hits:
            self.confirmed = True
        self.det_score = det_score
        self._ingest_pose(keypoints, scores, dt)
        if appearance_mem is not None:
            self.appearance_mem = appearance_mem
```

`reactivate`:

```python
    def reactivate(self, box, keypoints, scores, dt, appearance_mem=None, det_score=None) -> None:
        """Revive a lost/coasting track at a new detection, keeping id and color."""
        self.kalman = KalmanBoxTracker(box)
        self.last_box = np.asarray(box, float)
        self.time_since_update = 0
        self.hits += 1
        self.confirmed = True
        self.det_score = det_score
        self._ingest_pose(keypoints, scores, dt)
        if appearance_mem is not None:
            self.appearance_mem = appearance_mem
```

- [ ] **Step 4: Run the Track tests**

Run: `uv run pytest tests/tracking/test_track.py -v`
Expected: PASS (existing lifecycle tests + the 4 new appearance_mem tests).

- [ ] **Step 5: Update the tracker stub and write new re-attach tests**

In `tests/tracking/test_tracker.py`, replace the `_ColorStub` class (which has
only `extract`/`similarity`) with the full seam, and add two tests after the
existing re-attach tests:

```python
class _ColorStub:
    """Appearance bound to the BGR pixel at a box center; gallery = list of colors."""
    def extract(self, frame, box, keypoints, scores):
        cx = int((box[0] + box[2]) / 2.0); cy = int((box[1] + box[3]) / 2.0)
        h, w = frame.shape[:2]
        cx = min(max(cx, 0), w - 1); cy = min(max(cy, 0), h - 1)
        px = frame[cy, cx]
        return None if int(px.sum()) == 0 else px.astype(np.float32)

    def extract_batch(self, frame, boxes, keypoints, scores):
        return [self.extract(frame, b, None, None) for b in boxes]

    def new_memory(self, desc):
        return [] if desc is None else [desc]

    def update_memory(self, mem, desc):
        mem = list(mem or [])
        if desc is not None:
            mem.append(desc)
        return mem

    def score(self, mem, desc):
        if not mem or desc is None:
            return -1.0
        return 1.0 if any(np.array_equal(e, desc) for e in mem) else 0.0


class _ConstScoreStub:
    """Appearance whose score is a fixed value, to probe the gallery margin."""
    def __init__(self, value):
        self.value = value

    def extract(self, frame, box, keypoints, scores):
        return np.array([1.0], np.float32)

    def extract_batch(self, frame, boxes, keypoints, scores):
        return [np.array([1.0], np.float32) for _ in boxes]

    def new_memory(self, desc):
        return [desc]

    def update_memory(self, mem, desc):
        return list(mem or []) + ([] if desc is None else [desc])

    def score(self, mem, desc):
        return self.value
```

```python
def test_gallery_reattaches_across_full_frame_exit():
    tr = _reid_tracker()  # max_age=3, reid_max_age=50
    left = [20, 180, 60, 260]; red = (0, 0, 220)
    out = tr.update(make_result([left]), 1 / 30, _frame([(left, red)]))
    assert out[0].id == 0
    blank = np.zeros((400, 400, 3), np.uint8)
    for _ in range(6):                 # age out past max_age -> gallery
        tr.update(make_result([]), 1 / 30, blank)
    far = [340, 180, 380, 260]         # opposite side, beyond the old spatial gate cap
    out2 = tr.update(make_result([far]), 1 / 30, _frame([(far, red)]))
    assert len(out2) == 1 and out2[0].id == 0   # gallery has no spatial gate


def test_gallery_threshold_is_stricter_than_active():
    # score 0.52 clears the active floor (0.5) but not the gallery floor (0.55)
    tr = MultiObjectTracker(
        max_age=2, min_hits=1, iou_thr=0.3, smooth=False,
        appearance=_ConstScoreStub(0.52), reid=True, reid_max_age=50, reid_thr=0.5,
    )
    box = [100, 100, 140, 180]
    f = _frame([(box, (0, 0, 220))])
    assert tr.update(make_result([box]), 1 / 30, f)[0].id == 0
    blank = np.zeros((400, 400, 3), np.uint8)
    for _ in range(4):                 # age out -> gallery
        tr.update(make_result([]), 1 / 30, blank)
    back = [110, 105, 150, 185]
    out = tr.update(make_result([back]), 1 / 30, _frame([(back, (0, 0, 220))]))
    assert out[0].id == 1              # gallery match rejected -> new id
```

- [ ] **Step 6: Run tracker tests to verify the new ones fail**

Run: `uv run pytest tests/tracking/test_tracker.py -k "gallery or reattach" -v`
Expected: FAIL — the tracker still calls `extract`/`similarity` and reads
`t.descriptor`, so `_ColorStub` (now without `similarity`) and the new gallery
behavior break.

- [ ] **Step 7: Update the tracker**

In `src/storepose/tracking/tracker.py`:

Add the gallery margin constant after `_GATE_CAP_FRAC` (line ~18):

```python
_GALLERY_MARGIN = 0.05  # gallery (lost) re-attach needs reid_thr + this (no spatial gate)
```

Drop the `descriptor` field from `_Candidate` (read it from the track instead):

```python
@dataclass
class _Candidate:
    """A re-attach candidate: an unmatched active track or a gallery entry."""
    track: Track
    lost: _LostEntry | None
    center: tuple[float, float]
    gap: int
```

In `update`, replace the descriptor-extraction block (the `if use_reid: det_descs = [...]` list comprehension) with a single batched call:

```python
        use_reid = self._reid and self._appearance is not None and frame is not None
        if use_reid:
            det_descs = self._appearance.extract_batch(frame, boxes, keypoints, scores)
        else:
            det_descs = [None] * n
```

Replace the matched-update loop (step 3) so memory flows through the model:

```python
        # 3. update matched tracks
        for d, tr in matches:
            t = self._tracks[tr]
            mem = (
                self._appearance.update_memory(t.appearance_mem, det_descs[d])
                if use_reid else None
            )
            t.update(
                boxes[d], keypoints[d], scores[d], dt,
                appearance_mem=mem, det_score=float(det_scores[d]),
            )
```

Replace the spawn loop (step 5):

```python
        # 5. spawn tracks for still-unmatched detections
        for d in unmatched_dets:
            mem = self._appearance.new_memory(det_descs[d]) if use_reid else None
            self._tracks.append(
                Track(
                    self._next_id, boxes[d], keypoints[d], scores[d], dt,
                    min_hits=self.min_hits, smooth=self.smooth,
                    min_cutoff=self.min_cutoff, beta=self.beta,
                    appearance_mem=mem, det_score=float(det_scores[d]),
                )
            )
            self._next_id += 1
```

In the cull-to-gallery block (step 6), change the gallery guard from
`t.descriptor is not None` to `t.appearance_mem is not None`:

```python
            if t.time_since_update > self.max_age:
                if use_reid and t.confirmed and t.appearance_mem is not None:
                    self._lost.append(
                        _LostEntry(track=t, center=_center(t.last_box), lost_age=0)
                    )
                continue
```

Replace `_reattach` with the split-gate version:

```python
    def _reattach(
        self, unmatched_dets, unmatched_tracks, boxes, keypoints, scores,
        det_descs, frame, dt, det_scores,
    ) -> list[int]:
        """Revive ids for leftover detections via gated appearance match.

        Active (in-frame) candidates keep the spatial gate at ``reid_thr``;
        gallery (lost) candidates drop the spatial gate and use a stricter
        ``reid_thr + _GALLERY_MARGIN`` so cross-exit re-entry stays precise.
        Returns the detection indices that remain unmatched (to be spawned).
        """
        diag = float(np.hypot(frame.shape[1], frame.shape[0]))

        cands: list[_Candidate] = []
        for tr_idx in unmatched_tracks:
            t = self._tracks[tr_idx]
            if not t.confirmed or t.appearance_mem is None:
                continue
            cands.append(_Candidate(t, None, _center(t.last_box), t.time_since_update))
        for e in self._lost:
            if e.track.appearance_mem is None:
                continue
            cands.append(_Candidate(e.track, e, e.center, e.lost_age))
        if not cands:
            return unmatched_dets

        gallery_thr = min(1.0, self._reid_thr + _GALLERY_MARGIN)
        pairs: list[tuple[float, int, int]] = []
        for d in unmatched_dets:
            dd = det_descs[d]
            if dd is None:
                continue
            dcx, dcy = _center(boxes[d])
            for ci, cand in enumerate(cands):
                if cand.lost is None:
                    radius = min(
                        _SPATIAL_GATE_FRAC * diag * (1.0 + _GATE_GROWTH * cand.gap),
                        _GATE_CAP_FRAC * diag,
                    )
                    if np.hypot(dcx - cand.center[0], dcy - cand.center[1]) > radius:
                        continue
                    thr = self._reid_thr
                else:
                    thr = gallery_thr
                sim = self._appearance.score(cand.track.appearance_mem, dd)
                if sim < thr:
                    continue
                pairs.append((1.0 - sim, d, ci))

        pairs.sort(key=lambda p: p[0])
        used_d: set[int] = set()
        used_c: set[int] = set()
        for _cost, d, ci in pairs:
            if d in used_d or ci in used_c:
                continue
            used_d.add(d)
            used_c.add(ci)
            cand = cands[ci]
            mem = self._appearance.update_memory(cand.track.appearance_mem, det_descs[d])
            cand.track.reactivate(boxes[d], keypoints[d], scores[d], dt,
                                  appearance_mem=mem, det_score=float(det_scores[d]))
            if cand.lost is not None:
                self._lost.remove(cand.lost)
                self._tracks.append(cand.track)
        return [d for d in unmatched_dets if d not in used_d]
```

- [ ] **Step 8: Run the tracking suite**

Run: `uv run pytest tests/tracking -v`
Expected: PASS (all existing tracker/track/appearance/osnet tests + the 2 new
re-attach tests). The no-frame/no-appearance tests still pass because `use_reid`
is false and `appearance_mem` stays `None`.

- [ ] **Step 9: Commit**

```bash
git add src/storepose/tracking/track.py src/storepose/tracking/tracker.py tests/tracking/test_track.py tests/tracking/test_tracker.py
git commit -m "feat: per-track appearance gallery via seam; split re-attach gate for cross-exit re-entry"
```

---

## Task 4: Config — backend selector, weights override, per-backend threshold

**Files:**
- Modify: `src/storepose/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `AppConfig.reid_backend: str` (default `"osnet-x1"`), `AppConfig.reid_weights: str | None`, `AppConfig.reid_thr: float | None` (None = use the per-backend default), `REID_BACKENDS`, and `reid_thr_for(backend: str, override: float | None) -> float`. CLI: `--reid-backend`, `--reid-weights`, `--reid-thr` (default `None`).

- [ ] **Step 1: Write the failing test**

In `tests/test_config.py`, update the existing `test_reid_defaults_on` to expect
`reid_thr is None` (the default now resolves per backend in the runner), then add:

```python
def test_reid_defaults_on():
    cfg = from_args([])
    assert cfg.reid is True
    assert cfg.reid_backend == "osnet-x1"
    assert cfg.reid_thr is None


def test_reid_backend_choice_parses():
    assert from_args(["--reid-backend", "histogram"]).reid_backend == "histogram"
    assert from_args(["--reid-backend", "osnet-x025"]).reid_backend == "osnet-x025"


def test_reid_weights_override_parses():
    assert from_args(["--reid-weights", "/tmp/x.onnx"]).reid_weights == "/tmp/x.onnx"


def test_reid_backend_must_be_known():
    import pytest
    with pytest.raises(ValueError):
        AppConfig(reid_backend="osnet-x99")


def test_reid_thr_for_resolves_per_backend():
    from storepose.config import reid_thr_for
    assert reid_thr_for("osnet-x1", None) == 0.5
    assert reid_thr_for("osnet-x025", None) == 0.5
    assert reid_thr_for("histogram", None) == 0.6
    assert reid_thr_for("osnet-x1", 0.3) == 0.3    # explicit override wins


def test_reid_thr_override_validates_range():
    import pytest
    with pytest.raises(ValueError):
        AppConfig(reid_thr=2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -k reid -v`
Expected: FAIL — `AttributeError: 'AppConfig' object has no attribute 'reid_backend'`.

- [ ] **Step 3: Implement config**

In `src/storepose/config.py`, after the `MODES`/`DEVICES` constants, add:

```python
REID_BACKENDS = ("osnet-x1", "osnet-x025", "histogram")
_REID_THR_DEFAULTS = {"osnet-x1": 0.5, "osnet-x025": 0.5, "histogram": 0.6}


def reid_thr_for(backend: str, override: float | None) -> float:
    """Resolve the appearance similarity floor: explicit override, else per-backend."""
    if override is not None:
        return override
    return _REID_THR_DEFAULTS[backend]
```

In `AppConfig`, change the `reid_thr` field and add the two new fields (replace
the existing `reid_thr: float = 0.6` line):

```python
    reid_backend: str = "osnet-x1"
    reid_weights: str | None = None
    reid_thr: float | None = None
```

Update the `reid`/`reid_seconds`/`reid_thr` lines in the docstring Attributes
list to add:

```python
        reid_backend: Appearance backend / OSNet size (one of ``REID_BACKENDS``).
        reid_weights: Local OSNet ONNX path overriding the auto-downloaded weights.
        reid_thr: Appearance similarity floor for re-attach in [-1, 1]; ``None``
            uses the per-backend default (see ``reid_thr_for``).
```

In `__post_init__`, replace the `reid_thr` validation block with:

```python
        if self.reid_backend not in REID_BACKENDS:
            raise ValueError(
                f"reid_backend must be one of {REID_BACKENDS}, got {self.reid_backend!r}")
        if self.reid_thr is not None and not -1.0 <= self.reid_thr <= 1.0:
            raise ValueError(f"reid_thr must be in [-1, 1], got {self.reid_thr}")
```

In `_build_parser`, replace the `--reid-thr` argument and add the two new
arguments (after `--reid-seconds`):

```python
    parser.add_argument(
        "--reid-backend", choices=REID_BACKENDS, default="osnet-x1",
        help="Appearance backend for re-id: learned OSNet embedding (osnet-x1 = "
             "accurate, osnet-x025 = fast) or the HSV color histogram "
             "(default: osnet-x1).",
    )
    parser.add_argument(
        "--reid-weights", default=None, metavar="PATH",
        help="Local OSNet ONNX file to use instead of the auto-downloaded weights.",
    )
    parser.add_argument(
        "--reid-thr", type=float, default=None,
        help="Appearance similarity floor for re-attach, in [-1,1]. Default "
             "resolves per backend (osnet: 0.5, histogram: 0.6).",
    )
```

In `from_args`, add after `reid_seconds=args.reid_seconds,` (the existing
`reid_thr=args.reid_thr,` line stays — `args.reid_thr` now defaults to `None`):

```python
        reid_backend=args.reid_backend,
        reid_weights=args.reid_weights,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all config tests, including the updated/added reid ones).

- [ ] **Step 5: Commit**

```bash
git add src/storepose/config.py tests/test_config.py
git commit -m "feat: --reid-backend / --reid-weights / per-backend --reid-thr default"
```

---

## Task 5: Runner — select and construct the backend with fallback

**Files:**
- Modify: `src/storepose/runner.py:27` (imports), `:44-58` (`build_tracker`)

**Interfaces:**
- Consumes: `OsnetAppearance`, `reid_zoo.resolve`, `reid_thr_for`, `REID_BACKENDS`, the tracker's `appearance`/`reid_thr` params.
- Produces: `build_tracker` builds the configured backend, falling back to the histogram (with a stderr warning) if OSNet weights cannot load, and passes the resolved `reid_thr` for whichever backend was actually built.

- [ ] **Step 1: Add imports**

At the top of `src/storepose/runner.py`, near `from .tracking.appearance import HsvHistogramAppearance` (line 27), add:

```python
import sys

from .config import reid_thr_for
from .tracking.appearance import HsvHistogramAppearance
from .tracking.osnet import OsnetAppearance
from .tracking import reid_zoo
```

(Keep the existing `MultiObjectTracker` import. If `sys` is already imported at
the top of the module, do not duplicate it.)

- [ ] **Step 2: Replace `build_tracker`**

Replace the body of `build_tracker` (lines 44-58) with:

```python
def _build_appearance(config: AppConfig) -> tuple[object | None, str]:
    """Return ``(appearance_model, effective_backend)``.

    OSNet is default-on; if its weights cannot be fetched or loaded we warn and
    fall back to the histogram so a run never crashes offline.
    """
    if not config.reid:
        return None, config.reid_backend
    if config.reid_backend == "histogram":
        return HsvHistogramAppearance(kpt_thr=config.kpt_thr), "histogram"
    try:
        weights = config.reid_weights or str(reid_zoo.resolve(config.reid_backend))
        return OsnetAppearance(weights, device=config.device), config.reid_backend
    except Exception as exc:  # missing weights, bad onnx, no provider, etc.
        print(
            f"warning: OSNet re-id ({config.reid_backend}) unavailable ({exc}); "
            f"falling back to the color-histogram backend.",
            file=sys.stderr,
        )
        return HsvHistogramAppearance(kpt_thr=config.kpt_thr), "histogram"


def build_tracker(config: AppConfig, base_fps: float) -> "MultiObjectTracker":
    """Construct the multi-object tracker from config (shared by run + calibrate)."""
    max_age = max(1, round(config.hold_seconds * base_fps))
    appearance, backend = _build_appearance(config)
    reid_max_age = max(1, round(config.reid_seconds * base_fps))
    return MultiObjectTracker(
        max_age=max_age, min_hits=config.min_hits,
        iou_thr=config.iou_thr, max_overlap=config.max_overlap,
        smooth=config.smooth,
        min_cutoff=config.smooth_cutoff, beta=config.smooth_beta,
        appearance=appearance, reid=config.reid,
        reid_max_age=reid_max_age, reid_thr=reid_thr_for(backend, config.reid_thr),
    )
```

- [ ] **Step 3: Verify the suite still passes and the tracker builds**

Run: `uv run pytest -q`
Expected: PASS (full suite).

Run: `uv run python -c "from storepose.config import AppConfig; from storepose.runner import build_tracker; t = build_tracker(AppConfig(reid_backend='histogram'), 30.0); print('thr', t._reid_thr)"`
Expected: prints `thr 0.6` (histogram path needs no download).

- [ ] **Step 4: Commit**

```bash
git add src/storepose/runner.py
git commit -m "feat: runner builds the configured re-id backend with histogram fallback"
```

---

## Task 6: Launcher TUI — selectable re-id model column

**Files:**
- Modify: `src/storepose/launcher_core.py`
- Test: `tests/test_launcher_core.py`

**Interfaces:**
- Consumes: the `--reid-backend` / `--no-reid` CLI from Task 4.
- Produces: `Column.REID`, `REID_CYCLE = ("osnet-x1", "osnet-x025", "histogram", "off")`, `ColumnState.reid` (default `"osnet-x1"`); `toggle` cycles it; `build_run` emits `--reid-backend <v>` (or `--no-reid` when `off`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_launcher_core.py
from storepose.launcher_core import Column, REID_CYCLE, build_run, default_state, toggle


def _any_view():
    from storepose.launcher_core import View
    from pathlib import Path
    return View("v", Path("v.sh"), has_calib=False, has_alt=False)


def test_reid_column_cycles():
    view = _any_view()
    state = default_state(view)
    assert state.reid == "osnet-x1"
    seen = [state.reid]
    for _ in range(len(REID_CYCLE)):
        state = toggle(view, state, Column.REID)
        seen.append(state.reid)
    assert seen[1:] == ["osnet-x025", "histogram", "off", "osnet-x1"]


def test_build_run_emits_reid_backend():
    view = _any_view()
    state = default_state(view)
    _env, args = build_run(view, state)
    assert "--reid-backend" in args and "osnet-x1" in args


def test_build_run_off_emits_no_reid():
    view = _any_view()
    state = toggle(view, toggle(view, toggle(view, default_state(view), Column.REID),
                                Column.REID), Column.REID)  # -> "off"
    assert state.reid == "off"
    _env, args = build_run(view, state)
    assert "--no-reid" in args and "--reid-backend" not in args
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_launcher_core.py -k reid -v`
Expected: FAIL — `AttributeError: REID_CYCLE` / `Column has no member REID`.

- [ ] **Step 3: Implement the column**

In `src/storepose/launcher_core.py`:

Add the cycle constant after `STRATEGY_CYCLE` (line ~19):

```python
# reid column cycles through the OSNet sizes, the histogram, then off.
REID_CYCLE = ("osnet-x1", "osnet-x025", "histogram", "off")
```

Add `REID` to the `Column` enum (after `STRATEGY = 7`):

```python
    REID = 8
```

Add it to `COLUMNS` and `COLUMN_LABELS`:

```python
COLUMNS = (Column.DASHBOARD, Column.DEBUG, Column.CONF, Column.SAVE,
           Column.BLUR, Column.ALT, Column.CALIB, Column.STRATEGY, Column.REID)
COLUMN_LABELS = {
    Column.DASHBOARD: "dash",
    Column.DEBUG: "debug",
    Column.CONF: "conf",
    Column.SAVE: "save",
    Column.BLUR: "blur",
    Column.ALT: "alt",
    Column.CALIB: "calib",
    Column.STRATEGY: "strategy",
    Column.REID: "reid",
}
```

Add the `reid` field to `ColumnState` (after `strategy: str = "auto"`):

```python
    reid: str = "osnet-x1"
```

In `toggle`, handle `Column.REID` before the `if not view.has_calib:` guard (so
it is always available, like dashboard/debug):

```python
    if column == Column.REID:
        i = REID_CYCLE.index(state.reid)
        return replace(state, reid=REID_CYCLE[(i + 1) % len(REID_CYCLE)])
```

In `build_run`, emit the flag (add before `return env, args`):

```python
    if state.reid == "off":
        args.append("--no-reid")
    else:
        args += ["--reid-backend", state.reid]
```

(Always emitting `--reid-backend` for non-off values keeps the launched command
explicit; `default_state` already sets `reid="osnet-x1"`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_launcher_core.py -v`
Expected: PASS (existing launcher tests + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/storepose/launcher_core.py tests/test_launcher_core.py
git commit -m "feat: launcher TUI reid column selects osnet-x1/osnet-x025/histogram/off"
```

---

## Task 7: Documentation

**Files:**
- Modify: `README.md`, `docs/usage.md`

- [ ] **Step 1: Update the README flag table and tracking section**

In `README.md`, replace the existing `--reid-thr` flag-table row and add two
rows (after `--reid-seconds`):

```markdown
| `--reid-backend` | `osnet-x1` | Re-id appearance backend: `osnet-x1` (accurate), `osnet-x025` (fast), or `histogram`. |
| `--reid-weights` | —       | Local OSNet ONNX file overriding the auto-downloaded weights. |
| `--reid-thr`     | per-backend | Appearance similarity floor for re-attach (osnet 0.5, histogram 0.6). |
```

In the "Tracking & smoothing" section, replace the appearance-re-id sentence
with:

```markdown
By default a learned OSNet ReID embedding re-attaches a returning person to their
original id within `--reid-seconds` -- it holds up across similar clothing,
lighting shifts, and a full exit and re-entry anywhere in frame. Each track keeps
a small gallery of recent embeddings (nearest-neighbor match). Weights download
once on first run; choose accuracy vs speed with `--reid-backend osnet-x1` /
`osnet-x025`, fall back to the color histogram with `--reid-backend histogram`,
or disable re-id with `--no-reid`.
```

- [ ] **Step 2: Update usage.md**

In `docs/usage.md` Section 2 "Useful run flags" table, replace the `--reid-*`
rows (or add after `--no-reid`) with:

```markdown
| `--reid-backend` | `osnet-x1` | Re-id appearance backend: `osnet-x1`, `osnet-x025`, or `histogram`. |
| `--reid-weights` | — | Local OSNet ONNX file overriding the auto-downloaded weights. |
| `--reid-thr` | per-backend | Appearance similarity floor (osnet 0.5, histogram 0.6). |
```

Add a line in the tracking discussion noting the launcher's `reid` column cycles
`osnet-x1 -> osnet-x025 -> histogram -> off` per view.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/usage.md
git commit -m "docs: document OSNet re-id backend, weights, and TUI selector"
```

---

## Task 8: Full-suite verification + live A/B smoke

- [ ] **Step 1: Run the whole suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Live smoke on each backend (single lines, no backslashes)**

Run: `uv run python main.py --source /tmp/sco_clip.mp4 --zone "zones/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.json" --reid-backend osnet-x1 --save /tmp/reid_x1.mp4`
Expected: runs to completion; weights download on first run; `Done. Wrote /tmp/reid_x1.mp4`.

Run: `uv run python main.py --source /tmp/sco_clip.mp4 --zone "zones/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.json" --reid-backend osnet-x025 --save /tmp/reid_x025.mp4`
Expected: runs to completion; faster per-frame than x1.

Run: `uv run python main.py --source /tmp/sco_clip.mp4 --zone "zones/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.json" --reid-backend histogram --save /tmp/reid_hist.mp4`
Expected: runs to completion (no download).

(If `/tmp/sco_clip.mp4` is absent, trim ~200 frames from a video under `videos/`
with OpenCV first, or point `--source` at a full clip.)

- [ ] **Step 3: Confirm the offline fallback**

Temporarily force a download failure to confirm the warn-and-fall-back path:

Run: `uv run python main.py --source /tmp/sco_clip.mp4 --reid-weights /tmp/does_not_exist.onnx --no-dashboard --save /tmp/reid_fallback.mp4`
Expected: prints the `warning: OSNet re-id ... falling back to the color-histogram backend.` line and still completes.

- [ ] **Step 4: A/B fragmentation check (optional)**

Run the osnet vs histogram commands above with `--wait-log /tmp/w_osnet.csv` and
`--wait-log /tmp/w_hist.csv` and compare row counts on footage with crossings /
brief exits — OSNet should produce fewer, longer waits (less id fragmentation).

---

## Self-Review Notes

- **Spec coverage:** seam refactor with `extract_batch`/`new_memory`/`update_memory`/`score` (Task 1, 3) [done]; `OsnetAppearance` full-body crop + ImageNet preprocess + L2-normalized embedding + gallery + max-cosine (Task 2) [done]; weight resolver with cache + checksum + `--reid-weights` override (Task 2, 4) [done]; graceful histogram fallback (Task 5) [done]; split spatial gate — gallery appearance-only at `reid_thr + 0.05`, active keeps the gate (Task 3) [done]; per-backend `reid_thr` default via `reid_thr_for` (Task 4, 5) [done]; config `--reid-backend`/`--reid-weights`/`--reid-thr` (Task 4) [done]; launcher `Column.REID` cycle + `build_run` (Task 6) [done]; tests incl. cross-exit re-entry + stricter-gallery-threshold + osnet unit + config + launcher (Tasks 2,3,4,6) [done]; analyzer untouched (revive-id continues the wait timer) [done]; docs (Task 7) [done].
- **Signature consistency:** `extract(frame, box, keypoints, scores)`, `extract_batch(frame, boxes, keypoints, scores)`, `new_memory(desc)`, `update_memory(mem, desc)`, `score(mem, desc)`; `Track(..., appearance_mem=None, det_score=None)`, `Track.update/reactivate(..., appearance_mem=None, det_score=None)`; `OsnetAppearance(weights_path, device)`; `reid_zoo.resolve(backend) -> Path`; `reid_thr_for(backend, override) -> float`; `MultiObjectTracker(..., appearance, reid, reid_max_age, reid_thr)` unchanged.
- **Backward compatibility:** tracker still defaults `reid=False`/`frame=None`; `appearance_mem` stays `None` when re-id is off, so all no-frame tracker/track tests keep exact behavior. The `_GALLERY_MARGIN` and `K` are internal constants, not config.
- **One discovered value:** the OSNet ONNX URLs+sha (Task 2, Step 6) are pinned during implementation and verified by a real download; `--reid-weights` + histogram fallback de-risk it.
