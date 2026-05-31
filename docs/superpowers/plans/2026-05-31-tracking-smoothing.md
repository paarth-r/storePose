# Per-Person Tracking, Permanence & Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each detected person a stable ID whose bounding box persists through brief occlusions, and reduce jitter in boxes and pose skeletons.

**Architecture:** A SORT-style tracking layer sits between the existing `PosePipeline` and drawing. Each track owns a Kalman box filter (smoothing + coasting) and per-keypoint One-Euro filters. The tracker assigns stable IDs via IoU/Hungarian association, coasts unmatched tracks for a hold window, and emits `TrackedPerson`s the drawer renders with stable per-ID colors.

**Tech Stack:** Python 3.12, numpy, scipy (`linear_sum_assignment`), OpenCV, rtmlib. Managed by `uv`.

---

## File structure

```
src/storepose/tracking/
  __init__.py     package marker
  types.py        TrackedPerson dataclass
  assignment.py   iou(), iou_matrix(), match()
  kalman.py       KalmanBoxTracker (SORT constant-velocity)
  smoothing.py    OneEuroFilter, KeypointSmoother
  track.py        Track (id, kalman, lifecycle, color, smoothers)
  tracker.py      MultiObjectTracker (predict → associate → lifecycle)
```

Modified: `pyproject.toml` (scipy), `src/storepose/config.py` (flags),
`src/storepose/drawing.py` (`annotate_tracked`), `src/storepose/runner.py`
(wire tracker), `README.md`.

Tests: `tests/tracking/test_assignment.py`, `test_kalman.py`,
`test_smoothing.py`, `test_track.py`, `test_tracker.py`; plus additions to
`tests/test_drawing.py`, `tests/test_config.py`.

---

## Task 1: Add scipy dependency

**Files:**
- Modify: `pyproject.toml` (dependencies list)

- [ ] **Step 1: Add scipy to dependencies**

In `pyproject.toml`, change the `dependencies` array to include scipy:

```toml
dependencies = [
    "rtmlib>=0.0.13",
    "onnxruntime>=1.17",
    "opencv-python>=4.9",
    "numpy>=1.26,<2.3",
    "scipy>=1.11",
]
```

- [ ] **Step 2: Sync and verify import**

Run: `uv sync && uv run python -c "from scipy.optimize import linear_sum_assignment; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add scipy for Hungarian assignment"
```

---

## Task 2: TrackedPerson type

**Files:**
- Create: `src/storepose/tracking/__init__.py`
- Create: `src/storepose/tracking/types.py`
- Test: `tests/tracking/__init__.py`, `tests/tracking/test_types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracking/__init__.py` (empty) and `tests/tracking/test_types.py`:

```python
import numpy as np

from storepose.tracking.types import TrackedPerson


def test_tracked_person_fields():
    p = TrackedPerson(
        id=3,
        box=np.array([0, 0, 10, 20], float),
        keypoints=np.zeros((17, 2), float),
        scores=np.ones(17, float),
        coasting=False,
        color=(0, 255, 0),
    )
    assert p.id == 3
    assert p.coasting is False
    assert p.color == (0, 255, 0)
    assert p.keypoints.shape == (17, 2)


def test_tracked_person_coasting_has_no_pose():
    p = TrackedPerson(
        id=1, box=np.array([0, 0, 5, 5], float),
        keypoints=None, scores=None, coasting=True, color=(255, 0, 0),
    )
    assert p.keypoints is None
    assert p.scores is None
    assert p.coasting is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tracking/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: storepose.tracking`

- [ ] **Step 3: Write minimal implementation**

Create `src/storepose/tracking/__init__.py` (empty file).

Create `src/storepose/tracking/types.py`:

```python
"""Public result type for the tracking layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedPerson:
    """A confirmed track emitted for one frame.

    Attributes:
        id: Stable integer identity for this person.
        box: Smoothed ``(4,)`` xyxy bounding box (Kalman filter output).
        keypoints: Smoothed ``(17, 2)`` keypoints, or ``None`` while coasting.
        scores: ``(17,)`` per-keypoint confidences, or ``None`` while coasting.
        coasting: True when the box is predicted (no detection this frame).
        color: Stable BGR color for this id.
    """

    id: int
    box: np.ndarray
    keypoints: np.ndarray | None
    scores: np.ndarray | None
    coasting: bool
    color: tuple[int, int, int]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tracking/test_types.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/tracking/__init__.py src/storepose/tracking/types.py tests/tracking/__init__.py tests/tracking/test_types.py
git commit -m "feat: add TrackedPerson type for tracking layer"
```

---

## Task 3: IoU assignment

**Files:**
- Create: `src/storepose/tracking/assignment.py`
- Test: `tests/tracking/test_assignment.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracking/test_assignment.py`:

```python
import numpy as np

from storepose.tracking.assignment import iou, iou_matrix, match


def test_iou_identical_is_one():
    b = np.array([0, 0, 10, 10], float)
    assert iou(b, b) == 1.0


def test_iou_disjoint_is_zero():
    assert iou(np.array([0, 0, 10, 10], float),
               np.array([20, 20, 30, 30], float)) == 0.0


def test_iou_half_overlap():
    a = np.array([0, 0, 10, 10], float)
    b = np.array([5, 0, 15, 10], float)  # overlap 50x100? inter=50, union=150
    assert iou(a, b) == 0.5 / 1.5  # inter 50, union 150 -> 1/3


def test_iou_matrix_shape_and_values():
    dets = [np.array([0, 0, 10, 10], float), np.array([100, 100, 110, 110], float)]
    trks = [np.array([0, 0, 10, 10], float)]
    m = iou_matrix(dets, trks)
    assert m.shape == (2, 1)
    assert m[0, 0] == 1.0
    assert m[1, 0] == 0.0


def test_match_pairs_overlapping():
    dets = [np.array([0, 0, 10, 10], float), np.array([100, 0, 110, 10], float)]
    trks = [np.array([101, 0, 111, 10], float), np.array([1, 1, 11, 11], float)]
    matches, ud, ut = match(dets, trks, iou_thr=0.3)
    # det0 -> trk1, det1 -> trk0
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert ud == [] and ut == []


def test_match_gates_low_iou():
    dets = [np.array([0, 0, 10, 10], float)]
    trks = [np.array([50, 50, 60, 60], float)]
    matches, ud, ut = match(dets, trks, iou_thr=0.3)
    assert matches == []
    assert ud == [0] and ut == [0]


def test_match_empty_inputs():
    assert match([], [], 0.3) == ([], [], [])
    assert match([np.array([0, 0, 1, 1], float)], [], 0.3) == ([], [0], [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tracking/test_assignment.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/storepose/tracking/assignment.py`:

```python
"""IoU-based detection-to-track association (Hungarian)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two ``xyxy`` boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(dets: list[np.ndarray], tracks: list[np.ndarray]) -> np.ndarray:
    """``(len(dets), len(tracks))`` matrix of pairwise IoU."""
    m = np.zeros((len(dets), len(tracks)), dtype=np.float32)
    for i, d in enumerate(dets):
        for j, t in enumerate(tracks):
            m[i, j] = iou(d, t)
    return m


def match(
    dets: list[np.ndarray], tracks: list[np.ndarray], iou_thr: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Associate detections to tracks by maximum IoU, gated by ``iou_thr``.

    Returns ``(matches, unmatched_dets, unmatched_tracks)`` where ``matches`` is
    a list of ``(det_index, track_index)`` pairs.
    """
    if len(dets) == 0 or len(tracks) == 0:
        return [], list(range(len(dets))), list(range(len(tracks)))

    m = iou_matrix(dets, tracks)
    rows, cols = linear_sum_assignment(-m)  # maximize total IoU

    matches: list[tuple[int, int]] = []
    unmatched_dets = set(range(len(dets)))
    unmatched_tracks = set(range(len(tracks)))
    for r, c in zip(rows, cols):
        if m[r, c] >= iou_thr:
            matches.append((int(r), int(c)))
            unmatched_dets.discard(int(r))
            unmatched_tracks.discard(int(c))
    return matches, sorted(unmatched_dets), sorted(unmatched_tracks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tracking/test_assignment.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/tracking/assignment.py tests/tracking/test_assignment.py
git commit -m "feat: add IoU/Hungarian assignment for tracking"
```

---

## Task 4: Kalman box filter

**Files:**
- Create: `src/storepose/tracking/kalman.py`
- Test: `tests/tracking/test_kalman.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracking/test_kalman.py`:

```python
import numpy as np
import pytest

from storepose.tracking.kalman import KalmanBoxTracker


def _cx(box):
    return (box[0] + box[2]) / 2


def test_init_box_roundtrips():
    box = np.array([10, 20, 30, 60], float)
    t = KalmanBoxTracker(box)
    np.testing.assert_allclose(t.box, box, atol=1e-6)


def test_predict_static_keeps_center():
    t = KalmanBoxTracker(np.array([0, 0, 10, 10], float))
    t.predict()
    assert _cx(t.box) == pytest.approx(5.0, abs=1e-6)


def test_update_moves_toward_measurement():
    t = KalmanBoxTracker(np.array([0, 0, 10, 10], float))
    for _ in range(5):
        t.update(np.array([10, 0, 20, 10], float))  # shifted +10 in x
    assert _cx(t.box) > 5.0


def test_predict_advances_by_velocity():
    t = KalmanBoxTracker(np.array([0, 0, 10, 10], float))
    for i in range(1, 6):
        t.update(np.array([10 * i, 0, 10 * i + 10, 10], float))  # +10/frame
    before = _cx(t.box)
    t.predict()
    assert _cx(t.box) > before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tracking/test_kalman.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/storepose/tracking/kalman.py`:

```python
"""SORT-style constant-velocity Kalman filter for a single box."""

from __future__ import annotations

import numpy as np


def _box_to_z(box: np.ndarray) -> np.ndarray:
    """``xyxy`` -> measurement ``[cx, cy, area, aspect]`` as a ``(4, 1)``."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w / 2.0
    cy = box[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 0.0
    return np.array([[cx], [cy], [s], [r]], dtype=float)


def _x_to_box(x: np.ndarray) -> np.ndarray:
    """State -> ``xyxy`` box ``(4,)``."""
    cx, cy, s, r = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    w = np.sqrt(max(s * r, 1e-6))
    h = s / w if w > 0 else 0.0
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], float)


class KalmanBoxTracker:
    """Tracks one box with state ``[cx, cy, area, aspect, vx, vy, v_area]``."""

    def __init__(self, box: np.ndarray):
        self.F = np.eye(7)
        for i in range(3):  # position components gain a velocity term
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 7))
        for i in range(4):
            self.H[i, i] = 1.0

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # high uncertainty on unobserved velocities
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[6, 6] *= 0.01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0  # area/aspect are noisier measurements

        self.x = np.zeros((7, 1))
        self.x[:4] = _box_to_z(box)

    def predict(self) -> np.ndarray:
        """Advance the state by one step; returns the predicted box."""
        if (self.x[6] + self.x[2])[0] <= 0:  # keep area non-negative
            self.x[6] *= 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box

    def update(self, box: np.ndarray) -> np.ndarray:
        """Correct the state with a measured box; returns the filtered box."""
        z = _box_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        return self.box

    @property
    def box(self) -> np.ndarray:
        return _x_to_box(self.x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tracking/test_kalman.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/tracking/kalman.py tests/tracking/test_kalman.py
git commit -m "feat: add SORT Kalman box filter"
```

---

## Task 5: One-Euro smoothing

**Files:**
- Create: `src/storepose/tracking/smoothing.py`
- Test: `tests/tracking/test_smoothing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracking/test_smoothing.py`:

```python
import numpy as np

from storepose.tracking.smoothing import KeypointSmoother, OneEuroFilter


def test_first_sample_passes_through():
    f = OneEuroFilter()
    assert f(5.0, dt=1 / 30) == 5.0


def test_constant_signal_stays_constant():
    f = OneEuroFilter()
    f(3.0, 1 / 30)
    for _ in range(20):
        out = f(3.0, 1 / 30)
    assert abs(out - 3.0) < 1e-6


def test_reduces_variance_on_noisy_constant():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    noisy = 10.0 + rng.normal(0, 1.0, size=200)
    out = np.array([f(float(v), 1 / 30) for v in noisy])
    assert out[50:].std() < noisy[50:].std()


def test_tracks_a_ramp():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.1)
    last = 0.0
    for i in range(100):
        last = f(float(i), 1 / 30)
    assert last > 80.0  # follows the ramp upward, some lag allowed


def test_keypoint_smoother_shape_and_smoothing():
    rng = np.random.default_rng(1)
    s = KeypointSmoother(num_keypoints=17, min_cutoff=0.5, beta=0.0)
    base = np.full((17, 2), 100.0)
    outs = []
    for _ in range(60):
        outs.append(s.update(base + rng.normal(0, 2.0, (17, 2)), 1 / 30))
    outs = np.array(outs)
    assert outs[0].shape == (17, 2)
    assert outs[20:, 0, 0].std() < 2.0  # smoothed below input noise std
    assert s.last is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tracking/test_smoothing.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/storepose/tracking/smoothing.py`:

```python
"""One-Euro filtering for low-jitter, low-lag signal smoothing."""

from __future__ import annotations

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Adaptive low-pass filter (Casiez et al.).

    Cutoff rises with the signal's speed, so it is smooth when still and
    responsive when moving. State is per-instance; use one per scalar channel.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if dt <= 0:
            dt = 1e-6
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class KeypointSmoother:
    """One-Euro filter per keypoint (independent x and y channels)."""

    def __init__(self, num_keypoints: int = 17, min_cutoff: float = 1.0, beta: float = 0.007):
        self._fx = [OneEuroFilter(min_cutoff, beta) for _ in range(num_keypoints)]
        self._fy = [OneEuroFilter(min_cutoff, beta) for _ in range(num_keypoints)]
        self._last: np.ndarray | None = None

    def update(self, keypoints: np.ndarray, dt: float) -> np.ndarray:
        """Return a smoothed copy of ``(num_keypoints, 2)`` keypoints."""
        out = np.empty_like(keypoints, dtype=float)
        for i in range(len(keypoints)):
            out[i, 0] = self._fx[i](float(keypoints[i, 0]), dt)
            out[i, 1] = self._fy[i](float(keypoints[i, 1]), dt)
        self._last = out
        return out

    @property
    def last(self) -> np.ndarray | None:
        return self._last
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tracking/test_smoothing.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/tracking/smoothing.py tests/tracking/test_smoothing.py
git commit -m "feat: add One-Euro keypoint smoothing"
```

---

## Task 6: Track (lifecycle unit + color)

**Files:**
- Create: `src/storepose/tracking/track.py`
- Test: `tests/tracking/test_track.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracking/test_track.py`:

```python
import numpy as np

from storepose.tracking.track import Track, color_for


def _box():
    return np.array([0, 0, 10, 20], float)


def _kpts():
    return np.full((17, 2), 5.0)


def test_color_is_stable_and_bgr_tuple():
    c = color_for(2)
    assert color_for(2) == c
    assert len(c) == 3


def test_new_track_with_min_hits_one_is_confirmed():
    t = Track(0, _box(), _kpts(), np.ones(17), dt=1 / 30,
              min_hits=1, smooth=False, min_cutoff=1.0, beta=0.0)
    assert t.confirmed is True
    assert t.coasting is False
    assert t.keypoints is not None


def test_track_confirms_after_min_hits():
    t = Track(0, _box(), _kpts(), np.ones(17), dt=1 / 30,
              min_hits=3, smooth=False, min_cutoff=1.0, beta=0.0)
    assert t.confirmed is False
    t.predict(); t.update(_box(), _kpts(), np.ones(17), 1 / 30)
    assert t.confirmed is False
    t.predict(); t.update(_box(), _kpts(), np.ones(17), 1 / 30)
    assert t.confirmed is True


def test_predict_marks_coasting():
    t = Track(0, _box(), _kpts(), np.ones(17), dt=1 / 30,
              min_hits=1, smooth=False, min_cutoff=1.0, beta=0.0)
    t.predict()
    assert t.coasting is True
    assert t.time_since_update == 1
    t.update(_box(), _kpts(), np.ones(17), 1 / 30)
    assert t.coasting is False
    assert t.time_since_update == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tracking/test_track.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/storepose/tracking/track.py`:

```python
"""A single tracked person: identity, motion, lifecycle, and smoothing."""

from __future__ import annotations

import numpy as np

from .kalman import KalmanBoxTracker
from .smoothing import KeypointSmoother

# Distinct BGR colors cycled by track id.
_PALETTE: list[tuple[int, int, int]] = [
    (56, 56, 255), (56, 255, 56), (255, 56, 56), (56, 255, 255),
    (255, 56, 255), (255, 255, 56), (56, 153, 255), (255, 153, 56),
    (153, 56, 255), (56, 255, 153),
]


def color_for(track_id: int) -> tuple[int, int, int]:
    """Stable BGR color for a track id."""
    return _PALETTE[track_id % len(_PALETTE)]


class Track:
    """One person's track. Created from a detection, updated or coasted per frame."""

    def __init__(
        self,
        track_id: int,
        box: np.ndarray,
        keypoints: np.ndarray | None,
        scores: np.ndarray | None,
        dt: float,
        *,
        min_hits: int,
        smooth: bool,
        min_cutoff: float,
        beta: float,
    ):
        self.id = track_id
        self.kalman = KalmanBoxTracker(box)
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
        self._ingest_pose(keypoints, scores, dt)

    def _ingest_pose(self, keypoints, scores, dt) -> None:
        self.scores = scores
        if keypoints is None:
            return
        if self._smoother is not None:
            self.keypoints = self._smoother.update(keypoints, dt)
        else:
            self.keypoints = np.asarray(keypoints, float)

    def predict(self) -> None:
        """Advance motion; mark the track as coasting for this frame."""
        self.kalman.predict()
        self.time_since_update += 1

    def update(self, box, keypoints, scores, dt) -> None:
        """Correct with a matched detection."""
        self.kalman.update(box)
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self.min_hits:
            self.confirmed = True
        self._ingest_pose(keypoints, scores, dt)

    @property
    def box(self) -> np.ndarray:
        return self.kalman.box

    @property
    def coasting(self) -> bool:
        return self.time_since_update > 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tracking/test_track.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/tracking/track.py tests/tracking/test_track.py
git commit -m "feat: add Track lifecycle unit with color and smoothing"
```

---

## Task 7: MultiObjectTracker

**Files:**
- Create: `src/storepose/tracking/tracker.py`
- Test: `tests/tracking/test_tracker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracking/test_tracker.py`:

```python
import numpy as np

from storepose.pipeline import FrameResult
from storepose.tracking.tracker import MultiObjectTracker


def make_result(boxes, kpts=None):
    boxes = np.array(boxes, float).reshape(-1, 4)
    n = len(boxes)
    if kpts is None:
        kpts = np.zeros((n, 17, 2), float)
    return FrameResult(boxes=boxes, keypoints=kpts, scores=np.ones((n, 17), float))


def test_track_confirms_after_min_hits():
    tr = MultiObjectTracker(max_age=10, min_hits=3, iou_thr=0.3, smooth=False)
    box = [10, 10, 50, 90]
    assert tr.update(make_result([box]), 1 / 30) == []
    assert tr.update(make_result([box]), 1 / 30) == []
    out = tr.update(make_result([box]), 1 / 30)
    assert len(out) == 1 and out[0].id == 0 and out[0].coasting is False


def test_stable_id_across_frames():
    tr = MultiObjectTracker(max_age=10, min_hits=1, iou_thr=0.3, smooth=False)
    box = [10, 10, 50, 90]
    a = tr.update(make_result([box]), 1 / 30)
    b = tr.update(make_result([box]), 1 / 30)
    assert a[0].id == b[0].id == 0


def test_two_detections_get_distinct_ids():
    tr = MultiObjectTracker(max_age=10, min_hits=1, iou_thr=0.3, smooth=False)
    out = tr.update(make_result([[0, 0, 20, 40], [100, 0, 120, 40]]), 1 / 30)
    assert {p.id for p in out} == {0, 1}


def test_coasts_then_dies():
    tr = MultiObjectTracker(max_age=3, min_hits=1, iou_thr=0.3, smooth=False)
    box = [10, 10, 50, 90]
    assert tr.update(make_result([box]), 1 / 30)[0].coasting is False
    o1 = tr.update(make_result([]), 1 / 30)
    assert len(o1) == 1 and o1[0].coasting is True and o1[0].keypoints is None
    tr.update(make_result([]), 1 / 30)
    assert len(tr.update(make_result([]), 1 / 30)) == 1  # t=3, still within max_age
    assert tr.update(make_result([]), 1 / 30) == []      # t=4, culled


def test_reentry_gets_new_id():
    tr = MultiObjectTracker(max_age=1, min_hits=1, iou_thr=0.3, smooth=False)
    box = [10, 10, 50, 90]
    tr.update(make_result([box]), 1 / 30)   # id 0
    tr.update(make_result([]), 1 / 30)      # coast t=1
    tr.update(make_result([]), 1 / 30)      # t=2 > max_age -> culled
    out = tr.update(make_result([box]), 1 / 30)
    assert len(out) == 1 and out[0].id == 1


def test_tentative_track_dropped_on_miss():
    tr = MultiObjectTracker(max_age=10, min_hits=3, iou_thr=0.3, smooth=False)
    tr.update(make_result([[10, 10, 50, 90]]), 1 / 30)  # tentative (hits=1)
    tr.update(make_result([]), 1 / 30)                  # missed before confirm
    # a fresh detection should now be id 1 (the tentative track is gone)
    out = tr.update(make_result([[10, 10, 50, 90]]), 1 / 30)
    tr.update(make_result([[10, 10, 50, 90]]), 1 / 30)
    out = tr.update(make_result([[10, 10, 50, 90]]), 1 / 30)
    assert out and out[0].id == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tracking/test_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/storepose/tracking/tracker.py`:

```python
"""SORT-style multi-object tracker producing stable, coasted tracks."""

from __future__ import annotations

from .assignment import match
from .track import Track
from .types import TrackedPerson


class MultiObjectTracker:
    """Associates per-frame detections to persistent, smoothed tracks.

    Each ``update`` predicts existing tracks, associates them to the frame's
    detections by IoU, updates matches, coasts the rest (up to ``max_age``), and
    emits confirmed tracks as :class:`TrackedPerson`.
    """

    def __init__(
        self,
        max_age: int = 45,
        min_hits: int = 3,
        iou_thr: float = 0.3,
        smooth: bool = True,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thr = iou_thr
        self.smooth = smooth
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._tracks: list[Track] = []
        self._next_id = 0

    def update(self, result, dt: float) -> list[TrackedPerson]:
        boxes = result.boxes
        keypoints = result.keypoints
        scores = result.scores

        # 1. predict existing tracks forward
        for t in self._tracks:
            t.predict()

        # 2. associate detections to predicted tracks
        det_boxes = [boxes[i] for i in range(len(boxes))]
        track_boxes = [t.box for t in self._tracks]
        matches, unmatched_dets, _ = match(det_boxes, track_boxes, self.iou_thr)

        # 3. update matched tracks
        for d, tr in matches:
            self._tracks[tr].update(boxes[d], keypoints[d], scores[d], dt)

        # 4. spawn tracks for unmatched detections
        for d in unmatched_dets:
            self._tracks.append(
                Track(
                    self._next_id, boxes[d], keypoints[d], scores[d], dt,
                    min_hits=self.min_hits, smooth=self.smooth,
                    min_cutoff=self.min_cutoff, beta=self.beta,
                )
            )
            self._next_id += 1

        # 5. cull dead tracks (aged out, or tentative + missed)
        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= self.max_age
            and not (t.time_since_update >= 1 and not t.confirmed)
        ]

        # 6. emit confirmed tracks
        people: list[TrackedPerson] = []
        for t in self._tracks:
            if not t.confirmed:
                continue
            coasting = t.coasting
            people.append(
                TrackedPerson(
                    id=t.id,
                    box=t.box,
                    keypoints=None if coasting else t.keypoints,
                    scores=None if coasting else t.scores,
                    coasting=coasting,
                    color=t.color,
                )
            )
        return people
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tracking/test_tracker.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/tracking/tracker.py tests/tracking/test_tracker.py
git commit -m "feat: add SORT-style MultiObjectTracker"
```

---

## Task 8: Draw tracked people

**Files:**
- Modify: `src/storepose/drawing.py` (add `annotate_tracked`)
- Test: `tests/test_drawing.py` (append tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_drawing.py`:

```python
from storepose.drawing import annotate_tracked
from storepose.tracking.types import TrackedPerson


def test_annotate_tracked_draws_box_and_skeleton():
    frame = _blank()
    p = TrackedPerson(
        id=2, box=np.array([20, 20, 100, 100], float),
        keypoints=np.full((NUM_KEYPOINTS, 2), 50.0), scores=np.ones(NUM_KEYPOINTS),
        coasting=False, color=(0, 255, 0),
    )
    out = annotate_tracked(frame, [p], AppConfig(), fps=30.0)
    assert out.shape == frame.shape
    assert (out[:, :, 1] > 0).sum() > 0
    assert np.array_equal(frame, _blank())  # input untouched


def test_annotate_tracked_coasting_has_no_pose_and_no_crash():
    p = TrackedPerson(
        id=1, box=np.array([10, 10, 50, 50], float),
        keypoints=None, scores=None, coasting=True, color=(255, 0, 0),
    )
    out = annotate_tracked(_blank(), [p], AppConfig(), fps=None)
    assert out.shape == (120, 160, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_drawing.py -k tracked -v`
Expected: FAIL with `ImportError: cannot import name 'annotate_tracked'`

- [ ] **Step 3: Write minimal implementation**

In `src/storepose/drawing.py`, add the import of the tracking type at the top
(after the existing `from .pipeline import FrameResult`):

```python
from .tracking.types import TrackedPerson
```

Then append this function to the end of `src/storepose/drawing.py`:

```python
def annotate_tracked(
    frame: np.ndarray,
    people: list[TrackedPerson],
    config: AppConfig,
    fps: float | None = None,
) -> np.ndarray:
    """Draw tracked people: stable-color box + ``ID n`` label, skeleton when
    not coasting. Safe on an empty list and on coasting (poseless) people."""
    canvas = frame.copy()

    for p in people:
        x1, y1, x2, y2 = (int(round(v)) for v in p.box[:4])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), p.color, 2)
        label = f"ID {p.id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), p.color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if p.keypoints is not None and p.scores is not None:
            canvas = draw_skeleton(
                canvas, p.keypoints[None, ...], p.scores[None, ...],
                kpt_thr=config.kpt_thr, radius=3, line_width=2,
            )

    header = f"people: {len(people)}"
    if config.show_fps and fps is not None:
        header += f"   fps: {fps:4.1f}"
    cv2.putText(canvas, header, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                FPS_COLOR, 2, cv2.LINE_AA)
    return canvas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_drawing.py -v`
Expected: PASS (all drawing tests, including the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/drawing.py tests/test_drawing.py
git commit -m "feat: draw tracked people with stable colors and ids"
```

---

## Task 9: Config flags

**Files:**
- Modify: `src/storepose/config.py`
- Test: `tests/test_config.py` (append tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_tracking_defaults():
    c = from_args([])
    assert c.track is True
    assert c.hold_seconds == 1.5
    assert c.min_hits == 3
    assert c.iou_thr == 0.3
    assert c.smooth is True
    assert c.smooth_cutoff == 1.0
    assert c.smooth_beta == 0.007


def test_tracking_flags():
    c = from_args([
        "--no-track", "--no-smooth", "--hold-seconds", "2.5",
        "--min-hits", "5", "--iou-thr", "0.4",
        "--smooth-cutoff", "0.5", "--smooth-beta", "0.01",
    ])
    assert c.track is False
    assert c.smooth is False
    assert c.hold_seconds == 2.5
    assert c.min_hits == 5
    assert c.iou_thr == 0.4
    assert c.smooth_cutoff == 0.5
    assert c.smooth_beta == 0.01


@pytest.mark.parametrize("kwargs", [
    {"min_hits": 0},
    {"iou_thr": 1.5},
    {"hold_seconds": -1.0},
    {"smooth_cutoff": 0.0},
    {"smooth_beta": -0.1},
])
def test_tracking_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        AppConfig(**kwargs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -k tracking -v`
Expected: FAIL with `TypeError` / `AttributeError` (fields don't exist yet)

- [ ] **Step 3: Write minimal implementation**

In `src/storepose/config.py`, add fields to the `AppConfig` dataclass after
`save: str | None = None`:

```python
    track: bool = True
    hold_seconds: float = 1.5
    min_hits: int = 3
    iou_thr: float = 0.3
    smooth: bool = True
    smooth_cutoff: float = 1.0
    smooth_beta: float = 0.007
```

Extend `__post_init__` (after the existing threshold loop) with:

```python
        if self.min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {self.min_hits}")
        if not 0.0 <= self.iou_thr <= 1.0:
            raise ValueError(f"iou_thr must be in [0, 1], got {self.iou_thr}")
        if self.hold_seconds < 0:
            raise ValueError(f"hold_seconds must be >= 0, got {self.hold_seconds}")
        if self.smooth_cutoff <= 0:
            raise ValueError(f"smooth_cutoff must be > 0, got {self.smooth_cutoff}")
        if self.smooth_beta < 0:
            raise ValueError(f"smooth_beta must be >= 0, got {self.smooth_beta}")
```

In `_build_parser`, add these arguments before `return parser`:

```python
    parser.add_argument(
        "--no-track", dest="track", action="store_false",
        help="Disable tracking; draw raw per-frame detections.",
    )
    parser.add_argument(
        "--hold-seconds", type=float, default=1.5,
        help="How long a lost person's box keeps coasting (default: 1.5).",
    )
    parser.add_argument(
        "--min-hits", type=int, default=3,
        help="Detections before a track is confirmed/drawn (default: 3).",
    )
    parser.add_argument(
        "--iou-thr", type=float, default=0.3,
        help="Min IoU to associate a detection to a track (default: 0.3).",
    )
    parser.add_argument(
        "--no-smooth", dest="smooth", action="store_false",
        help="Disable One-Euro keypoint smoothing.",
    )
    parser.add_argument(
        "--smooth-cutoff", type=float, default=1.0,
        help="One-Euro min_cutoff; lower = smoother/laggier (default: 1.0).",
    )
    parser.add_argument(
        "--smooth-beta", type=float, default=0.007,
        help="One-Euro beta; higher = more responsive to speed (default: 0.007).",
    )
```

In `from_args`, extend the returned `AppConfig(...)` with the new fields:

```python
        save=args.save,
        track=args.track,
        hold_seconds=args.hold_seconds,
        min_hits=args.min_hits,
        iou_thr=args.iou_thr,
        smooth=args.smooth,
        smooth_cutoff=args.smooth_cutoff,
        smooth_beta=args.smooth_beta,
```

(Keep the existing `save=args.save,` line — the block above shows it for
placement; do not duplicate it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all config tests)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/config.py tests/test_config.py
git commit -m "feat: add tracking and smoothing CLI flags"
```

---

## Task 10: Wire tracker into the runner

**Files:**
- Modify: `src/storepose/runner.py`

- [ ] **Step 1: Rewrite the runner to use the tracker**

Replace the entire contents of `src/storepose/runner.py` with:

```python
"""Realtime loop: capture → process → (track) → annotate → display (+ save)."""

from __future__ import annotations

import time
from contextlib import ExitStack

import cv2

from .config import AppConfig
from .drawing import annotate, annotate_tracked
from .fps import FpsMeter
from .pipeline import PosePipeline
from .tracking.tracker import MultiObjectTracker
from .video_sink import VideoSink
from .video_source import VideoSource

WINDOW_NAME = "storePose"
_QUIT_KEYS = {ord("q"), 27}  # 'q' or Esc
_DEFAULT_FPS = 30.0


class Runner:
    """Owns the realtime display loop and its resources."""

    def __init__(self, config: AppConfig):
        self._config = config

    def run(self) -> None:
        config = self._config
        print(f"Loading models (mode={config.mode}, device={config.device})...")
        pipeline = PosePipeline(config)
        meter = FpsMeter()
        print(f"Models ready. Source: {config.source}. Press 'q' or Esc to quit.")

        try:
            with ExitStack() as stack:
                source = stack.enter_context(VideoSource(config.source))
                sink = None
                if config.save:
                    sink = stack.enter_context(VideoSink(config.save, fps=source.fps))
                    print(f"Saving annotated video to {config.save}")

                tracker = None
                if config.track:
                    base_fps = source.fps or _DEFAULT_FPS
                    max_age = max(1, round(config.hold_seconds * base_fps))
                    tracker = MultiObjectTracker(
                        max_age=max_age, min_hits=config.min_hits,
                        iou_thr=config.iou_thr, smooth=config.smooth,
                        min_cutoff=config.smooth_cutoff, beta=config.smooth_beta,
                    )

                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                prev = None
                for frame in source:
                    now = time.perf_counter()
                    dt = (now - prev) if prev else (1.0 / (source.fps or _DEFAULT_FPS))
                    prev = now

                    result = pipeline.process(frame)
                    fps = meter.tick()
                    if tracker is not None:
                        people = tracker.update(result, dt)
                        canvas = annotate_tracked(frame, people, config, fps)
                    else:
                        canvas = annotate(frame, result, config, fps)

                    if sink is not None:
                        sink.write(canvas)
                    cv2.imshow(WINDOW_NAME, canvas)
                    if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

        if config.save:
            print(f"Done. Wrote {config.save}")
```

- [ ] **Step 2: Run the full unit suite**

Run: `uv run pytest -q`
Expected: PASS (all tests, no import errors from the new runner)

- [ ] **Step 3: Headless end-to-end verification on a store clip**

Run this to process a segment through pipeline + tracker + drawing and confirm
stable ids and a clean saved file (downloads models on first run):

```bash
uv run python -c "
import cv2, glob, os
from storepose.config import AppConfig
from storepose.pipeline import PosePipeline
from storepose.tracking.tracker import MultiObjectTracker
from storepose.drawing import annotate_tracked
from storepose.video_sink import VideoSink

vid = sorted(glob.glob('videos/*.mp4'))[2]
cap = cv2.VideoCapture(vid)
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
cfg = AppConfig(mode='balanced', device='mps', save='outputs/demo_tracked.mp4')
pipe = PosePipeline(cfg)
trk = MultiObjectTracker(max_age=max(1, round(cfg.hold_seconds*src_fps)),
    min_hits=cfg.min_hits, iou_thr=cfg.iou_thr, smooth=cfg.smooth,
    min_cutoff=cfg.smooth_cutoff, beta=cfg.smooth_beta)
os.makedirs('outputs', exist_ok=True)
ids_seen=set(); n=0
with VideoSink(cfg.save, fps=src_fps) as sink:
    while n < 200:
        ok, frame = cap.read()
        if not ok: break
        people = trk.update(pipe.process(frame), 1/src_fps)
        ids_seen.update(p.id for p in people)
        sink.write(annotate_tracked(frame, people, cfg, fps=None)); n+=1
cap.release()
print('frames', n, 'distinct ids', len(ids_seen), 'wrote', cfg.save, os.path.getsize(cfg.save))
" 2>&1 | grep -vE "Downloading|load |%\||onnxruntime|x26|encoded|frame [IPB]|GetCapability|VerifyEach"
```

Expected: prints a frame count of 200, a small-ish number of distinct ids
(roughly the number of distinct people, not hundreds), and a non-zero file size.
Open `outputs/demo_tracked.mp4` and confirm boxes carry `ID n` labels that stay
on the same person, persist briefly through occlusion, and look less jittery.

- [ ] **Step 4: Commit**

```bash
git add src/storepose/runner.py
git commit -m "feat: wire tracker and smoothing into the realtime runner"
```

---

## Task 11: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document tracking**

In `README.md`, add these rows to the Flags table (after the `--save` row):

```markdown
| `--no-track`     | —          | Disable tracking; draw raw per-frame detections.  |
| `--hold-seconds` | `1.5`      | How long a lost person's box keeps coasting.      |
| `--min-hits`     | `3`        | Detections before a track is confirmed/drawn.     |
| `--iou-thr`      | `0.3`      | Min IoU to associate a detection to a track.      |
| `--no-smooth`    | —          | Disable One-Euro keypoint smoothing.              |
```

Add a new section after the Performance section:

```markdown
## Tracking & smoothing

By default each person gets a stable `ID n` and a SORT-style tracker (Kalman +
IoU) keeps that box alive through brief occlusions (coasting), while a One-Euro
filter smooths the skeleton. A box that loses its detection is predicted forward
for `--hold-seconds` (skeleton hidden while coasting), then dropped. Someone who
fully leaves and returns gets a new id (no appearance re-identification).

A/B the behavior with `--no-track` (raw per-frame boxes) and `--no-smooth`.
```

Add to the architecture file listing (inside the code block, under
`video_sink.py`):

```markdown
  tracking/        SORT tracker: assignment, kalman, smoothing, track, tracker
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document tracking and smoothing"
```

---

## Final verification

- [ ] **Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS — all prior tests plus the new tracking tests (≈28 new).

- [ ] **A/B sanity check (optional, manual)**

Run on a clip with tracking on vs off and confirm ids are stable and motion is
calmer with tracking on:

```bash
uv run python main.py --source "videos/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.mp4" --save outputs/tracked.mp4
uv run python main.py --source "videos/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.mp4" --no-track --no-smooth --save outputs/raw.mp4
```
