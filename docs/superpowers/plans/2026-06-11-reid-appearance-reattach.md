# Re-id (Appearance Re-attach) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop one person from fragmenting into several track IDs (which inflates busy/wait metrics) by re-attaching a dropped track to the same person on reappearance, using a cheap appearance match behind a swappable interface.

**Architecture:** A new `AppearanceModel` protocol (v1 = HSV torso histogram) is injected into the SORT tracker. Each `Track` carries an EMA appearance descriptor. Confirmed tracks that age past `max_age` move to a TTL-bounded "lost gallery"; each frame, leftover detections are re-attached to unmatched active tracks or gallery entries by gated appearance similarity, reviving the original ID. Overlays color each person by their persistent track color.

**Tech Stack:** Python 3.12, numpy, OpenCV (`cv2`), scipy (existing), pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-06-11-reid-appearance-reattach-design.md`

---

## File Structure

- Create: `src/storepose/tracking/appearance.py` — `AppearanceModel` protocol + `HsvHistogramAppearance`.
- Modify: `src/storepose/tracking/track.py` — EMA descriptor storage + `reactivate()`.
- Modify: `src/storepose/tracking/tracker.py` — gallery, appearance re-attach, `update(..., frame)`.
- Modify: `src/storepose/config.py` — `reid`, `reid_seconds`, `reid_thr` + CLI.
- Modify: `src/storepose/runner.py` — build appearance model, pass `frame` into `update`.
- Modify: `src/storepose/drawing.py` — per-ID overlay colors, prune palette of near-orange.
- Create: `tests/tracking/test_appearance.py`
- Modify: `tests/tracking/test_track.py`, `tests/tracking/test_tracker.py`, `tests/test_config.py`, `tests/test_drawing.py`
- Modify: `README.md`, `docs/usage.md`

---

## Task 1: Appearance model (protocol + HSV histogram)

**Files:**
- Create: `src/storepose/tracking/appearance.py`
- Test: `tests/tracking/test_appearance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_appearance.py
import numpy as np

from storepose.tracking.appearance import HsvHistogramAppearance


def _frame_with_rect(color, x1, y1, x2, y2, size=200):
    f = np.zeros((size, size, 3), np.uint8)
    f[y1:y2, x1:x2] = color  # BGR
    return f


def test_same_color_crop_high_similarity():
    app = HsvHistogramAppearance(kpt_thr=0.5)
    box = np.array([40, 40, 120, 160], float)
    fa = _frame_with_rect((0, 0, 220), 40, 40, 120, 160)   # red person
    fb = _frame_with_rect((0, 0, 220), 50, 45, 130, 165)   # same red, shifted
    a = app.extract(fa, box, None, None)
    b = app.extract(fb, np.array([50, 45, 130, 165], float), None, None)
    assert a is not None and b is not None
    assert app.similarity(a, b) > 0.8


def test_different_color_low_similarity():
    app = HsvHistogramAppearance(kpt_thr=0.5)
    box = np.array([40, 40, 120, 160], float)
    red = app.extract(_frame_with_rect((0, 0, 220), 40, 40, 120, 160), box, None, None)
    blue = app.extract(_frame_with_rect((220, 0, 0), 40, 40, 120, 160), box, None, None)
    assert red is not None and blue is not None
    assert app.similarity(red, blue) < 0.3


def test_keypoints_localize_torso_over_box():
    app = HsvHistogramAppearance(kpt_thr=0.5)
    # box spans a green background; only the torso quad is red
    f = np.zeros((200, 200, 3), np.uint8)
    f[20:180, 20:120] = (0, 200, 0)      # green box area
    f[60:120, 50:90] = (0, 0, 220)       # red torso
    box = np.array([20, 20, 120, 180], float)
    kpts = np.zeros((17, 2), float)
    kpts[5] = (50, 60); kpts[6] = (90, 60); kpts[11] = (50, 120); kpts[12] = (90, 120)
    scores = np.ones(17, float)
    desc = app.extract(f, box, kpts, scores)
    red_only = app.extract(_frame_with_rect((0, 0, 220), 50, 60, 90, 120), box, None, None)
    assert app.similarity(desc, red_only) > 0.8  # torso (red) not box (green)


def test_degenerate_crop_returns_none():
    app = HsvHistogramAppearance(kpt_thr=0.5)
    f = np.zeros((200, 200, 3), np.uint8)            # all black -> fully masked
    assert app.extract(f, np.array([10, 10, 60, 120], float), None, None) is None
    tiny = app.extract(f, np.array([10, 10, 11, 11], float), None, None)
    assert tiny is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tracking/test_appearance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'storepose.tracking.appearance'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/storepose/tracking/appearance.py
"""Appearance descriptors for re-id. v1: HSV torso-color histogram.

Injectable behind the AppearanceModel protocol so a stronger ReID embedding
(e.g. OSNet ONNX) can replace the histogram without touching the tracker.
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

# COCO keypoint indices for the torso quad.
_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 5, 6, 11, 12

_H_BINS, _S_BINS = 32, 32
_SAT_MIN, _VAL_MIN, _VAL_MAX = 40, 40, 250  # background/shadow/highlight mask (0-255)


class AppearanceModel(Protocol):
    """Extracts a per-person descriptor and scores descriptor similarity."""

    def extract(self, frame, box, keypoints, scores) -> np.ndarray | None: ...
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float: ...


def _torso_rect(box, keypoints, scores, kpt_thr) -> tuple[int, int, int, int]:
    """Torso pixel rect from shoulder/hip keypoints, else the box's chest band."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if keypoints is not None and scores is not None:
        idx = (_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP)
        if all(float(scores[i]) >= kpt_thr for i in idx):
            xs = [float(keypoints[i][0]) for i in idx]
            ys = [float(keypoints[i][1]) for i in idx]
            if max(xs) > min(xs) and max(ys) > min(ys):
                return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    w, h = x2 - x1, y2 - y1
    cx = x1 + w / 2.0
    return (int(cx - w * 0.25), int(y1 + h * 0.15),
            int(cx + w * 0.25), int(y1 + h * 0.45))


class HsvHistogramAppearance:
    """Normalized H-S histogram of the masked torso crop; correlation similarity."""

    def __init__(self, kpt_thr: float = 0.5):
        self.kpt_thr = kpt_thr

    def extract(self, frame, box, keypoints, scores) -> np.ndarray | None:
        h_img, w_img = frame.shape[:2]
        rx1, ry1, rx2, ry2 = _torso_rect(box, keypoints, scores, self.kpt_thr)
        rx1 = max(0, min(rx1, w_img - 1)); rx2 = max(0, min(rx2, w_img))
        ry1 = max(0, min(ry1, h_img - 1)); ry2 = max(0, min(ry2, h_img))
        if rx2 - rx1 < 2 or ry2 - ry1 < 2:
            return None
        hsv = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        mask = ((s >= _SAT_MIN) & (v >= _VAL_MIN) & (v <= _VAL_MAX)).astype(np.uint8) * 255
        if int(mask.sum()) == 0:
            return None
        hist = cv2.calcHist([hsv], [0, 1], mask, [_H_BINS, _S_BINS], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.astype(np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return -1.0
        return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tracking/test_appearance.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/tracking/appearance.py tests/tracking/test_appearance.py
git commit -m "feat: HSV torso-histogram appearance model for re-id"
```

---

## Task 2: Track EMA descriptor + reactivate()

**Files:**
- Modify: `src/storepose/tracking/track.py`
- Test: `tests/tracking/test_track.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/tracking/test_track.py
import numpy as np
from storepose.tracking.track import Track


def _track(box, descriptor=None):
    return Track(0, np.array(box, float), None, None, 1 / 30,
                 min_hits=1, smooth=False, min_cutoff=1.0, beta=0.007,
                 descriptor=descriptor)


def test_descriptor_seeded_then_blended():
    t = _track([0, 0, 10, 20], descriptor=np.array([1.0, 0.0], np.float32))
    assert np.allclose(t.descriptor, [1.0, 0.0])
    t.update(np.array([0, 0, 10, 20], float), None, None, 1 / 30,
             descriptor=np.array([0.0, 1.0], np.float32))
    # EMA alpha=0.3 -> 0.7*[1,0] + 0.3*[0,1]
    assert np.allclose(t.descriptor, [0.7, 0.3])


def test_reactivate_reseats_motion_and_keeps_identity():
    t = _track([0, 0, 10, 20], descriptor=np.array([1.0], np.float32))
    t.confirmed = True
    for _ in range(5):
        t.predict()
    assert t.time_since_update == 5
    t.reactivate(np.array([100, 100, 110, 120], float), None, None, 1 / 30,
                 descriptor=np.array([1.0], np.float32))
    assert t.time_since_update == 0
    assert t.confirmed is True
    assert np.allclose(t.box, [100, 100, 110, 120], atol=1.0)


def test_update_descriptor_none_keeps_previous():
    t = _track([0, 0, 10, 20], descriptor=np.array([1.0, 0.0], np.float32))
    t.update(np.array([0, 0, 10, 20], float), None, None, 1 / 30, descriptor=None)
    assert np.allclose(t.descriptor, [1.0, 0.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tracking/test_track.py -k "descriptor or reactivate" -v`
Expected: FAIL — `Track.__init__() got an unexpected keyword argument 'descriptor'`

- [ ] **Step 3: Write minimal implementation**

In `src/storepose/tracking/track.py`, add the EMA constant near the top (after `_PALETTE`):

```python
_DESC_ALPHA = 0.3  # appearance descriptor EMA weight for new observations
```

Change `Track.__init__` signature to accept `descriptor` and store it (add the
keyword-only param after `beta` and set the attribute before `_ingest_pose`):

```python
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
        descriptor: np.ndarray | None = None,
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
        self.descriptor: np.ndarray | None = (
            None if descriptor is None else np.asarray(descriptor, np.float32)
        )
        self._ingest_pose(keypoints, scores, dt)
```

Add a descriptor-blend helper and call it from `update`; add `reactivate`:

```python
    def _update_descriptor(self, descriptor) -> None:
        if descriptor is None:
            return
        descriptor = np.asarray(descriptor, np.float32)
        if self.descriptor is None:
            self.descriptor = descriptor
        else:
            self.descriptor = (
                (1.0 - _DESC_ALPHA) * self.descriptor + _DESC_ALPHA * descriptor
            ).astype(np.float32)

    def update(self, box, keypoints, scores, dt, descriptor=None) -> None:
        """Correct with a matched detection."""
        self.kalman.update(box)
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self.min_hits:
            self.confirmed = True
        self._ingest_pose(keypoints, scores, dt)
        self._update_descriptor(descriptor)

    def reactivate(self, box, keypoints, scores, dt, descriptor=None) -> None:
        """Revive a lost/coasting track at a new detection, keeping id and color."""
        self.kalman = KalmanBoxTracker(box)
        self.time_since_update = 0
        self.hits += 1
        self.confirmed = True
        self._ingest_pose(keypoints, scores, dt)
        self._update_descriptor(descriptor)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tracking/test_track.py -v`
Expected: PASS (existing track tests + 3 new)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/tracking/track.py tests/tracking/test_track.py
git commit -m "feat: Track carries EMA appearance descriptor + reactivate()"
```

---

## Task 3: Tracker gallery + appearance re-attach

**Files:**
- Modify: `src/storepose/tracking/tracker.py`
- Test: `tests/tracking/test_tracker.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/tracking/test_tracker.py
class _ColorStub:
    """Appearance descriptor = BGR pixel at the box center of the frame.

    Lets tests bind identity to a painted color, exercising the real frame
    plumbing. Similarity is exact-match (1.0 / 0.0).
    """
    def extract(self, frame, box, keypoints, scores):
        cx = int((box[0] + box[2]) / 2.0); cy = int((box[1] + box[3]) / 2.0)
        h, w = frame.shape[:2]
        cx = min(max(cx, 0), w - 1); cy = min(max(cy, 0), h - 1)
        px = frame[cy, cx]
        return None if int(px.sum()) == 0 else px.astype(np.float32)

    def similarity(self, a, b):
        return 1.0 if np.array_equal(a, b) else 0.0


def _frame(boxes_colors, size=400):
    f = np.zeros((size, size, 3), np.uint8)
    for (x1, y1, x2, y2), color in boxes_colors:
        f[int(y1):int(y2), int(x1):int(x2)] = color
    return f


def _reid_tracker(max_age=3, reid_max_age=50):
    return MultiObjectTracker(
        max_age=max_age, min_hits=1, iou_thr=0.3, smooth=False,
        appearance=_ColorStub(), reid=True, reid_max_age=reid_max_age, reid_thr=0.5,
    )


def test_reattach_same_appearance_revives_id():
    tr = _reid_tracker()
    box = [100, 100, 140, 180]
    red = (0, 0, 220)
    out = tr.update(make_result([box]), 1 / 30, _frame([(box, red)]))
    assert out[0].id == 0
    blank = np.zeros((400, 400, 3), np.uint8)
    for _ in range(5):                       # age out past max_age -> gallery
        tr.update(make_result([]), 1 / 30, blank)
    back = [110, 105, 150, 185]
    out2 = tr.update(make_result([back]), 1 / 30, _frame([(back, red)]))
    assert len(out2) == 1 and out2[0].id == 0   # same id revived


def test_reattach_different_appearance_gets_new_id():
    tr = _reid_tracker()
    box = [100, 100, 140, 180]
    tr.update(make_result([box]), 1 / 30, _frame([(box, (0, 0, 220))]))  # red, id 0
    blank = np.zeros((400, 400, 3), np.uint8)
    for _ in range(5):
        tr.update(make_result([]), 1 / 30, blank)
    back = [110, 105, 150, 185]
    out2 = tr.update(make_result([back]), 1 / 30, _frame([(back, (0, 220, 0))]))  # green
    assert out2[0].id == 1                       # different person -> new id


def test_reattach_past_ttl_gets_new_id():
    tr = _reid_tracker(max_age=2, reid_max_age=2)
    box = [100, 100, 140, 180]; red = (0, 0, 220)
    tr.update(make_result([box]), 1 / 30, _frame([(box, red)]))   # id 0
    blank = np.zeros((400, 400, 3), np.uint8)
    for _ in range(8):                            # past max_age AND reid TTL
        tr.update(make_result([]), 1 / 30, blank)
    back = [110, 105, 150, 185]
    out2 = tr.update(make_result([back]), 1 / 30, _frame([(back, red)]))
    assert out2[0].id == 1                        # gallery expired -> new id


def test_occluded_person_reattaches_without_swapping_neighbor():
    tr = _reid_tracker()
    a = [40, 100, 80, 180]; b = [300, 100, 340, 180]
    red, green = (0, 0, 220), (0, 220, 0)
    out = tr.update(make_result([a, b]), 1 / 30, _frame([(a, red), (b, green)]))
    ids = {tuple(p.box[:2].round()): p.id for p in out}
    assert len(out) == 2
    # A occluded for several frames; B stays
    for _ in range(5):
        tr.update(make_result([b]), 1 / 30, _frame([(b, green)]))
    a_back = [45, 105, 85, 185]
    out2 = tr.update(make_result([a_back, b]), 1 / 30, _frame([(a_back, red), (b, green)]))
    by_color = {}
    for p in out2:
        cx = int((p.box[0] + p.box[2]) / 2); by_color[p.id] = cx
    assert len(out2) == 2 and len(set(by_color)) == 2   # two distinct ids, no merge


def test_reid_disabled_returns_new_id_on_reappearance():
    tr = MultiObjectTracker(max_age=2, min_hits=1, iou_thr=0.3, smooth=False)  # reid off
    box = [100, 100, 140, 180]
    tr.update(make_result([box]), 1 / 30)        # no frame arg
    for _ in range(4):
        tr.update(make_result([]), 1 / 30)
    out = tr.update(make_result([[110, 105, 150, 185]]), 1 / 30)
    assert out[0].id == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tracking/test_tracker.py -k reattach -v`
Expected: FAIL — `MultiObjectTracker.__init__() got an unexpected keyword argument 'appearance'`

- [ ] **Step 3: Write minimal implementation**

Replace `src/storepose/tracking/tracker.py` with:

```python
"""SORT-style multi-object tracker producing stable, coasted tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .appearance import AppearanceModel
from .assignment import iou, match
from .track import Track
from .types import TrackedPerson

# Appearance re-attach spatial gate: a candidate must lie within this radius of
# the detection. Radius grows with the gap (a person moves while gone), capped.
_SPATIAL_GATE_FRAC = 0.12   # base radius as a fraction of the frame diagonal
_GATE_GROWTH = 0.05         # extra radius per gap frame
_GATE_CAP_FRAC = 0.5        # never exceed half the frame diagonal


def _center(box: np.ndarray) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


@dataclass
class _LostEntry:
    """A confirmed track that aged out, awaiting appearance re-attach."""
    track: Track
    center: tuple[float, float]
    lost_age: int


@dataclass
class _Candidate:
    """A re-attach candidate: an unmatched active track or a gallery entry."""
    track: Track
    lost: _LostEntry | None
    center: tuple[float, float]
    gap: int
    descriptor: np.ndarray


def suppress_coasting_duplicates(tracks: list, max_overlap: float) -> list:
    """Remove coasting (predicted) tracks that duplicate a kept track.

    A coasting track is dropped when its box overlaps a higher-priority kept
    track by more than ``max_overlap``. Non-coasting tracks are never dropped,
    so two genuinely-detected people are never merged — this only removes the
    "ghost" left behind when a track coasts while the same person re-spawns a
    new track. Priority: actively-matched, then confirmed, then more hits.
    """
    order = sorted(tracks, key=lambda t: (t.coasting, not t.confirmed, -t.hits))
    kept: list = []
    for t in order:
        if t.coasting and any(iou(t.box, k.box) > max_overlap for k in kept):
            continue
        kept.append(t)
    return kept


class MultiObjectTracker:
    """Associates per-frame detections to persistent, smoothed tracks.

    Each ``update`` predicts existing tracks, associates them to the frame's
    detections by IoU, updates matches, coasts the rest (up to ``max_age``), and
    emits confirmed tracks as :class:`TrackedPerson`. When ``reid`` is enabled
    and a frame is supplied, leftover detections are re-attached to unmatched
    active tracks or aged-out gallery entries by gated appearance similarity,
    reviving the original id instead of spawning a new one.
    """

    def __init__(
        self,
        max_age: int = 45,
        min_hits: int = 3,
        iou_thr: float = 0.3,
        smooth: bool = True,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        max_overlap: float = 0.5,
        appearance: AppearanceModel | None = None,
        reid: bool = False,
        reid_max_age: int = 150,
        reid_thr: float = 0.6,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thr = iou_thr
        self.smooth = smooth
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.max_overlap = max_overlap
        self._appearance = appearance
        self._reid = reid
        self._reid_max_age = reid_max_age
        self._reid_thr = reid_thr
        self._tracks: list[Track] = []
        self._lost: list[_LostEntry] = []
        self._next_id = 0

    def update(self, result, dt: float, frame=None) -> list[TrackedPerson]:
        boxes = result.boxes
        keypoints = result.keypoints
        scores = result.scores
        n = len(boxes)

        use_reid = self._reid and self._appearance is not None and frame is not None
        if use_reid:
            det_descs = [
                self._appearance.extract(frame, boxes[i], keypoints[i], scores[i])
                for i in range(n)
            ]
        else:
            det_descs = [None] * n

        # 1. predict existing tracks forward; age the lost gallery
        for t in self._tracks:
            t.predict()
        if use_reid:
            for e in self._lost:
                e.lost_age += 1
            self._lost = [e for e in self._lost if e.lost_age <= self._reid_max_age]

        # 2. associate detections to predicted tracks by IoU
        det_boxes = [boxes[i] for i in range(n)]
        track_boxes = [t.box for t in self._tracks]
        matches, unmatched_dets, unmatched_tracks = match(
            det_boxes, track_boxes, self.iou_thr
        )

        # 3. update matched tracks
        for d, tr in matches:
            self._tracks[tr].update(
                boxes[d], keypoints[d], scores[d], dt, descriptor=det_descs[d]
            )

        # 4. appearance re-attach leftover detections
        if use_reid and unmatched_dets:
            unmatched_dets = self._reattach(
                unmatched_dets, unmatched_tracks, boxes, keypoints, scores,
                det_descs, frame, dt,
            )

        # 5. spawn tracks for still-unmatched detections
        for d in unmatched_dets:
            self._tracks.append(
                Track(
                    self._next_id, boxes[d], keypoints[d], scores[d], dt,
                    min_hits=self.min_hits, smooth=self.smooth,
                    min_cutoff=self.min_cutoff, beta=self.beta,
                    descriptor=det_descs[d],
                )
            )
            self._next_id += 1

        # 6. cull: aged-out confirmed tracks go to the gallery; drop tentatives
        survivors: list[Track] = []
        for t in self._tracks:
            if t.time_since_update >= 1 and not t.confirmed:
                continue
            if t.time_since_update > self.max_age:
                if use_reid and t.confirmed and t.descriptor is not None:
                    self._lost.append(
                        _LostEntry(track=t, center=_center(t.box), lost_age=0)
                    )
                continue
            survivors.append(t)
        self._tracks = survivors

        # 6b. drop coasting ghosts that duplicate another track
        self._tracks = suppress_coasting_duplicates(self._tracks, self.max_overlap)

        # 7. emit confirmed tracks
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

    def _reattach(
        self, unmatched_dets, unmatched_tracks, boxes, keypoints, scores,
        det_descs, frame, dt,
    ) -> list[int]:
        """Revive ids for leftover detections via gated appearance match.

        Returns the detection indices that remain unmatched (to be spawned).
        """
        diag = float(np.hypot(frame.shape[1], frame.shape[0]))

        cands: list[_Candidate] = []
        for tr_idx in unmatched_tracks:
            t = self._tracks[tr_idx]
            if not t.confirmed or t.descriptor is None:
                continue
            cands.append(_Candidate(t, None, _center(t.box), t.time_since_update, t.descriptor))
        for e in self._lost:
            if e.track.descriptor is None:
                continue
            cands.append(_Candidate(e.track, e, e.center, e.lost_age, e.track.descriptor))
        if not cands:
            return unmatched_dets

        # (cost, det_index, cand_index) for every gated, above-threshold pair
        pairs: list[tuple[float, int, int]] = []
        for d in unmatched_dets:
            dd = det_descs[d]
            if dd is None:
                continue
            dcx, dcy = _center(boxes[d])
            for ci, cand in enumerate(cands):
                radius = min(
                    _SPATIAL_GATE_FRAC * diag * (1.0 + _GATE_GROWTH * cand.gap),
                    _GATE_CAP_FRAC * diag,
                )
                if np.hypot(dcx - cand.center[0], dcy - cand.center[1]) > radius:
                    continue
                sim = self._appearance.similarity(dd, cand.descriptor)
                if sim < self._reid_thr:
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
            cand.track.reactivate(boxes[d], keypoints[d], scores[d], dt, descriptor=det_descs[d])
            if cand.lost is not None:
                self._lost.remove(cand.lost)
                self._tracks.append(cand.track)
        return [d for d in unmatched_dets if d not in used_d]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tracking/test_tracker.py -v`
Expected: PASS (existing tracker tests + 5 new). The existing no-frame calls
still work because `frame=None` disables re-id.

- [ ] **Step 5: Commit**

```bash
git add src/storepose/tracking/tracker.py tests/tracking/test_tracker.py
git commit -m "feat: lost-track gallery + gated appearance re-attach in tracker"
```

---

## Task 4: Config flags (reid, reid_seconds, reid_thr)

**Files:**
- Modify: `src/storepose/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config.py
from storepose.config import AppConfig, from_args


def test_reid_defaults_on():
    cfg = from_args([])
    assert cfg.reid is True
    assert cfg.reid_seconds == 5.0
    assert cfg.reid_thr == 0.6


def test_no_reid_flag_disables():
    assert from_args(["--no-reid"]).reid is False


def test_reid_flags_parse():
    cfg = from_args(["--reid-seconds", "8", "--reid-thr", "0.4"])
    assert cfg.reid_seconds == 8.0 and cfg.reid_thr == 0.4


def test_reid_seconds_must_be_nonnegative():
    import pytest
    with pytest.raises(ValueError):
        AppConfig(reid_seconds=-1.0)


def test_reid_thr_must_be_in_range():
    import pytest
    with pytest.raises(ValueError):
        AppConfig(reid_thr=2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -k reid -v`
Expected: FAIL — `AttributeError: 'AppConfig' object has no attribute 'reid'`

- [ ] **Step 3: Write minimal implementation**

In `src/storepose/config.py`, add fields to `AppConfig` (after `max_overlap` block, before `smooth`):

```python
    reid: bool = True
    reid_seconds: float = 5.0
    reid_thr: float = 0.6
```

Add validation in `__post_init__` (after the `max_overlap` check):

```python
        if self.reid_seconds < 0:
            raise ValueError(f"reid_seconds must be >= 0, got {self.reid_seconds}")
        if not -1.0 <= self.reid_thr <= 1.0:
            raise ValueError(f"reid_thr must be in [-1, 1], got {self.reid_thr}")
```

Add CLI args in `_build_parser` (after the `--max-overlap` argument):

```python
    parser.add_argument(
        "--no-reid", dest="reid", action="store_false",
        help="Disable appearance re-id (re-attaching a returning person's id).",
    )
    parser.add_argument(
        "--reid-seconds", type=float, default=5.0,
        help="How long a lost track stays re-attachable, in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--reid-thr", type=float, default=0.6,
        help="Appearance similarity floor for re-attach, in [-1,1] (default: 0.6).",
    )
```

Wire into `from_args`'s `AppConfig(...)` call (add after `max_overlap=args.max_overlap,`):

```python
        reid=args.reid,
        reid_seconds=args.reid_seconds,
        reid_thr=args.reid_thr,
```

Also add the three attributes to the `AppConfig` docstring Attributes list (after `max_overlap`):

```python
        reid: Re-attach a returning person's id via appearance (requires tracking).
        reid_seconds: How long a lost track stays re-attachable, in seconds.
        reid_thr: Appearance similarity floor for re-attach, in [-1, 1].
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (existing config tests + 5 new)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/config.py tests/test_config.py
git commit -m "feat: --no-reid / --reid-seconds / --reid-thr config flags"
```

---

## Task 5: Wire re-id into the runner

**Files:**
- Modify: `src/storepose/runner.py:50-59` (tracker construction) and `:109-110` (update call)

- [ ] **Step 1: Add the appearance import**

At the top of `src/storepose/runner.py`, add to the tracking imports group:

```python
from .tracking.appearance import HsvHistogramAppearance
```

- [ ] **Step 2: Build the appearance model and pass reid params**

Replace the tracker-construction block (currently `if config.track:` ... `MultiObjectTracker(...)`):

```python
                tracker = None
                if config.track:
                    base_fps = source.fps or _DEFAULT_FPS
                    max_age = max(1, round(config.hold_seconds * base_fps))
                    appearance = (
                        HsvHistogramAppearance(kpt_thr=config.kpt_thr)
                        if config.reid else None
                    )
                    reid_max_age = max(1, round(config.reid_seconds * base_fps))
                    tracker = MultiObjectTracker(
                        max_age=max_age, min_hits=config.min_hits,
                        iou_thr=config.iou_thr, max_overlap=config.max_overlap,
                        smooth=config.smooth,
                        min_cutoff=config.smooth_cutoff, beta=config.smooth_beta,
                        appearance=appearance, reid=config.reid,
                        reid_max_age=reid_max_age, reid_thr=config.reid_thr,
                    )
```

- [ ] **Step 3: Pass the frame into update**

Change the tracker update call (currently `people = tracker.update(result, dt)`):

```python
                        people = tracker.update(result, dt, frame)
```

- [ ] **Step 4: Verify nothing broke and re-id is wired**

Run: `uv run pytest -q`
Expected: PASS (full suite).

Run: `uv run python main.py --source /tmp/sco_clip.mp4 --zone "zones/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.json" --save /tmp/reid_out.mp4 2>&1 | grep -v -E "onnxruntime|CoreML"`
Expected: runs to completion, prints `Done. Wrote /tmp/reid_out.mp4` (re-id on by default, no crash). If `/tmp/sco_clip.mp4` is absent, regenerate it with the 200-frame OpenCV trim from earlier, or use the full video path.

- [ ] **Step 5: Commit**

```bash
git add src/storepose/runner.py
git commit -m "feat: wire appearance re-id into the runner loop"
```

---

## Task 6: Per-ID overlay colors + palette prune

**Files:**
- Modify: `src/storepose/tracking/track.py` (`_PALETTE`)
- Modify: `src/storepose/drawing.py` (`annotate_queue`)
- Test: `tests/test_drawing.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_drawing.py
def test_queue_waiting_fill_uses_person_color_not_green():
    from storepose.drawing import annotate_queue
    from storepose.queue.types import PersonStatus, QueueResult
    from storepose.queue.zone import Zone
    frame = _blank()
    blue = (255, 0, 0)  # BGR blue person color
    person = TrackedPerson(id=1, box=np.array([20, 20, 100, 110], float),
                           keypoints=None, scores=None, coasting=False, color=blue)
    result = QueueResult(
        statuses=[PersonStatus(id=1, waiting=True, candidate=False, progress=1.0, wait_seconds=3.4)],
        count=1,
    )
    zone = Zone([(0, 0), (160, 0), (160, 120), (0, 120)])
    out = annotate_queue(frame.copy(), [person], result, zone, AppConfig())
    cy, cx = 65, 60  # inside the box
    b, g, r = out[cy, cx]
    assert b > g and b > r  # tinted toward the person's blue, not green


def test_queue_candidate_fill_uses_person_color_not_orange():
    from storepose.drawing import annotate_queue
    from storepose.queue.types import PersonStatus, QueueResult
    from storepose.queue.zone import Zone
    frame = _blank()
    blue = (255, 0, 0)
    person = TrackedPerson(id=3, box=np.array([20, 20, 100, 110], float),
                           keypoints=None, scores=None, coasting=False, color=blue)
    result = QueueResult(
        statuses=[PersonStatus(id=3, waiting=False, candidate=True, progress=0.9, wait_seconds=0.0)],
        count=0,
    )
    zone = Zone([(0, 0), (160, 0), (160, 120), (0, 120)])
    out = annotate_queue(frame.copy(), [person], result, zone, AppConfig())
    cy, cx = 100, 60  # inside the rising fill near the box bottom
    b, g, r = out[cy, cx]
    assert b > r  # person blue, not orange (orange would have r >= b)


def test_palette_has_no_near_orange_color():
    from storepose.tracking.track import _PALETTE
    from storepose.drawing import ZONE_COLOR
    zb, zg, zr = ZONE_COLOR  # (0, 180, 255) orange
    for b, g, r in _PALETTE:
        # an orange is low-blue, mid/high-green, high-red; ensure none collide
        assert not (b < 80 and g > 120 and r > 200)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_drawing.py -k "person_color or palette" -v`
Expected: FAIL — waiting/candidate fills are still green/orange; `_PALETTE` still contains `(56, 153, 255)`.

- [ ] **Step 3: Write minimal implementation**

In `src/storepose/tracking/track.py`, replace `_PALETTE` (drop the near-orange `(56, 153, 255)`):

```python
# Distinct BGR colors cycled by track id. No near-orange entry, so a person's
# color never collides with the orange queue zone fill (drawing.ZONE_COLOR).
_PALETTE: list[tuple[int, int, int]] = [
    (56, 56, 255), (56, 255, 56), (255, 56, 56), (56, 255, 255),
    (255, 56, 255), (255, 255, 56), (255, 153, 56), (255, 56, 153),
    (153, 56, 255), (56, 255, 153),
]
```

In `src/storepose/drawing.py`, change `annotate_queue` so the per-person fills
use `p.color`. Replace the `if s.waiting:` / `elif s.candidate:` block body
(keep the zone-drawing and header code around it unchanged) with:

```python
        if s.waiting:
            # solid translucent fill over the whole box, in the person's color
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), p.color, -1)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
            tag = f"WAIT {s.wait_seconds:0.1f}s"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = min(y2 + th + 6, canvas.shape[0] - 2)
            cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty), p.color, -1)
            cv2.putText(canvas, tag, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 2, cv2.LINE_AA)
        elif s.candidate:
            # "sheer" fill rising from the bottom as inclusion progresses
            fill_h = int(round((y2 - y1) * max(0.0, min(s.progress, 1.0))))
            fy1 = y2 - fill_h
            if fill_h > 0:
                overlay = canvas.copy()
                cv2.rectangle(overlay, (x1, fy1), (x2, y2), p.color, -1)
                cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
            pct = f"{int(round(s.progress * 100))}%"
            cv2.putText(canvas, pct, (x1 + 3, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, p.color, 2, cv2.LINE_AA)
```

(The zone polygon still draws in `ZONE_COLOR`, and the `in line: N` header stays
`ZONE_COLOR` — both unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_drawing.py -v`
Expected: PASS (existing drawing tests + 3 new). Note the existing
`test_annotate_queue_draws_zone_and_count` still passes because the zone polygon
remains orange (red channel present).

- [ ] **Step 5: Commit**

```bash
git add src/storepose/tracking/track.py src/storepose/drawing.py tests/test_drawing.py
git commit -m "feat: color queue overlays by persistent track id; prune palette of orange"
```

---

## Task 7: Documentation

**Files:**
- Modify: `README.md`, `docs/usage.md`

- [ ] **Step 1: Update the README flag table and tracking section**

In `README.md`, add rows to the flag table (after the `--max-overlap` row):

```markdown
| `--no-reid`      | —       | Disable appearance re-id (returning person keeps their id). |
| `--reid-seconds` | `5.0`   | How long a lost track stays re-attachable.        |
| `--reid-thr`     | `0.6`   | Appearance similarity floor for re-attach (HSV histogram correlation). |
```

In the "Tracking & smoothing" section, replace the sentence
"Someone who fully leaves and returns gets a new id (no appearance
re-identification)." with:

```markdown
By default, appearance re-id (HSV torso-color histogram) re-attaches a returning
person to their original id within `--reid-seconds`; disable it with `--no-reid`.
A genuinely new person still gets a new id. Each id keeps a persistent overlay
color across re-attach.
```

- [ ] **Step 2: Update usage.md**

In `docs/usage.md` Section 2 "Overlay layers" table, change the Queue row's
"What you see" to note per-id color:

```markdown
| Queue | `--zone PATH` | The zone polygon (orange); a rising fill + `%` for candidates and a full fill + `WAIT n.n s` for people in line, each in that person's **persistent id color**; an `in line: N` header count. |
```

In Section 2 "Useful run flags" table, add after the `--no-track` row:

```markdown
| `--no-reid` | — | Disable appearance re-id; a returning person gets a new id. |
| `--reid-seconds` | `5.0` | How long a lost track stays re-attachable. |
| `--reid-thr` | `0.6` | Appearance similarity floor for re-attach. |
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/usage.md
git commit -m "docs: document appearance re-id flags and per-id overlay color"
```

---

## Task 8: Full-suite verification

- [ ] **Step 1: Run the whole suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Single-line live smoke (re-id on)**

Note: one line — no backslash continuations.

Run: `uv run python main.py --source /tmp/sco_clip.mp4 --zone "zones/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.json" --busy --busy-log /tmp/busy.csv --wait-log /tmp/waits.csv --save /tmp/reid_out.mp4`
Expected: runs to completion; `busy.csv`/`waits.csv`/`reid_out.mp4` written; people in line carry their own persistent colors and IDs survive brief occlusions. (Press `q`/Esc only if running the full video instead of the clip.)

- [ ] **Step 3: A/B check (optional)**

Run the same command with `--no-reid` and compare `waits.csv` row counts — re-id
should yield fewer, longer waits (less fragmentation) on footage where people are
briefly occluded.

---

## Self-Review Notes

- **Spec coverage:** appearance seam (Task 1) ✓; Track EMA + reactivate (Task 2) ✓; gallery + gated re-attach with active∪gallery candidates, precision bias, revive-id (Task 3) ✓; config `--no-reid`/`--reid-seconds`/`--reid-thr` defaults on/5.0/0.6 (Task 4) ✓; runner frame plumbing + model construction (Task 5) ✓; per-id overlay color + palette prune + zone stays orange (Task 6) ✓; analyzer untouched (no task needed — revive-id continues the wait timer) ✓; docs (Task 7) ✓.
- **Signatures are consistent across tasks:** `extract(frame, box, keypoints, scores)`, `similarity(a, b)`, `Track.update(..., descriptor=None)`, `Track.reactivate(...)`, `MultiObjectTracker(..., appearance, reid, reid_max_age, reid_thr)`, `update(result, dt, frame=None)`.
- **Backward compatibility:** tracker constructor defaults `reid=False`; `update` defaults `frame=None`; so all pre-existing tracker tests (no frame, no appearance) keep their exact behavior.
- **`reid_max_age` units:** frames (runner converts `reid_seconds * fps`); tests pass it directly in frames.
```
