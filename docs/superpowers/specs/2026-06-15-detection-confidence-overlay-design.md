# Detection-confidence overlay (`--conf`) — design

**Date:** 2026-06-15
**Status:** approved (brainstorming) → implementing

## Problem

There's no way to see how confident the detector is about each person. rtmlib's
YOLOX in `human` mode computes `final_scores` but **discards them** —
`postprocess` returns boxes only, so `PersonDetector.detect()` yields `(N,4)`
with no score. We want a `--conf` flag that overlays the true detector
person-score per person, wired from the detector through to the draw stage.

## Decision (from brainstorming)

Show the **true YOLOX detector score** (not pose/keypoint confidence). This
requires exposing `final_scores` and threading a per-person score through the
tracker to each `TrackedPerson`.

## Architecture — a vertical slice

### 1. Expose the score — `detector.py`
- `_ScoredYOLOX(YOLOX)`: override `postprocess` to return `(boxes, scores)`,
  mirroring rtmlib's human-mode logic but keeping `final_scores`. Comment notes it
  mirrors the pinned rtmlib version.
- `PersonDetector.detect(frame) -> tuple[np.ndarray, np.ndarray]` returns
  `(boxes (N,4), scores (N,))`.
- Refactor box suppression so scores stay aligned: extract
  `_contained_keep_indices(boxes, thr) -> list[int]`; the public
  `suppress_contained_boxes(boxes, thr)` keeps its current boxes-returning
  signature (existing tests unaffected) and is implemented via the index helper.
  `detect()` uses the indices to filter **both** boxes and scores.
- Empty frame → `(np.empty((0,4)), np.empty((0,)))`.

### 2. Carry it — `pipeline.py`
- `FrameResult` gains `det_scores: np.ndarray` shape `(N,)`.
- `process()` unpacks `(boxes, scores)` from the detector and stores both.

### 3. Thread through tracking
- `tracking/track.py`: `Track` gains `self.score: float | None`; `__init__`,
  `update`, `reactivate` accept a `score` and store it. (Coasting leaves the
  previous score; the emit step decides what to surface.)
- `tracking/tracker.py`: read `det_scores = result.det_scores`; pass
  `det_scores[d]` on match (`update`), create (`Track(...)`), and `reactivate`.
- `tracking/types.py`: `TrackedPerson` gains `score: float | None` — set to the
  track's score when fresh, `None` while coasting (same convention as
  `keypoints`/`scores`).

### 4. Draw it — `drawing.py`
- When `config.show_conf`:
  - tracked path (`annotate_tracked`): append the score to the ID label
    (`ID 3  0.91`); skip when `score is None` (coasting).
  - untracked path (`annotate`): draw `det_scores[i]` near each box.
- Default off → no change to current overlays.

### 5. Flag — `config.py`
- `show_conf: bool = False`; `--conf` (`store_true`); threaded into `from_args`.

## Data flow

```
detector.detect -> (boxes, scores)
pipeline.process -> FrameResult(boxes, keypoints, scores, det_scores)
tracker.update -> per det d: Track gets det_scores[d]
              -> TrackedPerson.score (None while coasting)
drawing: if show_conf -> label box with the score
```

## Testing

`tests/test_detector.py` (extend): `detect`-level alignment is hard to unit-test
without a model, so test the suppression path keeps scores aligned via the index
helper (boxes + parallel scores in, correctly filtered out together).
`tests/tracking/`: tracker propagates `det_scores[d]` to `TrackedPerson.score`;
`score is None` on a coasting frame. `tests/test_config.py`: `--conf` parses to
`show_conf=True`. Drawing (cv2) stays thin, not unit-tested.

## Out of scope

- Pose/keypoint confidence display (rejected — chose detector score).
- Showing confidence in the dashboard or debug table.
- Logging confidences to CSV.
