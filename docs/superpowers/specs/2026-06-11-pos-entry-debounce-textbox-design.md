# POS entry debounce + bottom serving textbox

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation
**Branch:** feat/realtime-pose

## Goal

Make `WAITING → SERVING` require a short debounce (a foot grazing the POS edge for
1–2 frames no longer flips someone to serving), and show a small bottom panel
listing each person currently at the POS with their serving time.

## Components

### `src/storepose/queue/analyzer.py`
- `__init__` gains `pos_enter_frames: int = 3` (stored as `max(1, ...)`).
- `_VisitState` gains `pos_frames: int = 0`.
- WAITING branch: while `in_pos`, accumulate `pos_frames` (and keep `out_streak`
  at 0); transition to SERVING only once `pos_frames >= pos_enter_frames` (then
  reset `pos_frames`). If `in_line` but not `in_pos`, reset `pos_frames` to 0.
  Otherwise (out of both) the existing abandon countdown applies. The grazing
  frames still accrue **waiting** time.
- Leaving SERVING is unchanged — already debounced by `exit_seconds` (a brief blip
  out of POS doesn't end serving; a sustained `exit_seconds` out finalizes SERVED).
- The `OUT → SERVING` walk-up is unchanged (already gated by `enter_frames` /
  `min_dwell`).

### `src/storepose/config.py`
- `pos_enter_frames: int = 3` + `--pos-enter-frames` flag; validate `>= 1`.

### `src/storepose/runner.py`
- Pass `pos_enter_frames=config.pos_enter_frames` to `QueueAnalyzer`.

### `src/storepose/drawing.py`
- A `_draw_pos_panel(canvas, people, result)` helper, called at the end of
  `annotate_queue` only when at least one status has `serving=True`. Draws a
  semi-transparent panel anchored to the **bottom** of the frame, one line per
  serving person: `AT POS  —  ID {id}: {serving_seconds:.1f}s`, the `ID n` chip in
  that person's color (looked up from `people`). Hidden when nobody is serving.
  Stroke/font resolution-scaled like the other overlays (reuse the `_scale`
  helper).

## Testing
- analyzer: a waiting person in POS for `pos_enter_frames - 1` frames is still
  WAITING; on the `pos_enter_frames`-th consecutive in-POS frame becomes SERVING;
  a single in-POS frame followed by in-line-not-POS resets and stays WAITING.
- config: `--pos-enter-frames` parses; default 3; `< 1` raises.
- drawing: with a serving status the panel draws pixels near the bottom; with no
  serving status the bottom rows are unchanged from the no-panel render.

## Out of scope
The web dashboard (next sub-project).
