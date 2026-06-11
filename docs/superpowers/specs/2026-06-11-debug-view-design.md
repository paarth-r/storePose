# Frame-by-frame debug view

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation
**Branch:** feat/realtime-pose

## Goal

A `--debug` mode to step through the video frame by frame (and scrub back through a
rolling buffer) while the dashboard shows a per-person **Debug tab** that explains
each person's classification (state, timers, speed, zone membership, transiting) —
so correctness can be verified by eye, one frame at a time.

## Components

### `config.py`
- `debug: bool = False` + `--debug`. Enables step mode and (always-on, cheap)
  per-person debug rows pushed to the dashboard.

### `queue/types.py`
- `PersonStatus` gains `debug: dict | None = None`.

### `queue/analyzer.py`
- After computing `in_line / in_pos / in_alt / transiting / speed_norm` for a
  person, build `dbg = {"speed": speed_norm, "transit": transiting,
  "line": in_line, "pos": in_pos, "reg": in_alt}` and attach it to **every**
  emitted `PersonStatus` for that person (normal, the debounce-limbo override, and
  the finalize-on-leave neutral). Cheap; populated always.

### `dashboard/state.py`, `metrics.py`, `server.py`
- `DashboardState.set_debug(frame, rows)` stores the latest `(frame, rows)`;
  `debug_snapshot()` returns it.
- `build_payload(snapshot, busy=(None,[]), debug=(None,[]))` adds a `debug` block:
  `{frame, rows}`. The server passes `state.debug_snapshot()`.

### `dashboard/page.py`
- A 5th tab **"Debug"** with a live table — one row per person:
  `ID · state · wait s · serve s · speed · line/POS/REG · transit` plus the frame
  number on top. Empty/"run with --debug" hint when no rows. Rendered from
  `/metrics`'s `debug` block each poll. (The charts stay cumulative.)

### `runner.py` — step mode + per-frame rows
- Build `rows` from `qresult.statuses` each frame: per person `{id, state, wait,
  serve, speed, line, pos, reg, transit}` (state = `transiting` / `serving-Mashgin`
  / `serving-REG` / `waiting` / `candidate NN%` / `out`).
- **Non-debug:** unchanged play loop, but also `dash_state.set_debug(idx, rows)`
  each frame so the Debug tab works live.
- **Debug:** keep the existing per-frame processing inline (it advances the
  pipeline/dashboard once per real frame); after it, append `(idx, jpeg(canvas),
  rows)` to a `deque(maxlen=300)` and enter an inner display loop:
  - `view` = frames back from latest (0 = newest). Show `buffer[-1-view]`, decoded,
    with a `DEBUG · frame N · PAUSED/PLAY` banner; call `set_debug(b_idx, b_rows)`
    so the tab matches the viewed frame (even when scrubbed back).
  - Keys (`waitKey(0)` when paused, `waitKey(30)` when playing): `q`/Esc quit;
    `→`/space → if `view>0` decrement (toward latest) else break to advance source;
    `←` → `view = min(view+1, len-1)` (review older); `c` play; `p` pause; on a
    play-timeout auto-advance (decrement view, or break at latest).
  - JPEG-encoding the buffer keeps ~300 frames cheap in memory.
- The pipeline/tracker/analyzer state only ever moves forward; `←` is review-only.

## Data flow
Each real frame: process → `canvas`, `rows` → buffer + `set_debug(viewed)`. The
dashboard polls `/metrics`; the Debug tab renders `debug.rows` for the viewed
frame; the cumulative charts/comparison reflect the latest processed frame.

## Testing
- `PersonStatus.debug` populated by the analyzer (a moving person shows
  `transit: True`, `line/pos/reg` flags correct).
- `metrics.build_payload` includes a `debug` block; `DashboardState.set_debug` /
  `debug_snapshot` round-trip.
- A tiny pure helper for the buffer index math (`view` clamping / "frames back")
  unit-tested; the GUI step loop verified manually.
- Existing suite stays green (`debug` defaults make all current paths unchanged).

## Out of scope
True random-access seek (replay to arbitrary frame); editing/annotating frames;
rewinding the cumulative charts.
