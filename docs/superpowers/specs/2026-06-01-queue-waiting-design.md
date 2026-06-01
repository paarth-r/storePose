# storePose — Queue / "Waiting in Line" Detection

**Date:** 2026-06-01
**Status:** Approved

## Goal

Tell whether each tracked person is **waiting in line**, then derive (1) a
per-person waiting flag + dwell timer, (2) a live **queue count**, and (3)
**per-person wait time**. Uses a fixed per-camera queue **zone**.

## Decisions (locked)

| Decision        | Choice                                                          |
|-----------------|-----------------------------------------------------------------|
| Foundation      | Per-person waiting flag (#1); count (#2) and wait time (#3) derive from it. |
| Zone            | Polygon ROI defined once per fixed camera, stored as JSON.       |
| "Waiting"       | Foot point inside zone AND low speed, sustained past a debounce. |
| Speed source    | Foot-point displacement of the **smoothed (Kalman) box** between frames, divided by `dt` and box height → body-heights/sec (framerate- and perspective-robust). Light EMA. (Simpler/cleaner than plumbing Kalman velocity through `TrackedPerson`.) |
| Zone authoring  | Interactive click-to-draw editor, saved to `zones/<name>.json`.  |

## Data flow

```
tracker → list[TrackedPerson]
        → QueueAnalyzer.update(people, dt) → QueueResult(statuses, count, completed)
        → annotate_tracked (boxes/skeletons) → annotate_queue (zone + WAIT tags + count)
        → display / save ; completed rows appended to --wait-log CSV
```

## Waiting state machine (per track id) — this is #1

Per person each frame: `foot = ((x1+x2)/2, y2)`; `in_zone = zone.contains(foot)`;
`speed_bh_s = EMA(|foot - prev_foot| / dt / box_height)`; `slow = speed_bh_s <
wait_speed`. Condition `in_cond = in_zone and slow`.

```
NOT_WAITING ──(in_cond held ≥ enter_seconds)──▶ WAITING
WAITING ──(¬in_cond held ≥ exit_seconds, OR track id disappears)──▶ ended
```

While WAITING, accumulate `wait_seconds` (includes the enter debounce). A
monotonic clock (sum of `dt`) timestamps entry/exit.

- **#2 count** = number of present persons currently WAITING.
- **#3 wait time** = `wait_seconds` when a person leaves WAITING → emitted in
  `QueueResult.completed` as `(id, entered_s, exited_s, wait_seconds)` and
  appended to the CSV if `--wait-log` is set.

A waiting person whose track id vanishes is finalized immediately (the tracker
already gave occlusion grace via coasting before the id was dropped).

## Modules

```
src/storepose/queue/
  __init__.py
  zone.py        Zone(points): contains(point) via cv2.pointPolygonTest; load/save JSON
  types.py       PersonStatus(id, waiting, wait_seconds); QueueResult(statuses, count, completed)
  analyzer.py    QueueAnalyzer(zone, wait_speed, enter_seconds, exit_seconds): update(people, dt)
  zone_editor.py define_zone(source, out_path): click polygon on a frame, save JSON (GUI)
```

Changes: `drawing.py` add `annotate_queue(canvas, people, result, zone, config)`
(faint polygon, `WAIT n.n s` tags on waiting people, `in line: N` header);
`runner.py` load zone + run analyzer + write wait log; `config.py` flags;
`main.py` route `--define-zone` to the editor.

## Config flags

| Flag                  | Default | Meaning                                         |
|-----------------------|---------|-------------------------------------------------|
| `--zone PATH`         | —       | Queue-zone JSON to load; enables waiting detection. |
| `--define-zone`       | —       | Launch the interactive zone editor and exit.    |
| `--wait-speed`        | `0.15`  | Max speed (body-heights/sec) to count as "slow".|
| `--wait-enter-seconds`| `1.5`   | In-zone+slow time before WAITING is declared.    |
| `--wait-exit-seconds` | `2.0`   | Out-of-condition time before WAITING ends.        |
| `--wait-log PATH`     | —       | Append completed waits as CSV rows.              |

## Testing

Pure/TDD: `Zone.contains` (in/out/on, <3 points), JSON round-trip;
`QueueAnalyzer` — stationary-in-zone promotes to WAITING after `enter_seconds`;
walk-through (fast) never waits; leaving zone ends WAITING after `exit_seconds`
and emits a completed record; id disappearance finalizes; count is correct.
Driven by synthetic `TrackedPerson` sequences, no models. Manual: zone editor
GUI and the live overlay on a store clip.

## Trade-offs

The zone *is* the definition of "the line," so it must be drawn tightly around
the queue area. Speed from smoothed boxes is robust but slightly lagged — fine
for a dwell-based decision with debounces.
