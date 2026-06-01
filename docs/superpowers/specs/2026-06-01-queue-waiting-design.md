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

Per person each frame we pick a **ground point** (for speed) and compute
**in_zone** as an **OR** of two signals so a held position isn't lost when one
drops out:
- **Ankle:** if a COCO ankle keypoint (15/16) clears `kpt_thr`, `ankle_inside =
  zone.contains(visible-ankle midpoint)` — precise, immune to box padding (carts)
  and bottom-edge occlusion. The midpoint is also the ground point for speed.
- **Coverage:** `covered = zone.coverage(foot_box) >= zone_coverage`, the fraction
  of the **foot region** inside the polygon (grid sample), where `foot_box` is the
  bottom `zone_foot_band` (default 0.3) of the box. Only the foot strip is used,
  not the whole box, because a standing person's box is mostly torso/head that
  projects above a floor zone — whole-box coverage would under-count. Robust when
  feet leave frame / are occluded.

`in_zone = ankle_inside or covered`.

**Speed** is measured from the stable Kalman box-bottom (never the ankle), so an
ankle flashing in/out can't create a fake speed spike that drops `in_cond`.

**Grace:** a brief loss of `in_cond` does not reset progress. `out_streak`
accumulates only while `in_cond` is false; a candidate's `in_frames` resets (and
a wait ends) only once `out_streak >= exit_seconds`. So a flickering ankle or a
momentary coverage dip holds the inclusion counter / keeps the wait timer
running rather than restarting it.

Then `speed_bh_s = EMA(|ground - prev_ground| / dt / box_height)`; `slow =
speed_bh_s < wait_speed`; `in_cond = in_zone and slow`.

```
NOT_WAITING ──(in_cond)──▶ CANDIDATE (progress = in_frames / enter_frames)
CANDIDATE   ──(in_frames ≥ enter_frames, default 5)──▶ WAITING
CANDIDATE   ──(¬in_cond)──▶ NOT_WAITING  (in_frames resets to 0)
WAITING ──(¬in_cond held ≥ exit_seconds, OR track id disappears)──▶ ended
```

Inclusion is gated on **consecutive frames** (`enter_frames`, default 5), which
also drives the candidate fill animation's `progress`. While WAITING, accumulate
`wait_seconds` from inclusion; a monotonic clock (sum of `dt`) timestamps
entry/exit. `PersonStatus` carries `waiting`, `candidate`, `progress`,
`wait_seconds`.

**Visualization:** candidate → amber "sheer" fill rising over the box
proportional to `progress` (flood animation) + join `%`; waiting → translucent
green overlay on the box + `WAIT n.n s`. Boxes are the tracker's Kalman-smoothed
output (no second filter needed).

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
