# Multi-contour zones + combined type-aware editor

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation
**Branch:** feat/realtime-pose

## Goal

Let a zone be **several disjoint polygons** (one zone, multiple contours), and let
one editor session author **both** the line zone and the POS zone — pressing `1`
draws line contours, `2` draws POS contours — saving two type-pure files. The
zone *type* still lives in the file/flag (`--zone` = line, `--pos-zone` = POS), so
the analyzer, runner, and busy code are untouched.

## Components

### `src/storepose/queue/zone.py` — `Zone` holds N polygons
- Internal `self.polygons: list[list[tuple[int, int]]]` (each a contour).
- **Back-compatible constructor:** `Zone([(x, y), ...])` (a flat list of points) is
  still a single-contour zone; `Zone([[(x,y),...], [(x,y),...]])` (a list of
  contours) is multi. Detection: the argument is nested when its first element's
  first item is itself a sequence. Empty input → no polygons.
- `Zone.from_polygons(polygons)` classmethod for explicit multi construction.
- `points` property (back-compat) returns the first contour (`[]` if none).
- `contains(pt)` → True if inside **any** contour (a contour with <3 points
  contributes nothing). `coverage(box, grid=7)` → fraction of the grid sample
  inside **any** contour (union). Both keep their current signatures, so the
  analyzer's waiting/serving membership tests are unchanged.
- `to_dict()` → `{"polygons": [[[x, y], ...], ...]}`. `from_dict()` reads the new
  `polygons` key, and falls back to the legacy `{"points": [...]}` (loaded as one
  contour) so existing zone files still load. `save`/`load` unchanged.

### `src/storepose/queue/zone_editor.py` — combined editor
`define_zones(source, line_path=None, pos_path=None, pos_only=False) -> dict[str, str]`
opens one frame and collects contours of two types:
- State: `line_contours`, `pos_contours`, an in-progress `current` contour, and a
  `mode` (`"line"` / `"pos"`, starting `"line"`; `pos_only` locks it to `"pos"`).
- Keys: **`1`** → line mode, **`2`** → POS mode (ignored when `pos_only`),
  left-click appends a point to `current`, **`n`** finishes `current` (needs ≥3
  pts) into the active type's list and starts a fresh contour, **`u`** undoes the
  last point of `current`, **`c`** clears everything, **`s`** saves, **`q`** quits.
- On save: finalize `current` (≥3 pts → active type). Write line contours to
  `line_path` (default `default_zone_path(source)`) and POS contours to `pos_path`
  (default `default_pos_zone_path(source)`) via `Zone.from_polygons(...).save(...)`
  — but only for a type that has ≥1 valid contour. Returns the saved paths
  (`{"line": ..., "pos": ...}`, omitting unsaved types). `pos_only` never writes
  the line file.
- Live overlay: finished line contours orange, POS contours azure, the in-progress
  contour in the active color with its points; the hint shows the active mode and
  keys.

The existing single-type `define_zone(source, out_path)` is kept (used by tests /
as a thin helper) but `main.py` routes through `define_zones`.

### `main.py`
- `--define-zone` → `define_zones(config.source, config.zone, config.pos_zone)`;
  print `Run with: --zone <line> --pos-zone <pos>` for whatever was saved.
- `--define-pos-zone` → `define_zones(config.source, pos_path=config.pos_zone,
  pos_only=True)`; print `Run with: --pos-zone <pos>`.

### `src/storepose/drawing.py`
`annotate_queue` iterates `zone.polygons` and `pos_zone.polygons`, filling +
outlining **each** contour (line orange, POS azure) instead of the single polygon
today. Factor a small `_draw_zone(canvas, zone, color)` helper that loops contours
(fill at 0.15 alpha for ≥3-pt contours, then polyline).

## Data flow / unchanged surface
`--zone` and `--pos-zone` still load one type-pure file each (now possibly
multi-contour). `QueueAnalyzer` calls `zone.contains` / `zone.coverage` — both now
union-aware — with **no analyzer or config changes**. Per-frame results,
waiting/serving logic, and the busy path are identical.

## Testing
`tests/queue/test_zone.py`:
- `Zone.from_polygons` with two disjoint squares: `contains` true for a point in
  either, false for a point in neither; `coverage` of a box over one square > 0.
- `to_dict` → `from_dict` round-trips multiple contours.
- Legacy `{"points": [...]}` loads as a one-contour zone (`contains` works).
- Back-compat: `Zone([(x,y), ...])` single-contour behavior unchanged.

`tests/test_drawing.py`:
- A two-contour zone draws pixels for both contours (sample a point near each).

The editor's GUI loop stays manually verified; its save step
(`Zone.from_polygons(contours).save(path)`) is covered by the `Zone` round-trip
tests.

## Out of scope
No per-contour type metadata in the file format (type stays per-file). No combined
single-file zone format. The dashboard (separate sub-project) is unaffected.
