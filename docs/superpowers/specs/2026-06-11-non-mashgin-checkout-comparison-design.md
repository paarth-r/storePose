# Non-Mashgin checkout zone + Mashgin-vs-traditional comparison

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation
**Branch:** feat/realtime-pose

## Goal

Add a third zone — a **non-Mashgin checkout** (traditional register) — alongside
the line and the Mashgin POS. Track serving time at each checkout separately and
show, side by side, the **average serve time per person** at the Mashgin kiosk
(**green**) vs the non-Mashgin register (**red**), plus the difference — the
headline that demonstrates Mashgin is faster. In the live overlay the Mashgin
checkout is drawn **green** and the non-Mashgin **red**.

## Components

### Config (`config.py`)
- `alt_zone: str | None = None` + `--alt-zone PATH` (the non-Mashgin checkout).
- `define_alt_zone: bool = False` + `--define-alt-zone`.
- `--alt-zone` requires `--zone` (line) and is most meaningful with `--pos-zone`.

### Zone editor (`zone_editor.py`, `main.py`)
`define_zones` gains a third draw mode: key **`3` = non-Mashgin** (drawn red), in
addition to `1`=line, `2`=Mashgin POS. Signature:
`define_zones(source, line_path, pos_path, alt_path, only=None)` where `only ∈
{None,"pos","alt"}`. `s` saves whichever types have ≥1 contour. New
`default_alt_zone_path(source)` → `zones/<stem>_alt.json`. `main.py`:
`--define-zone` → all three; `--define-pos-zone` → `only="pos"`;
`--define-alt-zone` → `only="alt"`.

### Analyzer (`queue/analyzer.py`) — serving by checkout
`__init__` gains `alt_zone: Zone | None = None`. Per frame:
```
in_pos  = pos_zone and in_zone(pos)          # Mashgin
in_alt  = alt_zone and in_zone(alt)          # non-Mashgin
in_line = in_zone(line)
if in_pos or in_alt: in_line = False         # a checkout beats the line (POS-priority generalized)
which = "mashgin" if in_pos else "other" if in_alt else None
in_check = which is not None
```
`_VisitState`: replace `reached_pos: bool` with `checkout: str | None`
(`"mashgin"|"other"`), set when a checkout is first reached and updated to the
current checkout while serving; `serving_seconds` accrues whenever `in_check`. The
existing `pos_enter_frames` debounce gates `WAITING → SERVING` for **either**
checkout; the "in a checkout leaves the line count instantly" rule applies to
`in_check` (was `in_pos`). Transitions mirror today's machine with `in_pos`
replaced by `in_check` and the checkout recorded.

`_finalize` → `CompletedWait.outcome`:
- `"served"` if `checkout == "mashgin"`,
- `"served_other"` if `checkout == "other"`,
- `"served"` if **no checkout zones are configured** (preserves single-zone /
  busy behavior),
- else `"abandoned"`.
`serving_seconds` is the time at whichever checkout they used.

### Types (`queue/types.py`)
- `PersonStatus`: add `serving_other: bool = False`. The analyzer sets
  `serving = state=="serving" and checkout=="mashgin"`,
  `serving_other = state=="serving" and checkout=="other"`.
- `QueueResult`: add `serving_other_count: int = 0`.
- `CompletedWait`: no new field — the `outcome` string carries `served_other`.

### Drawing (`drawing.py`)
- `POS_COLOR` → **green** (Mashgin); new `ALT_COLOR` = **red** (non-Mashgin).
- Draw `pos_zone` green and `alt_zone` red (each multi-contour via `_draw_zone`).
- A serving person: green fill + `POS n.n s` (Mashgin) when `s.serving`; red fill
  + `REG n.n s` (non-Mashgin) when `s.serving_other`.
- Header: `in line: N   at POS: M   at REG: K` (the `at REG` shown when an alt zone
  is active / `serving_other_count` can be > 0).
- Bottom panel lists Mashgin people (green) and non-Mashgin people (red).

### Runner (`runner.py`)
Load `alt_zone` when set; pass to `QueueAnalyzer` and `annotate_queue`. Busy is fed
by **both** `served` and `served_other` (line throughput counts either checkout):
`if c.outcome in ("served", "served_other")`. Print a note if `--alt-zone` without
`--zone`. The dashboard observe call also reports `serving_other_count`.

### Dashboard metrics (`dashboard/metrics.py`)
- `checkout_stats(visits)` → `{mashgin_avg, mashgin_n, other_avg, other_n,
  delta}` where the avgs are mean `serving_seconds` over `served` vs `served_other`
  visits and `delta = other_avg - mashgin_avg` (seconds Mashgin saves).
- `checkout_series(visits, window)` → `{t, mashgin_ma, other_ma}` trailing-window
  mean serve time per checkout over time.
- `build_payload` gains a `checkouts` block (stats + series).

### Dashboard page (`dashboard/page.py`)
- A **comparison block** near the hero (shown only when `checkouts.other_n > 0`):
  two tiles — **Mashgin** (green, avg serve s) vs **Non-Mashgin** (red, avg serve
  s) — and the delta (e.g. "Mashgin 18.4s faster"). The dashboard's existing
  navy/teal occupancy theme is unchanged; green/red is reserved for this head-to-
  head.
- A **4th tab "Checkouts"** charting `mashgin_ma` (green) vs `other_ma` (red) over
  time, same interactive ECharts treatment.

## Data flow
`runner` loads three zones → `analyzer.update` emits per-person status (waiting /
serving(mashgin) / serving_other) + completed visits with `served` / `served_other`
/ `abandoned`; the runner feeds the dashboard and wait-log. `metrics.build_payload`
derives the checkout comparison from the visit outcomes — no new state buffer
needed (visits already carry `serving_seconds` + `outcome`).

## Testing
- analyzer: a person in the alt zone → `serving_other` true, `serving` false,
  `serving_other_count==1`; completes as `outcome=="served_other"` with
  `serving_seconds>0`; Mashgin person → `outcome=="served"`; priority
  (in both pos+alt → mashgin); single-zone completion still `served`.
- types: `PersonStatus.serving_other` / `QueueResult.serving_other_count` defaults.
- metrics: `checkout_stats` avgs + delta over mixed `served`/`served_other`/
  `abandoned`; `checkout_series` split by outcome; `build_payload` has `checkouts`.
- config: `--alt-zone` / `--define-alt-zone` parse; `default_alt_zone_path`.
- drawing: alt zone draws red; a `serving_other` person draws a red tag; header
  shows `at REG`.

## Out of scope
Re-identifying a person across both checkouts in one visit (assume one checkout per
visit; the machine records the last checkout occupied). Party grouping. Re-theming
the dashboard's live occupancy colors (stays navy/teal).
