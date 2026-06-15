# Launcher TUI — design

**Date:** 2026-06-15
**Status:** approved (brainstorming) → implementing

## Problem

`video-run.sh` picks a saved view via `fzf` or a numbered `select` menu, then runs
its `viewscripts/<stem>.sh`. It's pick-one-and-go: no way to tweak run flags
without editing the script or remembering CLI flags. We want an arrow-driven
table: scroll views, toggle per-view flag columns, Enter to run.

## Decisions (from brainstorming)

- **Surface:** upgrade the `video-run.sh` launcher (not the web dashboard or the
  OpenCV debug view).
- **Language:** Python `curses` (stdlib, no new deps). `video-run.sh` becomes a
  thin wrapper that `exec`s the launcher. Curses exits before the run starts.
- **Rows:** saved views = `viewscripts/*.sh` stems (same source as today).
- **Columns:** `dashboard` (on/off) · `debug` (on/off) · `conf` (on/off,
  `--conf` detector-confidence overlay) · `calib` (on/off) · `strategy` (cycle:
  auto→skewed→thirds→peak). `busy` stays always-on (baked into the view) and is
  **not** shown — the table lists only toggleable columns.
- **State:** reset each launch (no persistence file). Defaults: dashboard on,
  debug off, calib on iff `calib/<stem>.json` exists, strategy auto.
- **Launch model:** forward toggles to the existing viewscript (keep it the single
  source of truth for source path / zones / ports / log paths). The one flag a
  script can't un-bake — auto-`--calib` — is gated on an env var.

## Interaction

| key | action |
|-----|--------|
| `↑` / `↓` | move between views (rows) |
| `←` / `→` | move the selected column |
| `space` | toggle the selected column; `strategy` *cycles* auto→skewed→thirds→peak |
| `enter` | run the highlighted view with its current toggles |
| `q` / Esc | quit without running |

`calib`/`strategy` columns render disabled (`—`) for views with no calib file.

## Architecture

Two units, split so the logic is testable without a terminal.

### `src/storepose/launcher_core.py` — pure core (no curses, no I/O side effects)

- `Column` enum / order: `DASHBOARD, DEBUG, CALIB, STRATEGY`.
- `STRATEGY_CYCLE = ("auto", "skewed", "thirds", "peak")`.
- `discover_views(viewscripts_dir, calib_dir) -> list[View]` where
  `View = (stem, script_path, has_calib)`. Sorted by stem.
- `default_state(view) -> ColumnState` — dashboard=True, debug=False,
  calib=view.has_calib, strategy="auto".
- `toggle(state, column) -> ColumnState` — flips booleans; cycles strategy;
  no-op for calib/strategy when `not has_calib`.
- `build_run(view, state) -> tuple[dict[str,str], list[str]]` — returns
  `(env_overrides, extra_args)`:
  - `dashboard` off → `--no-dashboard`
  - `debug` on → `--debug`
  - `calib` off → `env STOREPOSE_NO_CALIB=1`
  - `strategy` != auto → `--busy-strategy <s>` (only when calib on)
  These are *appended* to the viewscript invocation; busy/zones/etc. stay in the
  script.

### `src/storepose/launcher.py` — thin curses shell

- Renders the table (header row of columns; one row per view; highlight selected
  cell). Footer shows the key legend.
- Maps keys to `launcher_core` calls.
- On Enter: tear down curses, then `os.execvpe` the viewscript with
  `extra_args` appended and `env_overrides` merged into `os.environ`.
- `python -m storepose.launcher [extra args]` — extra args pass through to the run
  (forwarded after the toggles), preserving today's `video-run.sh --no-dashboard`
  passthrough behavior.

### `video-run.sh`

Replace the fzf/select body with `exec uv run python -m storepose.launcher "$@"`.
Keep the `-h/--help` header.

### `view-setup.sh` (generated script change)

- Gate auto-calib on the env var and fix the `set -u` empty-array expansion:
  ```bash
  EXTRA=()
  [[ -z "${STOREPOSE_NO_CALIB:-}" && -f "$CALIB" ]] && EXTRA+=(--calib "$CALIB")
  uv run python main.py ... "${EXTRA[@]+"${EXTRA[@]}"}" "$@"
  ```
- Regenerate the two existing viewscripts (view2, Rock Hill) via
  `view-setup.sh -v <video> --no-run` so the launcher's calib/strategy toggles work.

## Data flow

```
launcher.py (curses)
  discover_views() -> rows
  per row: default_state()
  keys -> toggle()/move selection
  Enter -> build_run(view, state) -> (env, extra_args)
        -> exec viewscript with env + extra_args
            -> main.py runs (busy/zones baked in; toggles applied)
```

## Error handling

- **Not a TTY** (stdin/stdout piped): skip curses, fall back to a numbered
  prompt (today's behavior) and run with default flags.
- **Terminal too small** to draw the table: print a message and exit non-zero.
- **No viewscripts**: the existing "run view-setup first" hint, exit non-zero.
- **Selected view's script missing at exec time**: clear error, exit non-zero.

## Testing

`tests/test_launcher_core.py`:
- `discover_views`: finds stems, sets `has_calib` correctly (calib present/absent).
- `default_state`: calib defaults to `has_calib`; strategy `auto`; dashboard on.
- `toggle`: boolean flips; strategy cycles through all four and wraps; calib +
  strategy are no-ops when `not has_calib`.
- `build_run`: exact `(env, extra_args)` for representative states (dashboard off,
  debug on, calib off → env, strategy set → flag only when calib on).

Curses rendering/input is not unit-tested (kept thin).

## Out of scope

- Persisting toggle state between launches (chose reset-each-launch).
- A `busy` on/off toggle (busy stays always-on per the view).
- Editing zones / calibrating from the launcher (separate commands).
- Recomposing the run command in Python (we forward to the viewscript instead).
