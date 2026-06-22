"""Arrow-driven launcher TUI: pick a saved view, toggle flag columns, run it.

Thin curses shell over :mod:`storepose.launcher_core`. Rows are saved views
(``viewscripts/*.sh``); columns toggle run flags. Enter tears down curses and
``exec``s the chosen viewscript with the composed env + args forwarded to it.

Run via ``./video-run.sh`` (which execs ``python -m storepose.launcher``). Any
args are passed through to the launched run.
"""

from __future__ import annotations

import curses
import os
import sys

from .launcher_core import (
    COLUMN_LABELS,
    COLUMNS,
    Column,
    View,
    build_run,
    default_state,
    discover_views,
)

_VIEWSCRIPTS = "viewscripts"
_CALIB = "calib"
_ZONES = "zones"

_FLAG_W = 7        # width of each flag cell
_NAME_W_MAX = 34   # cap on the view-name column
_ARROW_W = 2       # left/right margin reserved for the ‹ › scroll arrows


def _col_window(col: int, col_off: int, n_cols: int, max_visible: int) -> int:
    """Return an updated first-visible-column index keeping ``col`` on screen.

    Pure horizontal-scroll math (no curses) so it is unit-testable: clamps the
    window to ``[0, n_cols - max_visible]`` and slides it just far enough that
    the selected ``col`` stays within ``[col_off, col_off + max_visible)``.
    """
    if max_visible <= 0 or max_visible >= n_cols:
        return 0
    if col < col_off:
        col_off = col
    elif col >= col_off + max_visible:
        col_off = col - max_visible + 1
    return max(0, min(col_off, n_cols - max_visible))


def _cell(view: View, state, column: Column) -> str:
    if column == Column.DASHBOARD:
        return "on" if state.dashboard else "·"
    if column == Column.DEBUG:
        return "on" if state.debug else "·"
    if column == Column.CONF:
        return "on" if state.conf else "·"
    if column == Column.SAVE:
        return "on" if state.save else "·"
    if column == Column.BLUR:
        return "on" if state.blur else "·"
    if column == Column.ALT:
        return "—" if not view.has_alt else ("on" if state.alt else "·")
    if column == Column.CALIB:
        return "—" if not view.has_calib else ("on" if state.calib else "·")
    if column == Column.STRATEGY:
        return "—" if not view.has_calib else state.strategy
    if column == Column.REID:
        return {"osnet-x1": "x1", "osnet-x025": "x025",
                "histogram": "hist", "off": "off"}[state.reid]
    return "?"


def _safe_add(stdscr, y: int, x: int, text: str, attr: int = 0) -> None:
    try:
        stdscr.addstr(y, x, text, attr)
    except curses.error:
        pass  # off-screen / too-small terminal: skip rather than crash


def _draw(stdscr, views, states, row, col, name_w, col_off, max_visible) -> None:
    _safe_add(stdscr, 0, 0, "storePose — pick a view, toggle columns, Enter to run",
              curses.A_BOLD)
    hx = 2 + name_w
    n_cols = len(COLUMNS)
    end = min(n_cols, col_off + max_visible)        # visible window [col_off, end)
    arrow_x = hx + (end - col_off) * _FLAG_W         # just past the last visible cell
    # header (only the visible column slice)
    _safe_add(stdscr, 2, 2, "view".ljust(name_w), curses.A_UNDERLINE)
    if col_off > 0:
        _safe_add(stdscr, 2, hx - 1, "‹", curses.A_BOLD)
    for vis, i in enumerate(range(col_off, end)):
        _safe_add(stdscr, 2, hx + vis * _FLAG_W,
                  COLUMN_LABELS[COLUMNS[i]].center(_FLAG_W), curses.A_UNDERLINE)
    if end < n_cols:
        _safe_add(stdscr, 2, arrow_x, "›", curses.A_BOLD)
    # rows
    for r, view in enumerate(views):
        y = 3 + r
        st = states[view.stem]
        marker = "> " if r == row else "  "
        _safe_add(stdscr, y, 0, marker + view.stem[:name_w].ljust(name_w),
                  curses.A_BOLD if r == row else 0)
        for vis, i in enumerate(range(col_off, end)):
            cx = hx + vis * _FLAG_W
            attr = curses.A_REVERSE if (r == row and i == col) else 0
            _safe_add(stdscr, y, cx, _cell(view, st, COLUMNS[i]).center(_FLAG_W), attr)
    # footer
    fy = 4 + len(views)
    _safe_add(stdscr, fy, 0,
              "↑/↓ view   ←/→ column   space toggle   enter run   q quit",
              curses.A_DIM)
    if end - col_off < n_cols:                       # only when some columns are hidden
        _safe_add(stdscr, fy + 1, 0,
                  f"columns {col_off + 1}–{end} of {n_cols}  (‹ › scroll)",
                  curses.A_DIM)


def _run_ui(stdscr, views):
    curses.curs_set(0)
    states = {v.stem: default_state(v) for v in views}
    name_w = min(_NAME_W_MAX, max((len(v.stem) for v in views), default=4))
    row, col = 0, 0
    col_off = 0
    while True:
        _, width = stdscr.getmaxyx()
        hx = 2 + name_w
        # columns that fit, leaving room for the ‹ › scroll arrows; >= 1
        max_visible = max(1, (width - hx - _ARROW_W) // _FLAG_W)
        col_off = _col_window(col, col_off, len(COLUMNS), max_visible)
        stdscr.erase()
        _draw(stdscr, views, states, row, col, name_w, col_off, max_visible)
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), 27):
            return None
        if key in (curses.KEY_UP, ord("k")):
            row = (row - 1) % len(views)
        elif key in (curses.KEY_DOWN, ord("j")):
            row = (row + 1) % len(views)
        elif key in (curses.KEY_LEFT, ord("h")):
            col = (col - 1) % len(COLUMNS)
        elif key in (curses.KEY_RIGHT, ord("l")):
            col = (col + 1) % len(COLUMNS)
        elif key == ord(" "):
            from .launcher_core import toggle
            v = views[row]
            states[v.stem] = toggle(v, states[v.stem], COLUMNS[col])
        elif key in (curses.KEY_ENTER, 10, 13):
            v = views[row]
            env, args = build_run(v, states[v.stem])
            return (v, env, args)


def _fallback(views):
    """Non-TTY: numbered prompt, run with default flags."""
    print("Select a view to run:")
    for i, v in enumerate(views, 1):
        print(f"  {i}) {v.stem}")
    try:
        raw = input("> ").strip()
    except EOFError:
        return None
    if not raw.isdigit() or not (1 <= int(raw) <= len(views)):
        print("invalid choice", file=sys.stderr)
        return None
    v = views[int(raw) - 1]
    env, args = build_run(v, default_state(v))
    return (v, env, args)


def _exec_run(view: View, env, extra_args, passthrough) -> None:
    script = str(view.script)
    if not os.access(script, os.X_OK):
        print(f"error: view script not executable: {script}", file=sys.stderr)
        raise SystemExit(1)
    full_env = {**os.environ, **env}
    cmd = [script, *extra_args, *passthrough]
    shown = " ".join(extra_args + passthrough) or "(defaults)"
    note = " STOREPOSE_NO_CALIB=1" if env.get("STOREPOSE_NO_CALIB") else ""
    print(f"==> running: {view.stem}  {shown}{note}")
    os.execvpe(script, cmd, full_env)


def main(argv: list[str] | None = None) -> int:
    passthrough = list(sys.argv[1:] if argv is None else argv)
    views = discover_views(_VIEWSCRIPTS, _CALIB, _ZONES)
    if not views:
        print(f"No view scripts in {_VIEWSCRIPTS}/. Create one first:  "
              f"./view-setup.sh -v <video>", file=sys.stderr)
        return 1
    if sys.stdin.isatty() and sys.stdout.isatty():
        chosen = curses.wrapper(_run_ui, views)
    else:
        chosen = _fallback(views)
    if chosen is None:
        return 0
    view, env, args = chosen
    _exec_run(view, env, args, passthrough)  # replaces the process on success
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
