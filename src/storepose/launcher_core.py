"""Pure logic for the view launcher TUI (no curses, no side effects).

Kept separate from the curses shell (``launcher.py``) so the model — discovering
views, default/toggled column state, and the env + args a run needs — is unit
testable without a terminal. See
``docs/superpowers/specs/2026-06-15-launcher-tui-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path

from .busy.types import BUSY_STRATEGIES

# strategy column cycles through "auto" (use the calib file's default) then the
# explicit strategies.
STRATEGY_CYCLE = ("auto", *BUSY_STRATEGIES)


class Column(IntEnum):
    DASHBOARD = 0
    DEBUG = 1
    CONF = 2
    CALIB = 3
    STRATEGY = 4


COLUMNS = (Column.DASHBOARD, Column.DEBUG, Column.CONF, Column.CALIB, Column.STRATEGY)
COLUMN_LABELS = {
    Column.DASHBOARD: "dash",
    Column.DEBUG: "debug",
    Column.CONF: "conf",
    Column.CALIB: "calib",
    Column.STRATEGY: "strategy",
}


@dataclass(frozen=True)
class View:
    """A saved view: its viewscript and whether a calib file exists for it."""

    stem: str
    script: Path
    has_calib: bool


@dataclass(frozen=True)
class ColumnState:
    """Toggle state for one view's columns."""

    dashboard: bool = True
    debug: bool = False
    conf: bool = False
    calib: bool = False
    strategy: str = "auto"


def discover_views(viewscripts_dir: str | Path, calib_dir: str | Path) -> list[View]:
    """Find saved views (``viewscripts/*.sh``), flagging which have a calib file."""
    vp = Path(viewscripts_dir)
    cp = Path(calib_dir)
    views: list[View] = []
    for script in sorted(vp.glob("*.sh")):
        stem = script.stem
        views.append(View(stem, script, (cp / f"{stem}.json").is_file()))
    return views


def default_state(view: View) -> ColumnState:
    """Per-launch defaults: dashboard on, debug off, calib on iff a file exists."""
    return ColumnState(
        dashboard=True, debug=False, conf=False,
        calib=view.has_calib, strategy="auto",
    )


def toggle(view: View, state: ColumnState, column: Column) -> ColumnState:
    """Return a new state with ``column`` toggled (strategy cycles).

    ``calib`` and ``strategy`` are no-ops for a view with no calib file.
    """
    if column == Column.DASHBOARD:
        return replace(state, dashboard=not state.dashboard)
    if column == Column.DEBUG:
        return replace(state, debug=not state.debug)
    if column == Column.CONF:
        return replace(state, conf=not state.conf)
    if not view.has_calib:
        return state  # calib + strategy disabled without a calib file
    if column == Column.CALIB:
        return replace(state, calib=not state.calib)
    if column == Column.STRATEGY:
        i = STRATEGY_CYCLE.index(state.strategy)
        return replace(state, strategy=STRATEGY_CYCLE[(i + 1) % len(STRATEGY_CYCLE)])
    return state


def build_run(view: View, state: ColumnState) -> tuple[dict[str, str], list[str]]:
    """Compose ``(env_overrides, extra_args)`` to forward to the viewscript.

    busy/zones/ports stay baked in the script; these only express the toggles.
    """
    env: dict[str, str] = {}
    args: list[str] = []
    if not state.dashboard:
        args.append("--no-dashboard")
    if state.debug:
        args.append("--debug")
    if state.conf:
        args.append("--conf")
    if not state.calib:
        env["STOREPOSE_NO_CALIB"] = "1"  # suppress the script's auto --calib
    elif state.strategy != "auto":
        args += ["--busy-strategy", state.strategy]
    return env, args
