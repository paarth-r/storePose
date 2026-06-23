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

# reid column cycles through the OSNet sizes, the histogram, then off. osnet-x025
# is first because it is the app default (emitted implicitly by build_run).
REID_CYCLE = ("osnet-x025", "osnet-x1", "histogram", "off")


class Column(IntEnum):
    DASHBOARD = 0
    DEBUG = 1
    CONF = 2
    SAVE = 3
    BLUR = 4
    ALT = 5
    CALIB = 6
    STRATEGY = 7
    REID = 8
    DRIFT = 9


COLUMNS = (Column.DASHBOARD, Column.DEBUG, Column.CONF, Column.SAVE,
           Column.BLUR, Column.ALT, Column.CALIB, Column.STRATEGY, Column.REID,
           Column.DRIFT)
COLUMN_LABELS = {
    Column.DASHBOARD: "dash",
    Column.DEBUG: "debug",
    Column.CONF: "conf",
    Column.SAVE: "save",
    Column.BLUR: "blur",
    Column.ALT: "alt",
    Column.CALIB: "calib",
    Column.STRATEGY: "strategy",
    Column.REID: "reid",
    Column.DRIFT: "drift",
}


@dataclass(frozen=True)
class View:
    """A launchable view.

    A *viewscript* view has ``script`` set (run the generated viewscript). A
    *chunk* view has ``source`` set (run main.py on that clip) and ``zone_stem``
    naming the parent source whose zones/calib it shares. ``has_calib``/``has_alt``
    flag which files exist for that stem.
    """

    stem: str
    script: Path | None
    has_calib: bool
    has_alt: bool = False
    source: Path | None = None
    zone_stem: str | None = None


@dataclass(frozen=True)
class ColumnState:
    """Toggle state for one view's columns."""

    dashboard: bool = True
    debug: bool = False
    conf: bool = False
    save: bool = False
    blur: bool = True
    alt: bool = True
    calib: bool = False
    strategy: str = "auto"
    reid: str = "osnet-x025"
    drift: bool = False  # predictive drift off by default; on emits --predict-drift


def discover_views(viewscripts_dir: str | Path, calib_dir: str | Path,
                   zones_dir: str | Path | None = None) -> list[View]:
    """Find saved views (``viewscripts/*.sh``), flagging which have calib / alt files."""
    vp = Path(viewscripts_dir)
    cp = Path(calib_dir)
    zp = Path(zones_dir) if zones_dir is not None else None
    views: list[View] = []
    for script in sorted(vp.glob("*.sh")):
        stem = script.stem
        has_alt = zp is not None and (zp / f"{stem}_alt.json").is_file()
        views.append(View(stem, script, (cp / f"{stem}.json").is_file(), has_alt))
    return views


def chunk_source_stem(path: str | Path) -> str | None:
    """Return the parent-source stem for a chunk clip (``.../chunks/<SRC>/x.mp4``
    -> ``SRC``), else ``None``. Chunks share their source's zones/calib."""
    p = Path(path)
    if p.parent.parent.name == "chunks":
        return p.parent.name
    return None


def discover_chunks(videos_dir: str | Path, zones_dir: str | Path,
                    calib_dir: str | Path) -> list[View]:
    """Find chunk clips (``<videos>/**/chunks/<SRC>/partNN.mp4``) as runnable
    views, resolving zones/calib by the parent source ``SRC`` so one zone set
    covers every chunk of a source."""
    vp, zp, cp = Path(videos_dir), Path(zones_dir), Path(calib_dir)
    views: list[View] = []
    for mp4 in sorted(vp.glob("**/chunks/*/*.mp4")):
        src = mp4.parent.name
        views.append(View(
            stem=f"{src}/{mp4.stem}",
            script=None,
            has_calib=(cp / f"{src}.json").is_file(),
            has_alt=(zp / f"{src}_alt.json").is_file(),
            source=mp4,
            zone_stem=src,
        ))
    return views


def _common_toggle_args(state: ColumnState) -> list[str]:
    """Run flags shared by viewscript and chunk launches (not zone/calib)."""
    args: list[str] = []
    if not state.dashboard:
        args.append("--no-dashboard")
    if state.debug:
        args.append("--debug")
    if state.conf:
        args.append("--conf")
    if state.save:
        args.append("--save-mp4")
    if not state.blur:
        args.append("--no-blur-faces")
    if state.reid == "off":
        args.append("--no-reid")
    elif state.reid != "osnet-x025":  # osnet-x025 is the app default; emit nothing
        args += ["--reid-backend", state.reid]
    if state.drift:
        args.append("--predict-drift")
    return args


def build_chunk_args(view: View, state: ColumnState, zones_dir: str | Path,
                     calib_dir: str | Path) -> tuple[dict[str, str], list[str]]:
    """Full ``main.py`` args to run a chunk view, with zones/calib resolved from
    the parent source stem (so all chunks of a source share one zone set)."""
    zp, cp = Path(zones_dir), Path(calib_dir)
    stem = view.zone_stem
    args: list[str] = ["--source", str(view.source)]
    zone = zp / f"{stem}.json"
    if zone.is_file():
        args += ["--zone", str(zone), "--busy"]
        for suffix, flag in (("_pos", "--pos-zone"), ("_alt", "--alt-zone"),
                             ("_blur", "--blur-zone")):
            f = zp / f"{stem}{suffix}.json"
            if f.is_file() and not (suffix == "_alt" and not state.alt):
                args += [flag, str(f)]
    if view.has_calib and state.calib:
        args += ["--calib", str(cp / f"{stem}.json")]
        if state.strategy != "auto":
            args += ["--busy-strategy", state.strategy]
    args += _common_toggle_args(state)
    return {}, args


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
    if column == Column.SAVE:
        return replace(state, save=not state.save)
    if column == Column.BLUR:
        return replace(state, blur=not state.blur)
    if column == Column.ALT:
        return replace(state, alt=not state.alt) if view.has_alt else state
    if column == Column.REID:
        i = REID_CYCLE.index(state.reid)
        return replace(state, reid=REID_CYCLE[(i + 1) % len(REID_CYCLE)])
    if column == Column.DRIFT:
        return replace(state, drift=not state.drift)
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
    if state.save:
        args.append("--save-mp4")
    if not state.blur:
        args.append("--no-blur-faces")
    if not state.alt:
        args.append("--no-alt")
    if state.reid == "off":
        args.append("--no-reid")
    elif state.reid != "osnet-x025":  # osnet-x025 is the app default; emit nothing
        args += ["--reid-backend", state.reid]
    if state.drift:  # predictive drift is off by default; opt in
        args.append("--predict-drift")
    if not state.calib:
        env["STOREPOSE_NO_CALIB"] = "1"  # suppress the script's auto --calib
    elif state.strategy != "auto":
        args += ["--busy-strategy", state.strategy]
    return env, args
