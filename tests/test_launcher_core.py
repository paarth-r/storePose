from pathlib import Path

from storepose.launcher_core import (
    STRATEGY_CYCLE,
    Column,
    ColumnState,
    View,
    build_run,
    default_state,
    discover_views,
)


def _make_views(tmp_path, stems_with_calib):
    vs = tmp_path / "viewscripts"
    cb = tmp_path / "calib"
    vs.mkdir()
    cb.mkdir()
    for stem, has_calib in stems_with_calib:
        (vs / f"{stem}.sh").write_text("#!/usr/bin/env bash\n")
        if has_calib:
            (cb / f"{stem}.json").write_text("{}")
    return vs, cb


# --- discover_views ------------------------------------------------------

def test_discover_views_sorted_with_calib_flag(tmp_path):
    vs, cb = _make_views(tmp_path, [("bbb", True), ("aaa", False)])
    views = discover_views(vs, cb)
    assert [v.stem for v in views] == ["aaa", "bbb"]   # sorted
    assert [v.has_calib for v in views] == [False, True]


def test_discover_views_empty(tmp_path):
    vs, cb = _make_views(tmp_path, [])
    assert discover_views(vs, cb) == []


# --- default_state -------------------------------------------------------

def test_default_state_calib_follows_file_presence():
    with_calib = View("x", Path("x.sh"), has_calib=True)
    without = View("y", Path("y.sh"), has_calib=False)
    assert default_state(with_calib) == ColumnState(
        dashboard=True, debug=False, calib=True, strategy="auto")
    assert default_state(without).calib is False


# --- toggle --------------------------------------------------------------

V = View("x", Path("x.sh"), has_calib=True)
V_NO = View("y", Path("y.sh"), has_calib=False)


def test_toggle_booleans_flip():
    s = default_state(V)
    assert toggle_(s, Column.DASHBOARD).dashboard is False
    assert toggle_(s, Column.DEBUG).debug is True
    assert toggle_(s, Column.CALIB).calib is False


def test_toggle_strategy_cycles_and_wraps():
    s = default_state(V)
    seen = [s.strategy]
    for _ in range(len(STRATEGY_CYCLE)):
        s = toggle_(s, Column.STRATEGY)
        seen.append(s.strategy)
    assert seen == ["auto", "skewed", "thirds", "peak", "auto"]


def test_toggle_conf_flips_even_without_calib_file():
    assert toggle_(default_state(V), Column.CONF).conf is True
    assert toggle_(default_state(V_NO), Column.CONF, view=V_NO).conf is True


def test_toggle_calib_and_strategy_noop_without_calib_file():
    s = default_state(V_NO)
    assert toggle_(s, Column.CALIB, view=V_NO) == s
    assert toggle_(s, Column.STRATEGY, view=V_NO) == s
    # but dashboard/debug/conf still work
    assert toggle_(s, Column.DEBUG, view=V_NO).debug is True
    assert toggle_(s, Column.CONF, view=V_NO).conf is True


def toggle_(state, column, view=V):
    from storepose.launcher_core import toggle
    return toggle(view, state, column)


# --- build_run -----------------------------------------------------------

def test_build_run_defaults_calibrated_view():
    # dashboard on, debug off, calib on, strategy auto -> nothing extra
    env, args = build_run(V, default_state(V))
    assert env == {}
    assert args == []


def test_build_run_dashboard_off_and_debug_on():
    s = ColumnState(dashboard=False, debug=True, calib=True, strategy="auto")
    env, args = build_run(V, s)
    assert env == {}
    assert args == ["--no-dashboard", "--debug"]


def test_build_run_calib_off_sets_env_no_strategy():
    s = ColumnState(dashboard=True, debug=False, calib=False, strategy="peak")
    env, args = build_run(V, s)
    assert env == {"STOREPOSE_NO_CALIB": "1"}
    assert "--busy-strategy" not in args  # strategy irrelevant when calib off


def test_build_run_strategy_flag_when_calib_on():
    s = ColumnState(dashboard=True, debug=False, calib=True, strategy="peak")
    env, args = build_run(V, s)
    assert env == {}
    assert args == ["--busy-strategy", "peak"]


def test_build_run_conf_flag():
    s = ColumnState(dashboard=True, debug=False, conf=True, calib=True, strategy="auto")
    env, args = build_run(V, s)
    assert env == {}
    assert args == ["--conf"]
