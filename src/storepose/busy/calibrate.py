"""Per-view busy calibration: infer Low/Med/High bands from a clip.

Runs the pipeline over a calibration clip (headless by default), collects the
busy metric's per-sub-window distribution, and derives three candidate band sets
from it. The three are written together so a run can switch between them at
run time. See ``docs/superpowers/specs/2026-06-15-busy-calibration-design.md``.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from statistics import median

from ..config import AppConfig
from .aggregator import BusyAggregator
from .types import BUSY_STRATEGIES, DEFAULT_BUSY_STRATEGY, BusyThresholds

_DEFAULT_FPS = 30.0
_DEFAULT_SUBWINDOW = 30.0

# Tunable cut points for each strategy.
SKEWED_PCTS = (60.0, 85.0)   # busy is rare
THIRDS_PCTS = (100.0 / 3.0, 200.0 / 3.0)  # even terciles
PEAK_FRACS = (0.30, 0.70)    # fractions of peak occupancy

# Auto-default selection: median/peak occupancy at or above this means the line
# rarely empties (congested clip, no quiet baseline) -> percentile cuts mis-anchor,
# so prefer `peak`. Below it the clip has a real low baseline -> `skewed`.
SITS_FULL_RATIO = 0.5


def pick_default_strategy(values: list[float]) -> tuple[str, float]:
    """Choose the default strategy from the clip's occupancy shape.

    Returns ``(strategy, fill_ratio)`` where ``fill_ratio = median/peak``. A high
    ratio means the line sits full (use ``peak``); a low ratio means it empties
    out and percentile cuts are trustworthy (use ``skewed``). Empty/flat clips
    fall back to :data:`DEFAULT_BUSY_STRATEGY`.
    """
    vals = [float(v) for v in values]
    peak = max(vals) if vals else 0.0
    if peak <= 0:
        return DEFAULT_BUSY_STRATEGY, 0.0
    ratio = median(vals) / peak
    return ("peak" if ratio >= SITS_FULL_RATIO else "skewed"), ratio


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list. ``q`` in ``[0, 100]``."""
    if not sorted_vals:
        return 0.0
    rank = int(math.ceil(q / 100.0 * len(sorted_vals))) - 1
    rank = max(0, min(rank, len(sorted_vals) - 1))
    return float(sorted_vals[rank])


def _ordered(a: float, b: float) -> dict[str, float]:
    """Return ``{low_max, medium_max}`` with ``medium_max >= low_max`` guaranteed."""
    lo, hi = (a, b) if a <= b else (b, a)
    return {"low_max": round(lo, 4), "medium_max": round(hi, 4)}


def compute_strategies(values: list[float]) -> dict[str, dict[str, float]]:
    """Derive the three band sets from a metric sample distribution.

    ``values`` is one metric value per sub-window. Empty/degenerate input yields
    all-zero bands (still valid, non-decreasing).
    """
    vals = sorted(float(v) for v in values)
    peak = vals[-1] if vals else 0.0
    return {
        "skewed": _ordered(_percentile(vals, SKEWED_PCTS[0]),
                           _percentile(vals, SKEWED_PCTS[1])),
        "thirds": _ordered(_percentile(vals, THIRDS_PCTS[0]),
                           _percentile(vals, THIRDS_PCTS[1])),
        "peak": _ordered(PEAK_FRACS[0] * peak, PEAK_FRACS[1] * peak),
    }


def calib_path(stem: str) -> Path:
    return Path("calib") / f"{stem}.json"


def write_calib(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_calib(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def thresholds_from_calib(calib: dict, strategy: str,
                          hysteresis: float = 0.0) -> BusyThresholds:
    """Build :class:`BusyThresholds` for one strategy from a loaded calib dict."""
    if strategy not in calib.get("strategies", {}):
        raise ValueError(
            f"strategy {strategy!r} not in calib (have "
            f"{sorted(calib.get('strategies', {}))})"
        )
    band = calib["strategies"][strategy]
    return BusyThresholds(
        metric=calib.get("metric", "occupancy_p90"),
        low_max=float(band["low_max"]),
        medium_max=float(band["medium_max"]),
        hysteresis=hysteresis,
    )


def calibrate(config: AppConfig) -> dict:
    """Run a calibration pass over ``config.source`` and write ``calib/<stem>.json``.

    Headless unless ``config.verbose`` (which shows an annotated preview window).
    Returns the written payload.
    """
    # Imported here to avoid a circular import (runner imports busy.*).
    from ..pipeline import PosePipeline
    from ..runner import build_analyzer, build_tracker
    from ..video_source import VideoSource

    if not config.zone:
        raise ValueError("--calibrate needs --zone (the line zone)")
    if config.busy_metric == "mean_wait":
        raise ValueError("calibration needs an occupancy metric, not mean_wait")

    sub = config.busy_subwindow or _DEFAULT_SUBWINDOW
    verbose = config.verbose

    print(f"Calibrating {config.source} (metric={config.busy_metric}, "
          f"sub-window={sub:g}s, headless={not verbose})...")
    pipeline = PosePipeline(config)

    if verbose:
        import cv2  # local: only needed for the preview window

    with VideoSource(config.source) as source:
        base_fps = source.fps or _DEFAULT_FPS
        dt = 1.0 / base_fps
        tracker = build_tracker(config, base_fps)
        _zone, analyzer, _pos, _alt = build_analyzer(config)
        agg = BusyAggregator(
            BusyThresholds(metric=config.busy_metric),
            window_seconds=config.busy_window,
            sub_window_seconds=sub,
        )

        total = source.frame_count
        started = time.perf_counter()
        clock = 0.0
        n = 0
        for frame in source:
            clock += dt
            result = pipeline.process(frame)
            people = tracker.update(result, dt, frame)
            qresult = analyzer.update(people, dt)
            agg.observe(clock, qresult.count, dt)
            n += 1
            if verbose:
                import cv2
                cv2.putText(frame, f"t={clock:6.1f}s  in line={qresult.count}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("storePose calibrate", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    print("  aborted by user")
                    break
            elif n % 100 == 0:
                wall = time.perf_counter() - started
                rate = n / wall if wall > 0 else 0.0
                pct = f" {100 * n / total:.0f}%" if total else ""
                print(f"  ...{n}{('/' + str(total)) if total else ''} frames{pct} "
                      f"| {wall:.0f}s wall @ {rate:.1f} fps (video t={clock:.0f}s)")

        if verbose:
            import cv2
            cv2.destroyAllWindows()

    values = agg.subwindow_values()
    strategies = compute_strategies(values)
    default_strategy, fill_ratio = pick_default_strategy(values)

    stem = Path(str(config.source)).stem
    payload = {
        "stem": stem,
        "label": "",
        "source": str(config.source),
        "metric": config.busy_metric,
        "subwindow_seconds": sub,
        "samples": len(values),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "occupancy_peak": max(values) if values else 0.0,
        "occupancy_median": median(values) if values else 0.0,
        "fill_ratio": round(fill_ratio, 4),
        "default_strategy": default_strategy,
        "strategies": strategies,
    }
    out = calib_path(stem)
    write_calib(out, payload)

    _print_report(payload, values)
    return payload


def _print_report(payload: dict, values: list[float]) -> None:
    vals = sorted(values)
    peak = vals[-1] if vals else 0.0
    print(f"\nCalibrated {payload['stem']}  "
          f"({payload['samples']} sub-windows, peak {payload['metric']}={peak:g})")
    print(f"  {'strategy':<8}  {'not-busy <=':>11}  {'medium <=':>9}  busy >")
    for name in BUSY_STRATEGIES:
        b = payload["strategies"][name]
        star = "  <- default" if name == payload["default_strategy"] else ""
        print(f"  {name:<8}  {b['low_max']:>11g}  {b['medium_max']:>9g}  "
              f"{b['medium_max']:g}{star}")
    ratio = payload["fill_ratio"]
    shape = "line sits full" if ratio >= SITS_FULL_RATIO else "line empties out"
    print(f"\nAuto-default: {payload['default_strategy']}  "
          f"(median/peak = {ratio:g}, {shape})")
    print(f"Wrote {calib_path(payload['stem'])}")
