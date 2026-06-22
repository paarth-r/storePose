"""Offline busy-signal tooling: turn a wait log into per-window Low/Med/High,
and evaluate predictions against a ground-truth label set.

    # video -> waits.csv (existing pipeline)
    uv run python main.py --source videos/clip.mp4 --zone zones/clip.json --wait-log waits.csv

    # waits.csv -> busy.csv (Low/Medium/High per window)
    uv run python busy_report.py aggregate waits.csv --window 600 -o busy.csv

    # score predictions against a hand-labeled ground-truth CSV
    uv run python busy_report.py eval busy.csv truth.csv

A ground-truth CSV needs at least ``window_index,level`` columns (level in
Low/Medium/High); a busy.csv written by ``aggregate`` already qualifies.
"""

from __future__ import annotations

import argparse
import sys

import os

from storepose.busy.aggregator import BusyAggregator
from storepose.busy.occupancy import sample_occupancy
from storepose.busy.report import (
    read_busy_levels,
    read_waits,
    write_busy,
    write_levels,
)
from storepose.busy.types import METRICS, BusyThresholds
from storepose.eval.cvat_import import (
    parse_cvat_xml,
    read_occupancy_csv,
    sample_occupancy_gt,
    write_occupancy_csv,
)
from storepose.eval.labeling import enumerate_windows, level_for_key, unlabeled
from storepose.eval.metrics import evaluate
from storepose.eval.occupancy_eval import occupancy_eval


def _aggregate(args: argparse.Namespace) -> int:
    waits = read_waits(args.waits)
    if not waits:
        print(f"No waits in {args.waits}; nothing to aggregate.", file=sys.stderr)
        return 1
    thresholds = BusyThresholds(
        metric=args.metric,
        low_max=args.low_max,
        medium_max=args.medium_max,
        hysteresis=args.hysteresis,
    )
    agg = BusyAggregator(
        thresholds,
        window_seconds=args.window,
        sub_window_seconds=args.subwindow or None,
    )
    for t, occ in sample_occupancy(waits, step=args.step):
        agg.observe(t, occ, dt=args.step)
    for wt in waits:
        agg.add_wait(wt)
    windows = agg.windows()

    print(
        f"{len(windows)} window(s) of {args.window:.0f}s; "
        f"metric={args.metric} thresholds Low<= {args.low_max}, Med<= {args.medium_max}"
    )
    for bw in windows:
        ft = bw.features
        print(
            f"  win {bw.index:>3} [{bw.start_s:7.1f}-{bw.end_s:7.1f}s]  "
            f"{bw.level.label:<6} ({args.metric}={bw.metric_value:.2f}, "
            f"mean_occ={ft.mean_occupancy:.2f}, max={ft.max_occupancy:.0f}, "
            f"served={ft.throughput}, mean_wait={ft.mean_wait_seconds:.1f}s)"
        )
    if args.output:
        write_busy(args.output, windows)
        print(f"Wrote {args.output}")
    return 0


def _eval(args: argparse.Namespace) -> int:
    pred = read_busy_levels(args.pred)
    truth = read_busy_levels(args.truth)
    if not set(pred) & set(truth):
        print("No overlapping window_index between prediction and truth.",
              file=sys.stderr)
        return 1
    print(evaluate(truth, pred).format())
    return 0


def _import_cvat(args: argparse.Namespace) -> int:
    try:
        with open(args.export) as f:
            tracks = parse_cvat_xml(f.read())
    except FileNotFoundError:
        print(f"File not found: {args.export}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Failed to parse {args.export}: {exc}", file=sys.stderr)
        return 1
    if not tracks:
        print(f"No tracks in {args.export}; nothing to import.", file=sys.stderr)
        return 1
    samples = sample_occupancy_gt(tracks, fps=args.fps, step=args.step)
    write_occupancy_csv(args.output, samples)
    print(
        f"{len(tracks)} track(s) -> {len(samples)} occupancy sample(s) "
        f"at {args.step}s; wrote {args.output}"
    )
    return 0


def _eval_occupancy(args: argparse.Namespace) -> int:
    gt = read_occupancy_csv(args.gt)
    waits = read_waits(args.waits)
    if not waits:
        print(f"No waits in {args.waits}; nothing to score.", file=sys.stderr)
        return 1
    pred = sample_occupancy(waits, step=args.step)
    rep = occupancy_eval(gt, pred)
    if rep.n == 0:
        print(
            "No overlapping timestamps between GT and predicted occupancy. "
            "Did you use the same --step for import-cvat and the pipeline?",
            file=sys.stderr,
        )
        return 1
    print(rep.format())
    return 0


def _label(args: argparse.Namespace) -> int:
    import cv2  # local: only the labeling UI needs OpenCV's GUI

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open {args.video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = n_frames / fps if fps else 0.0
    windows = enumerate_windows(duration, args.window)
    if not windows:
        print("Video has zero duration; nothing to label.", file=sys.stderr)
        return 1

    labels = read_busy_levels(args.output) if os.path.exists(args.output) else {}
    todo = unlabeled(windows, labels)
    print(
        f"{args.video}: {duration:.0f}s, {len(windows)} window(s) of {args.window:.0f}s; "
        f"{len(todo)} to label ({len(labels)} already done)."
    )
    print("Keys: 1/l=Low  2/m=Medium  3/h=High  s=skip  q=save & quit")

    win_name = "label (1/2/3 = Low/Med/High, s=skip, q=quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    delay = max(1, int(1000.0 / fps))
    quit_all = False
    for w in todo:
        if quit_all:
            break
        start_f = int(w.start_s * fps)
        end_f = int(w.end_s * fps)
        chosen = None
        while chosen is None and not quit_all:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            f = start_f
            while f < end_f:
                ok, frame = cap.read()
                if not ok:
                    break
                f += 1
                banner = f"window {w.index}  [{w.start_s:.0f}-{w.end_s:.0f}s]   1/2/3 label  s skip  q quit"
                cv2.putText(frame, banner, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(win_name, frame)
                key = cv2.waitKey(delay) & 0xFF
                if key == 0xFF:
                    continue
                ch = chr(key)
                if ch == "q":
                    quit_all = True
                    break
                if ch == "s":
                    chosen = "skip"
                    break
                lvl = level_for_key(ch)
                if lvl is not None:
                    labels[w.index] = lvl
                    write_levels(args.output, labels)  # save incrementally
                    chosen = lvl.label
                    print(f"  window {w.index}: {lvl.label}")
                    break
            # window played to the end with no key -> loop and replay
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {len(labels)} label(s) to {args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="busy_report", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("aggregate", help="wait log -> per-window busy labels")
    a.add_argument("waits", help="Path to a wait-log CSV (from --wait-log).")
    a.add_argument("-o", "--output", default=None, metavar="PATH",
                   help="Write the per-window busy report to this CSV.")
    a.add_argument("--window", type=float, default=600.0,
                   help="Window length in seconds (default: 600 = 10 min).")
    a.add_argument("--step", type=float, default=1.0,
                   help="Occupancy sampling step in seconds (default: 1.0).")
    a.add_argument("--subwindow", type=float, default=0.0,
                   help="Two-level smoothing sub-window in seconds; 0 disables "
                        "(e.g. 60 = per-minute robust estimate). Default: 0.")
    a.add_argument("--metric", choices=METRICS, default="occupancy_p90",
                   help="Feature that drives the label (default: occupancy_p90).")
    a.add_argument("--low-max", type=float, default=1.0,
                   help="Upper bound of the LOW band (default: 1.0). CALIBRATE.")
    a.add_argument("--medium-max", type=float, default=3.0,
                   help="Upper bound of the MEDIUM band (default: 3.0). CALIBRATE.")
    a.add_argument("--hysteresis", type=float, default=0.0,
                   help="Cross-window deadband to suppress flapping (default: 0).")
    a.set_defaults(func=_aggregate)

    e = sub.add_parser("eval", help="score predicted busy labels vs. ground truth")
    e.add_argument("pred", help="Predicted busy CSV (from aggregate -o).")
    e.add_argument("truth", help="Ground-truth CSV (window_index,level).")
    e.set_defaults(func=_eval)

    ic = sub.add_parser("import-cvat",
                        help="CVAT point-track XML -> occupancy GT CSV")
    ic.add_argument("export", help="Path to a CVAT-for-video XML export.")
    ic.add_argument("--fps", type=float, required=True,
                    help="Clip frame rate; maps CVAT frame numbers to seconds.")
    ic.add_argument("--step", type=float, default=1.0,
                    help="Occupancy sampling step in seconds (default: 1.0).")
    ic.add_argument("-o", "--output", required=True, metavar="PATH",
                    help="Occupancy GT CSV to write (t_s,occupancy).")
    ic.set_defaults(func=_import_cvat)

    eo = sub.add_parser("eval-occupancy",
                        help="score predicted occupancy (from waits) vs. GT")
    eo.add_argument("gt", help="Occupancy GT CSV (from import-cvat).")
    eo.add_argument("waits", help="Wait-log CSV (from --wait-log).")
    eo.add_argument("--step", type=float, default=1.0,
                    help="Sampling step; must match the import-cvat --step "
                         "(default: 1.0).")
    eo.set_defaults(func=_eval_occupancy)

    lb = sub.add_parser("label", help="hand-label a video's windows -> truth CSV")
    lb.add_argument("video", help="Path to the video to label.")
    lb.add_argument("-o", "--output", required=True, metavar="PATH",
                    help="Ground-truth CSV to write/resume (window_index,level).")
    lb.add_argument("--window", type=float, default=600.0,
                    help="Window length in seconds (default: 600 = 10 min).")
    lb.set_defaults(func=_label)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
