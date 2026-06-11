"""Application configuration and CLI parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .busy.types import METRICS as BUSY_METRICS

MODES = ("lightweight", "balanced", "performance")
DEVICES = ("cpu", "mps")


@dataclass(frozen=True)
class AppConfig:
    """Validated runtime configuration for the pose pipeline.

    Attributes:
        source: Webcam index (int) or a path to a video file (str).
        mode: rtmlib model mode — one of ``MODES``.
        det_conf: Person-detection score threshold in ``[0, 1]``.
        kpt_thr: Per-keypoint confidence threshold for drawing, in ``[0, 1]``.
        device: Compute device — ``"cpu"`` or ``"mps"`` (CoreML).
        show_fps: Whether to overlay a rolling FPS counter.
        save: Optional output path for an annotated .mp4; ``None`` to not save.
        track: Whether to track people (stable ids, coasting) vs raw per-frame.
        hold_seconds: How long a lost person's box keeps coasting.
        min_hits: Detections before a track is confirmed/drawn.
        iou_thr: Min IoU to associate a detection to a track.
        max_overlap: A coasting track overlapping a kept track by more than this
            is dropped (suppresses duplicate boxes on one person).
        reid: Re-attach a returning person's id via appearance (requires tracking).
        reid_seconds: How long a lost track stays re-attachable, in seconds.
        reid_thr: Appearance similarity floor for re-attach, in [-1, 1].
        smooth: Whether to One-Euro smooth keypoints.
        smooth_cutoff: One-Euro min_cutoff (lower = smoother/laggier).
        smooth_beta: One-Euro beta (higher = more responsive to speed).
        zone: Path to a queue-zone JSON; enables waiting detection when set.
        define_zone: Launch the interactive zone editor and exit.
        wait_enter_frames: Consecutive in-zone frames before WAITING.
        wait_exit_seconds: Out-of-condition time before WAITING ends.
        zone_coverage: When ankles are occluded, min fraction of the foot region
            inside the zone to count as in-zone.
        zone_foot_band: Bottom fraction of the box treated as the foot region
            for coverage.
        wait_log: Optional CSV path to append completed waits.
        busy: Aggregate occupancy into a Low/Medium/High busy signal and show a
            live badge (requires an active zone).
        busy_log: Optional CSV path for the per-window busy report at exit.
        busy_window: Busy aggregation window length in seconds (default 600=10m).
        busy_metric: Window feature that drives the label (see busy.types.METRICS).
        busy_low_max: Upper bound of the LOW band (metric units). Calibrate.
        busy_medium_max: Upper bound of the MEDIUM band (metric units). Calibrate.
        busy_hysteresis: Cross-window deadband to suppress label flapping.
    """

    source: int | str = 0
    mode: str = "balanced"
    det_conf: float = 0.95
    kpt_thr: float = 0.5
    device: str = "mps"
    show_fps: bool = True
    save: str | None = None
    track: bool = True
    hold_seconds: float = 1.5
    min_hits: int = 3
    iou_thr: float = 0.3
    max_overlap: float = 0.5
    reid: bool = True
    reid_seconds: float = 5.0
    reid_thr: float = 0.6
    smooth: bool = True
    smooth_cutoff: float = 1.0
    smooth_beta: float = 0.007
    zone: str | None = None
    define_zone: bool = False
    wait_enter_frames: int = 5
    wait_exit_seconds: float = 2.0
    zone_coverage: float = 0.5
    zone_foot_band: float = 0.3
    wait_min_dwell: float = 0.0
    wait_log: str | None = None
    busy: bool = False
    busy_log: str | None = None
    busy_window: float = 600.0
    busy_subwindow: float = 0.0
    busy_metric: str = "occupancy_p90"
    busy_low_max: float = 1.0
    busy_medium_max: float = 3.0
    busy_hysteresis: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        for name in ("det_conf", "kpt_thr"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {self.min_hits}")
        if not 0.0 <= self.iou_thr <= 1.0:
            raise ValueError(f"iou_thr must be in [0, 1], got {self.iou_thr}")
        if not 0.0 <= self.max_overlap <= 1.0:
            raise ValueError(f"max_overlap must be in [0, 1], got {self.max_overlap}")
        if self.reid_seconds < 0:
            raise ValueError(f"reid_seconds must be >= 0, got {self.reid_seconds}")
        if not -1.0 <= self.reid_thr <= 1.0:
            raise ValueError(f"reid_thr must be in [-1, 1], got {self.reid_thr}")
        if self.hold_seconds < 0:
            raise ValueError(f"hold_seconds must be >= 0, got {self.hold_seconds}")
        if self.smooth_cutoff <= 0:
            raise ValueError(f"smooth_cutoff must be > 0, got {self.smooth_cutoff}")
        if self.smooth_beta < 0:
            raise ValueError(f"smooth_beta must be >= 0, got {self.smooth_beta}")
        if self.wait_enter_frames < 1:
            raise ValueError(f"wait_enter_frames must be >= 1, got {self.wait_enter_frames}")
        if self.wait_exit_seconds < 0:
            raise ValueError(f"wait_exit_seconds must be >= 0, got {self.wait_exit_seconds}")
        if not 0.0 <= self.zone_coverage <= 1.0:
            raise ValueError(f"zone_coverage must be in [0, 1], got {self.zone_coverage}")
        if not 0.0 < self.zone_foot_band <= 1.0:
            raise ValueError(f"zone_foot_band must be in (0, 1], got {self.zone_foot_band}")
        if self.wait_min_dwell < 0:
            raise ValueError(f"wait_min_dwell must be >= 0, got {self.wait_min_dwell}")
        from .busy.types import METRICS  # local import avoids package import cost
        if self.busy_window <= 0:
            raise ValueError(f"busy_window must be > 0, got {self.busy_window}")
        if self.busy_subwindow < 0:
            raise ValueError(f"busy_subwindow must be >= 0, got {self.busy_subwindow}")
        if self.busy_metric not in METRICS:
            raise ValueError(f"busy_metric must be one of {METRICS}, got {self.busy_metric!r}")
        if self.busy_medium_max < self.busy_low_max:
            raise ValueError("busy_medium_max must be >= busy_low_max")
        if self.busy_hysteresis < 0:
            raise ValueError(f"busy_hysteresis must be >= 0, got {self.busy_hysteresis}")


def parse_source(value: str | int) -> int | str:
    """Interpret a source as a camera index if numeric, else a file path."""
    if isinstance(value, int):
        return value
    return int(value) if value.isdigit() else value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="storepose",
        description="Realtime multi-person bounding boxes + RTMPose skeletons.",
    )
    parser.add_argument(
        "--source",
        type=parse_source,
        default=0,
        help="Webcam index (e.g. 0) or path to a video file (default: 0).",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="balanced",
        help="Model speed/accuracy mode (default: balanced).",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.95,
        help="Person detection confidence threshold (default: 0.95).",
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.5,
        help="Keypoint confidence threshold for drawing (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="mps",
        help="Compute device; 'mps' uses CoreML, 'cpu' is the fallback (default: mps).",
    )
    parser.add_argument(
        "--no-fps",
        dest="show_fps",
        action="store_false",
        help="Disable the FPS overlay.",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Also write the annotated stream to this .mp4 file.",
    )
    parser.add_argument(
        "--no-track", dest="track", action="store_false",
        help="Disable tracking; draw raw per-frame detections.",
    )
    parser.add_argument(
        "--hold-seconds", type=float, default=1.5,
        help="How long a lost person's box keeps coasting (default: 1.5).",
    )
    parser.add_argument(
        "--min-hits", type=int, default=3,
        help="Detections before a track is confirmed/drawn (default: 3).",
    )
    parser.add_argument(
        "--iou-thr", type=float, default=0.3,
        help="Min IoU to associate a detection to a track (default: 0.3).",
    )
    parser.add_argument(
        "--max-overlap", type=float, default=0.5,
        help="Drop a coasting track overlapping another by more than this "
             "(suppresses duplicate boxes on one person; default: 0.5).",
    )
    parser.add_argument(
        "--no-reid", dest="reid", action="store_false",
        help="Disable appearance re-id (re-attaching a returning person's id).",
    )
    parser.add_argument(
        "--reid-seconds", type=float, default=5.0,
        help="How long a lost track stays re-attachable, in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--reid-thr", type=float, default=0.6,
        help="Appearance similarity floor for re-attach, in [-1,1] (default: 0.6).",
    )
    parser.add_argument(
        "--no-smooth", dest="smooth", action="store_false",
        help="Disable One-Euro keypoint smoothing.",
    )
    parser.add_argument(
        "--smooth-cutoff", type=float, default=1.0,
        help="One-Euro min_cutoff; lower = smoother/laggier (default: 1.0).",
    )
    parser.add_argument(
        "--smooth-beta", type=float, default=0.007,
        help="One-Euro beta; higher = more responsive to speed (default: 0.007).",
    )
    parser.add_argument(
        "--zone", default=None, metavar="PATH",
        help="Queue-zone JSON to load; enables waiting-in-line detection.",
    )
    parser.add_argument(
        "--define-zone", dest="define_zone", action="store_true",
        help="Launch the interactive zone editor for --source and exit.",
    )
    parser.add_argument(
        "--wait-enter-frames", type=int, default=5,
        help="Consecutive in-zone+slow frames before WAITING (default: 5).",
    )
    parser.add_argument(
        "--wait-exit-seconds", type=float, default=2.0,
        help="Out-of-condition time before WAITING ends (default: 2.0).",
    )
    parser.add_argument(
        "--zone-coverage", type=float, default=0.5,
        help="When ankles are occluded, min fraction of the foot region inside "
             "the zone to count as in-zone (default: 0.5).",
    )
    parser.add_argument(
        "--zone-foot-band", type=float, default=0.3,
        help="Bottom fraction of the box used as the foot region for coverage "
             "(default: 0.3).",
    )
    parser.add_argument(
        "--wait-min-dwell", type=float, default=0.0,
        help="Minimum in-zone dwell (seconds) before counting as in line; "
             "rejects pass-through bystanders (default: 0 = off).",
    )
    parser.add_argument(
        "--wait-log", default=None, metavar="PATH",
        help="Append completed waits (id, entered, exited, seconds) as CSV.",
    )
    parser.add_argument(
        "--busy", action="store_true",
        help="Aggregate occupancy into a live Low/Medium/High busy badge "
             "(needs an active --zone).",
    )
    parser.add_argument(
        "--busy-log", default=None, metavar="PATH",
        help="Write the per-window busy report to this CSV at exit (implies --busy).",
    )
    parser.add_argument(
        "--busy-window", type=float, default=600.0,
        help="Busy aggregation window in seconds (default: 600 = 10 min).",
    )
    parser.add_argument(
        "--busy-subwindow", type=float, default=0.0,
        help="Sub-window length in seconds for two-level smoothing; 0 disables "
             "(e.g. 60 = per-minute robust estimate, median across the window).",
    )
    parser.add_argument(
        "--busy-metric", choices=BUSY_METRICS, default="occupancy_p90",
        help="Window feature that drives the busy label (default: occupancy_p90).",
    )
    parser.add_argument(
        "--busy-low-max", type=float, default=1.0,
        help="Upper bound of the LOW band, in metric units (default: 1.0). Calibrate.",
    )
    parser.add_argument(
        "--busy-medium-max", type=float, default=3.0,
        help="Upper bound of the MEDIUM band, in metric units (default: 3.0). Calibrate.",
    )
    parser.add_argument(
        "--busy-hysteresis", type=float, default=0.0,
        help="Cross-window deadband to suppress busy-label flapping (default: 0).",
    )
    return parser


def from_args(argv: list[str] | None = None) -> AppConfig:
    """Parse ``argv`` (or ``sys.argv``) into a validated :class:`AppConfig`."""
    args = _build_parser().parse_args(argv)
    return AppConfig(
        source=args.source,
        mode=args.mode,
        det_conf=args.det_conf,
        kpt_thr=args.kpt_thr,
        device=args.device,
        show_fps=args.show_fps,
        save=args.save,
        track=args.track,
        hold_seconds=args.hold_seconds,
        min_hits=args.min_hits,
        iou_thr=args.iou_thr,
        max_overlap=args.max_overlap,
        reid=args.reid,
        reid_seconds=args.reid_seconds,
        reid_thr=args.reid_thr,
        smooth=args.smooth,
        smooth_cutoff=args.smooth_cutoff,
        smooth_beta=args.smooth_beta,
        zone=args.zone,
        define_zone=args.define_zone,
        wait_enter_frames=args.wait_enter_frames,
        wait_exit_seconds=args.wait_exit_seconds,
        zone_coverage=args.zone_coverage,
        zone_foot_band=args.zone_foot_band,
        wait_min_dwell=args.wait_min_dwell,
        wait_log=args.wait_log,
        busy=args.busy or args.busy_log is not None,
        busy_log=args.busy_log,
        busy_window=args.busy_window,
        busy_subwindow=args.busy_subwindow,
        busy_metric=args.busy_metric,
        busy_low_max=args.busy_low_max,
        busy_medium_max=args.busy_medium_max,
        busy_hysteresis=args.busy_hysteresis,
    )
