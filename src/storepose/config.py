"""Application configuration and CLI parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .busy.types import BUSY_STRATEGIES
from .busy.types import METRICS as BUSY_METRICS

MODES = ("lightweight", "balanced", "performance")
DEVICES = ("cpu", "mps")
REID_BACKENDS = ("osnet-x1", "osnet-x025", "histogram")
_REID_THR_DEFAULTS = {"osnet-x1": 0.8, "osnet-x025": 0.8, "histogram": 0.6}


def reid_thr_for(backend: str, override: float | None) -> float:
    """Resolve the appearance similarity floor: explicit override, else per-backend."""
    if override is not None:
        return override
    return _REID_THR_DEFAULTS[backend]


@dataclass(frozen=True)
class AppConfig:
    """Validated runtime configuration for the pose pipeline.

    Attributes:
        source: Webcam index (int) or a path to a video file (str).
        mode: rtmlib model mode — one of ``MODES``.
        det_conf: Person-detection score threshold in ``[0, 1]``.
        det_overlap: Drop a detection box more than this fraction *contained*
            within a larger box (duplicate-on-one-person suppression), in ``[0, 1]``.
        kpt_thr: Per-keypoint confidence threshold for drawing, in ``[0, 1]``.
        device: Compute device — ``"cpu"`` or ``"mps"`` (CoreML).
        show_fps: Whether to overlay a rolling FPS counter.
        save: Optional output path for an annotated .mp4; ``None`` to not save.
        save_mp4: Auto-save the annotated stream to a timestamped file under
            ``runs/`` (no path needed); ``--save`` overrides the auto name.
        blur_faces: Pixelate each person's face (from face keypoints, else the
            top quarter of their box) in displayed and saved frames; on by
            default, disable with ``--no-blur-faces``.
        track: Whether to track people (stable ids, coasting) vs raw per-frame.
        hold_seconds: How long a lost person's box keeps coasting.
        min_hits: Detections before a track is confirmed/drawn.
        iou_thr: Min IoU to associate a detection to a track.
        max_overlap: A coasting track overlapping a kept track by more than this
            is dropped (suppresses duplicate boxes on one person).
        reid: Re-attach a returning person's id via appearance (requires tracking).
        reid_seconds: How long a lost track stays re-attachable, in seconds.
        reid_backend: Appearance backend / OSNet size (one of ``REID_BACKENDS``).
        reid_weights: Local OSNet ONNX path overriding the auto-downloaded weights.
        reid_thr: Appearance similarity floor for re-attach in [-1, 1]; ``None``
            uses the per-backend default (see ``reid_thr_for``).
        smooth: Whether to One-Euro smooth keypoints.
        smooth_cutoff: One-Euro min_cutoff (lower = smoother/laggier).
        smooth_beta: One-Euro beta (higher = more responsive to speed).
        zone: Path to a queue-zone JSON; enables waiting detection when set.
        define_zone: Launch the interactive zone editor and exit.
        pos_zone: Path to a POS-zone JSON; enables waiting-vs-serving split.
        define_pos_zone: Launch the editor for the POS zone and exit.
        alt_zone: Path to a non-Mashgin checkout zone JSON (the comparison).
        define_alt_zone: Launch the editor for the non-Mashgin checkout and exit.
        blur_zone: Path to a censor-zone JSON; those polygon regions are pixelated
            in the live view, recording, and browser feed.
        define_blur_zone: Launch the editor for the censor/blur zone and exit.
        wait_enter_frames: Consecutive in-zone frames before WAITING.
        pos_enter_frames: Consecutive in-POS frames before SERVING (debounce).
        transit_speed: Reject walk-throughs: directional speed (body-heights/sec)
            above which a person counts in no zone; 0 disables the filter.
        wait_exit_seconds: Out-of-condition time before WAITING ends.
        zone_coverage: When ankles are occluded, min fraction of the foot region
            inside the zone to count as in-zone.
        zone_foot_band: Bottom fraction of the box treated as the foot region
            for coverage.
        wait_log: Optional CSV path to append completed waits.
        pos_reassign_seconds: Hold window for an occluded checkout serve so a new
            apparition in the same checkout can adopt it and continue the timer;
            0 disables. See ``QueueAnalyzer`` reassignment.
        pos_reassign_mashgin: Also apply occlusion re-assignment to the Mashgin
            checkout (default off; non-Mashgin only).
        busy: Aggregate occupancy into a Low/Medium/High busy signal and show a
            live badge (requires an active zone).
        busy_log: Optional CSV path for the per-window busy report at exit.
        busy_window: Busy aggregation window length in seconds (default 600=10m).
        busy_metric: Window feature that drives the label (see busy.types.METRICS).
        busy_low_max: Upper bound of the LOW band (metric units). Calibrate.
        busy_medium_max: Upper bound of the MEDIUM band (metric units). Calibrate.
        busy_hysteresis: Cross-window deadband to suppress label flapping.
        dashboard: Serve the live web dashboard during the run.
        dashboard_port: Port for the dashboard HTTP server.
        num_mashgins: Count of parallel Mashgin self-checkout kiosks; the
            dashboard divides the Mashgin time and the vs-staffed comparison by
            it to reflect parallel throughput.
        debug: Step through frames one at a time (scrub a rolling buffer) and push
            per-person reasoning rows to the dashboard Debug tab.
    """

    source: int | str = 0
    mode: str = "balanced"
    det_conf: float = 0.5
    det_overlap: float = 0.8
    kpt_thr: float = 0.5
    device: str = "mps"
    show_fps: bool = True
    show_conf: bool = False
    save: str | None = None
    save_mp4: bool = False
    blur_faces: bool = True
    track: bool = True
    hold_seconds: float = 1.5
    min_hits: int = 3
    iou_thr: float = 0.3
    max_overlap: float = 0.5
    reid: bool = True
    reid_seconds: float = 15.0
    reid_backend: str = "osnet-x025"
    reid_weights: str | None = None
    reid_thr: float | None = None
    smooth: bool = True
    smooth_cutoff: float = 1.0
    smooth_beta: float = 0.007
    predict_drift: bool = False
    coast: bool = False
    stationary_seconds: float = 20.0
    stationary_radius: float = 0.03
    zone: str | None = None
    define_zone: bool = False
    pos_zone: str | None = None
    define_pos_zone: bool = False
    alt_zone: str | None = None
    define_alt_zone: bool = False
    no_alt: bool = False
    blur_zone: str | None = None
    define_blur_zone: bool = False
    wait_enter_frames: int = 5
    pos_enter_frames: int = 3
    transit_speed: float = 0.4
    transit_window: float = 1.0
    wait_exit_seconds: float = 2.0
    zone_coverage: float = 0.5
    zone_foot_band: float = 0.3
    wait_min_dwell: float = 0.0
    min_wait: float = 5.0
    wait_log: str | None = None
    pos_reassign_seconds: float = 20.0
    pos_reassign_mashgin: bool = False
    reject_short: bool = False
    reject_floor: float = 2.0
    reject_frac: float = 0.25
    reject_warmup: int = 10
    busy: bool = False
    busy_log: str | None = None
    busy_window: float = 600.0
    busy_subwindow: float = 0.0
    busy_metric: str = "occupancy_p90"
    busy_low_max: float = 1.0
    busy_medium_max: float = 3.0
    busy_hysteresis: float = 0.0
    busy_live_window: float = 30.0
    calib: str | None = None
    busy_strategy: str | None = None  # None => use the calib file's auto default
    dashboard: bool = True
    dashboard_port: int = 8000
    num_mashgins: int = 1
    debug: bool = False
    calibrate: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        for name in ("det_conf", "det_overlap", "kpt_thr"):
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
        if self.reid_backend not in REID_BACKENDS:
            raise ValueError(
                f"reid_backend must be one of {REID_BACKENDS}, got {self.reid_backend!r}")
        if self.reid_thr is not None and not -1.0 <= self.reid_thr <= 1.0:
            raise ValueError(f"reid_thr must be in [-1, 1], got {self.reid_thr}")
        if self.hold_seconds < 0:
            raise ValueError(f"hold_seconds must be >= 0, got {self.hold_seconds}")
        if self.smooth_cutoff <= 0:
            raise ValueError(f"smooth_cutoff must be > 0, got {self.smooth_cutoff}")
        if self.smooth_beta < 0:
            raise ValueError(f"smooth_beta must be >= 0, got {self.smooth_beta}")
        if self.wait_enter_frames < 1:
            raise ValueError(f"wait_enter_frames must be >= 1, got {self.wait_enter_frames}")
        if self.transit_speed < 0:
            raise ValueError(f"transit_speed must be >= 0, got {self.transit_speed}")
        if self.transit_window <= 0:
            raise ValueError(f"transit_window must be > 0, got {self.transit_window}")
        if self.pos_enter_frames < 1:
            raise ValueError(f"pos_enter_frames must be >= 1, got {self.pos_enter_frames}")
        if self.wait_exit_seconds < 0:
            raise ValueError(f"wait_exit_seconds must be >= 0, got {self.wait_exit_seconds}")
        if not 0.0 <= self.zone_coverage <= 1.0:
            raise ValueError(f"zone_coverage must be in [0, 1], got {self.zone_coverage}")
        if not 0.0 < self.zone_foot_band <= 1.0:
            raise ValueError(f"zone_foot_band must be in (0, 1], got {self.zone_foot_band}")
        if self.wait_min_dwell < 0:
            raise ValueError(f"wait_min_dwell must be >= 0, got {self.wait_min_dwell}")
        if self.min_wait < 0:
            raise ValueError(f"min_wait must be >= 0, got {self.min_wait}")
        if self.pos_reassign_seconds < 0:
            raise ValueError(
                f"pos_reassign_seconds must be >= 0, got {self.pos_reassign_seconds}")
        if self.reject_floor < 0:
            raise ValueError(f"reject_floor must be >= 0, got {self.reject_floor}")
        if not 0.0 <= self.reject_frac <= 1.0:
            raise ValueError(f"reject_frac must be in [0, 1], got {self.reject_frac}")
        if self.reject_warmup < 0:
            raise ValueError(f"reject_warmup must be >= 0, got {self.reject_warmup}")
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
        if self.busy_live_window <= 0:
            raise ValueError(f"busy_live_window must be > 0, got {self.busy_live_window}")
        if self.busy_strategy is not None and self.busy_strategy not in BUSY_STRATEGIES:
            raise ValueError(
                f"busy_strategy must be one of {BUSY_STRATEGIES}, got {self.busy_strategy!r}"
            )
        if not 1 <= self.dashboard_port <= 65535:
            raise ValueError(f"dashboard_port must be in [1, 65535], got {self.dashboard_port}")
        if self.num_mashgins < 1:
            raise ValueError(f"num_mashgins must be >= 1, got {self.num_mashgins}")


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
        default=0.5,
        help="Person detection confidence threshold (default: 0.5). Props are "
             "not separable by score (use the stationary filter); a higher "
             "threshold mainly costs recall on partially-occluded people.",
    )
    parser.add_argument(
        "--det-overlap",
        type=float,
        default=0.8,
        help="Drop a detection box more than this fraction contained within a "
             "larger one (duplicate-on-one-person suppression; default: 0.8).",
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
        "--conf",
        dest="show_conf",
        action="store_true",
        help="Overlay each person's detector confidence next to their box/ID.",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Also write the annotated stream to this .mp4 file.",
    )
    parser.add_argument(
        "--save-mp4", dest="save_mp4", action="store_true",
        help="Auto-save the annotated stream to a timestamped runs/<source>_*.mp4 "
             "(no path needed). --save overrides the auto name if both are given.",
    )
    parser.add_argument(
        "--no-blur-faces", dest="blur_faces", action="store_false",
        help="Disable face pixelation (on by default; blurs from face "
             "keypoints, or the top quarter of the box).",
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
        "--reid-seconds", type=float, default=10.0,
        help="How long a lost track stays re-attachable, in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--reid-backend", choices=REID_BACKENDS, default="osnet-x025",
        help="Appearance backend for re-id: learned OSNet embedding (osnet-x025 = "
             "fast, osnet-x1 = accurate) or the HSV color histogram "
             "(default: osnet-x025).",
    )
    parser.add_argument(
        "--reid-weights", default=None, metavar="PATH",
        help="Local OSNet ONNX file to use instead of the auto-downloaded weights.",
    )
    parser.add_argument(
        "--reid-thr", type=float, default=None,
        help="Appearance similarity floor for re-attach, in [-1,1]. Default "
             "resolves per backend (osnet: 0.8, histogram: 0.6).",
    )
    parser.add_argument(
        "--no-smooth", dest="smooth", action="store_false",
        help="Disable One-Euro keypoint smoothing.",
    )
    parser.add_argument(
        "--predict-drift", dest="predict_drift", action="store_true",
        help="Extrapolate a coasting track's box along Kalman velocity. Off by "
             "default (the box holds its last detected position), which avoids "
             "drift away from the person and the mis-associations it causes.",
    )
    parser.add_argument(
        "--coast", dest="coast", action="store_true",
        help="Keep emitting a track that has no detection this frame (held box) "
             "for up to --hold-seconds. Off by default: a track with no detection "
             "is dropped immediately; a returning detection re-attaches its id by "
             "range/time/appearance.",
    )
    parser.add_argument(
        "--stationary-seconds", type=float, default=20.0,
        help="Suppress a track whose center stays within --stationary-radius for "
             "this many seconds (a fixed prop, not a person); 0 disables "
             "(default: 20).",
    )
    parser.add_argument(
        "--stationary-radius", type=float, default=0.03,
        help="Movement radius for the stationary filter, as a fraction of the "
             "frame diagonal (default: 0.03).",
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
        "--pos-zone", default=None, metavar="PATH",
        help="POS-zone JSON to load; splits line time into waiting vs serving.",
    )
    parser.add_argument(
        "--define-pos-zone", dest="define_pos_zone", action="store_true",
        help="Launch the interactive editor for the POS zone and exit.",
    )
    parser.add_argument(
        "--alt-zone", default=None, metavar="PATH",
        help="Non-Mashgin checkout zone JSON; enables the checkout comparison.",
    )
    parser.add_argument(
        "--define-alt-zone", dest="define_alt_zone", action="store_true",
        help="Launch the interactive editor for the non-Mashgin checkout and exit.",
    )
    parser.add_argument(
        "--blur-zone", default=None, metavar="PATH",
        help="Censor-zone JSON; pixelate these polygon regions in the live view, "
             "recording, and browser feed (e.g. to hide a monitor or doorway).",
    )
    parser.add_argument(
        "--define-blur-zone", dest="define_blur_zone", action="store_true",
        help="Launch the interactive editor for the censor/blur zone and exit.",
    )
    parser.add_argument(
        "--no-alt", dest="no_alt", action="store_true",
        help="Ignore the non-Mashgin (alt) checkout zone even if --alt-zone is "
             "given; disables the checkout comparison. Use when the staffed lane "
             "is too occluded / full of walk-throughs to measure reliably.",
    )
    parser.add_argument(
        "--wait-enter-frames", type=int, default=5,
        help="Consecutive in-zone+slow frames before WAITING (default: 5).",
    )
    parser.add_argument(
        "--pos-enter-frames", type=int, default=3,
        help="Consecutive in-POS frames before SERVING; debounces the POS edge "
             "(default: 3).",
    )
    parser.add_argument(
        "--transit-speed", type=float, default=0.4,
        help="Reject walk-throughs: average speed (body-heights/sec, net "
             "displacement over --transit-window) above which a person counts in "
             "no zone; 0 disables (default: 0.4).",
    )
    parser.add_argument(
        "--transit-window", type=float, default=1.0,
        help="Trailing window (seconds) over which transit displacement is "
             "measured (default: 1.0).",
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
        "--min-wait", type=float, default=5.0,
        help="Minimum real visit time in seconds: a completed visit whose "
             "outcome-relevant time (POS time if served, line time if abandoned) "
             "is under this is excluded from every average, chart, and the busy "
             "signal (kept in the wait log, flagged rejected). Also the dwell a "
             "person must sustain at the other checkout before a Mashgin<->staffed "
             "switch counts, so brief box-clips don't register an instant "
             "transition. 0 disables (default: 5.0).",
    )
    parser.add_argument(
        "--wait-log", default=None, metavar="PATH",
        help="Append completed waits (id, entered, exited, seconds) as CSV.",
    )
    parser.add_argument(
        "--pos-reassign-seconds", type=float, default=20.0,
        help="When a checkout serve loses its track without a clear exit "
             "(occlusion), hold it this long so a new apparition entering the "
             "same checkout can adopt it and continue the timer; 0 disables "
             "(default: 20.0). Applies to the non-Mashgin checkout.",
    )
    parser.add_argument(
        "--pos-reassign-mashgin", dest="pos_reassign_mashgin", action="store_true",
        help="Also apply occlusion re-assignment to the Mashgin checkout "
             "(off by default; non-Mashgin only).",
    )
    parser.add_argument(
        "--reject-short", action="store_true",
        help="Flag implausibly short visits (likely false detections) as rejected: "
             "kept in the wait log but excluded from the busy signal.",
    )
    parser.add_argument(
        "--reject-floor", type=float, default=2.0,
        help="Absolute minimum plausible visit duration in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--reject-frac", type=float, default=0.25,
        help="Reject a visit shorter than this fraction of the running median for "
             "its outcome (default: 0.25).",
    )
    parser.add_argument(
        "--reject-warmup", type=int, default=10,
        help="Accepted samples per outcome before the relative (median) term "
             "applies; until then only --reject-floor (default: 10).",
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
    parser.add_argument(
        "--busy-live-window", type=float, default=30.0,
        help="Trailing seconds the live badge summarises, so it tracks recent "
             "activity rather than the whole 10-min window (default: 30).",
    )
    parser.add_argument(
        "--calib", default=None, metavar="PATH",
        help="Load per-view busy bands from a calib JSON (see --calibrate); "
             "overrides the manual --busy-*-max thresholds.",
    )
    parser.add_argument(
        "--busy-strategy", choices=BUSY_STRATEGIES, default=None,
        help="Override which calibrated band set to use from --calib; default is "
             "the auto-selected strategy stored in the calib file.",
    )
    parser.add_argument(
        "--no-dashboard", dest="dashboard", action="store_false",
        help="Disable the live web dashboard.",
    )
    parser.add_argument(
        "--dashboard-port", type=int, default=8000,
        help="Port for the live dashboard HTTP server (default: 8000).",
    )
    parser.add_argument(
        "--num-mashgins", type=int, default=1,
        help="Number of Mashgin self-checkout kiosks running in parallel. The "
             "dashboard divides the Mashgin self-checkout time (and the "
             "vs-staffed comparison) by this to reflect parallel throughput "
             "(default: 1).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Frame-by-frame step mode: scrub a rolling buffer and read each "
             "person's classification in the dashboard Debug tab.",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Infer busy bands for --source from its occupancy distribution and "
             "write calib/<stem>.json (needs --zone). Headless unless -v.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="During --calibrate, show an annotated preview window (otherwise "
             "headless).",
    )
    return parser


def from_args(argv: list[str] | None = None) -> AppConfig:
    """Parse ``argv`` (or ``sys.argv``) into a validated :class:`AppConfig`."""
    args = _build_parser().parse_args(argv)
    return AppConfig(
        source=args.source,
        mode=args.mode,
        det_conf=args.det_conf,
        det_overlap=args.det_overlap,
        kpt_thr=args.kpt_thr,
        device=args.device,
        show_fps=args.show_fps,
        show_conf=args.show_conf,
        save=args.save,
        save_mp4=args.save_mp4,
        blur_faces=args.blur_faces,
        track=args.track,
        hold_seconds=args.hold_seconds,
        min_hits=args.min_hits,
        iou_thr=args.iou_thr,
        max_overlap=args.max_overlap,
        reid=args.reid,
        reid_seconds=args.reid_seconds,
        reid_backend=args.reid_backend,
        reid_weights=args.reid_weights,
        reid_thr=args.reid_thr,
        smooth=args.smooth,
        predict_drift=args.predict_drift,
        coast=args.coast,
        stationary_seconds=args.stationary_seconds,
        stationary_radius=args.stationary_radius,
        smooth_cutoff=args.smooth_cutoff,
        smooth_beta=args.smooth_beta,
        zone=args.zone,
        define_zone=args.define_zone,
        pos_zone=args.pos_zone,
        define_pos_zone=args.define_pos_zone,
        alt_zone=args.alt_zone,
        define_alt_zone=args.define_alt_zone,
        no_alt=args.no_alt,
        blur_zone=args.blur_zone,
        define_blur_zone=args.define_blur_zone,
        wait_enter_frames=args.wait_enter_frames,
        pos_enter_frames=args.pos_enter_frames,
        transit_speed=args.transit_speed,
        transit_window=args.transit_window,
        wait_exit_seconds=args.wait_exit_seconds,
        zone_coverage=args.zone_coverage,
        zone_foot_band=args.zone_foot_band,
        wait_min_dwell=args.wait_min_dwell,
        min_wait=args.min_wait,
        wait_log=args.wait_log,
        pos_reassign_seconds=args.pos_reassign_seconds,
        pos_reassign_mashgin=args.pos_reassign_mashgin,
        reject_short=args.reject_short,
        reject_floor=args.reject_floor,
        reject_frac=args.reject_frac,
        reject_warmup=args.reject_warmup,
        busy=args.busy or args.busy_log is not None,
        busy_log=args.busy_log,
        busy_window=args.busy_window,
        busy_subwindow=args.busy_subwindow,
        busy_metric=args.busy_metric,
        busy_low_max=args.busy_low_max,
        busy_medium_max=args.busy_medium_max,
        busy_hysteresis=args.busy_hysteresis,
        busy_live_window=args.busy_live_window,
        calib=args.calib,
        busy_strategy=args.busy_strategy,
        dashboard=args.dashboard,
        dashboard_port=args.dashboard_port,
        num_mashgins=args.num_mashgins,
        debug=args.debug,
        calibrate=args.calibrate,
        verbose=args.verbose,
    )
