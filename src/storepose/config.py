"""Application configuration and CLI parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

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
        smooth: Whether to One-Euro smooth keypoints.
        smooth_cutoff: One-Euro min_cutoff (lower = smoother/laggier).
        smooth_beta: One-Euro beta (higher = more responsive to speed).
        zone: Path to a queue-zone JSON; enables waiting detection when set.
        define_zone: Launch the interactive zone editor and exit.
        wait_speed: Max speed (body-heights/sec) to count as "slow".
        wait_enter_frames: Consecutive in-zone+slow frames before WAITING.
        wait_exit_seconds: Out-of-condition time before WAITING ends.
        zone_coverage: When ankles are occluded, min fraction of the foot region
            inside the zone to count as in-zone.
        zone_foot_band: Bottom fraction of the box treated as the foot region
            for coverage.
        wait_log: Optional CSV path to append completed waits.
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
    smooth: bool = True
    smooth_cutoff: float = 1.0
    smooth_beta: float = 0.007
    zone: str | None = None
    define_zone: bool = False
    wait_speed: float = 0.15
    wait_enter_frames: int = 5
    wait_exit_seconds: float = 2.0
    zone_coverage: float = 0.5
    zone_foot_band: float = 0.3
    wait_log: str | None = None

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
        if self.hold_seconds < 0:
            raise ValueError(f"hold_seconds must be >= 0, got {self.hold_seconds}")
        if self.smooth_cutoff <= 0:
            raise ValueError(f"smooth_cutoff must be > 0, got {self.smooth_cutoff}")
        if self.smooth_beta < 0:
            raise ValueError(f"smooth_beta must be >= 0, got {self.smooth_beta}")
        if self.wait_speed <= 0:
            raise ValueError(f"wait_speed must be > 0, got {self.wait_speed}")
        if self.wait_enter_frames < 1:
            raise ValueError(f"wait_enter_frames must be >= 1, got {self.wait_enter_frames}")
        if self.wait_exit_seconds < 0:
            raise ValueError(f"wait_exit_seconds must be >= 0, got {self.wait_exit_seconds}")
        if not 0.0 <= self.zone_coverage <= 1.0:
            raise ValueError(f"zone_coverage must be in [0, 1], got {self.zone_coverage}")
        if not 0.0 < self.zone_foot_band <= 1.0:
            raise ValueError(f"zone_foot_band must be in (0, 1], got {self.zone_foot_band}")


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
        "--wait-speed", type=float, default=0.15,
        help="Max speed in body-heights/sec to count as 'slow' (default: 0.15).",
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
        "--wait-log", default=None, metavar="PATH",
        help="Append completed waits (id, entered, exited, seconds) as CSV.",
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
        smooth=args.smooth,
        smooth_cutoff=args.smooth_cutoff,
        smooth_beta=args.smooth_beta,
        zone=args.zone,
        define_zone=args.define_zone,
        wait_speed=args.wait_speed,
        wait_enter_frames=args.wait_enter_frames,
        wait_exit_seconds=args.wait_exit_seconds,
        zone_coverage=args.zone_coverage,
        zone_foot_band=args.zone_foot_band,
        wait_log=args.wait_log,
    )
