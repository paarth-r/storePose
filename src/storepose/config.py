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
    """

    source: int | str = 0
    mode: str = "balanced"
    det_conf: float = 0.5
    kpt_thr: float = 0.5
    device: str = "mps"
    show_fps: bool = True
    save: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        for name in ("det_conf", "kpt_thr"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


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
        help="Person detection confidence threshold (default: 0.5).",
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
    )
