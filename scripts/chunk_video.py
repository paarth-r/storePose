"""Split a video into fixed-size frame chunks (no decimation, fps preserved).

Reads the source sequentially with OpenCV and writes consecutive runs of
``--chunk-size`` frames to ``<out-root>/<stem>/partNN.mp4``, preserving the
source frame rate and resolution. The original file is left untouched.

Usage:
    uv run python scripts/chunk_video.py --source videos/cumberland/foo.mp4
    uv run python scripts/chunk_video.py --source foo.mp4 --chunk-size 5000 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def chunk_video(source: Path, chunk_size: int, out_root: Path, force: bool) -> list[Path]:
    """Split ``source`` into ``chunk_size``-frame mp4s under ``out_root/<stem>/``.

    Returns the list of written chunk paths. Raises ValueError on bad input or a
    non-empty destination (unless ``force``).
    """
    if chunk_size < 1:
        raise ValueError(f"chunk-size must be >= 1, got {chunk_size}")
    if not source.is_file():
        raise ValueError(f"source not found: {source}")

    out_dir = out_root / source.stem
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise ValueError(f"destination not empty: {out_dir} (use --force to overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"could not open: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"bad video metadata: fps={fps} size={width}x{height}")

    n_chunks = max(1, -(-total // chunk_size)) if total > 0 else 0  # ceil; 0 means unknown
    pad = max(2, len(str(max(0, n_chunks - 1))))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    print(f"source : {source}  ({width}x{height} @ {fps:g}fps, {total} frames)")
    print(f"output : {out_dir}/  (chunk-size {chunk_size})")

    written: list[Path] = []
    writer = None
    idx = -1  # 0-based source frame index
    chunk_no = -1
    chunk_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if writer is None:
                chunk_no += 1
                path = out_dir / f"part{chunk_no:0{pad}d}.mp4"
                writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise ValueError(f"could not open writer: {path}")
                written.append(path)
                chunk_frames = 0
            writer.write(frame)
            chunk_frames += 1
            if chunk_frames >= chunk_size:
                writer.release()
                print(f"  wrote {written[-1].name}  ({chunk_frames} frames)")
                writer = None
    finally:
        if writer is not None:
            writer.release()
            print(f"  wrote {written[-1].name}  ({chunk_frames} frames)")
        cap.release()

    print(f"done: {len(written)} chunk(s), {idx + 1} frames total")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split a video into fixed-size frame chunks.")
    parser.add_argument("--source", required=True, help="Path to the source video (or a bare name under videos/).")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Frames per chunk (default: 5000).")
    parser.add_argument("--out-root", default=None,
                        help="Parent dir for chunk folders (default: <source-dir>/chunks).")
    parser.add_argument("--force", action="store_true", help="Overwrite a non-empty destination folder.")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.is_file() and (Path("videos") / args.source).is_file():
        source = Path("videos") / args.source
    out_root = Path(args.out_root) if args.out_root else source.parent / "chunks"

    try:
        chunk_video(source, args.chunk_size, out_root, args.force)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
