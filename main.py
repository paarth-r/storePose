"""storePose CLI entrypoint."""

from __future__ import annotations

import sys

from storepose.config import from_args
from storepose.runner import Runner
from storepose.video_source import CameraOpenError


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the realtime pose loop, return a process exit code."""
    config = from_args(argv)
    if config.define_zone:
        from storepose.queue.zone_editor import define_zones
        saved = define_zones(config.source, config.zone, config.pos_zone)
        parts = []
        if "line" in saved:
            parts.append(f"--zone {saved['line']}")
        if "pos" in saved:
            parts.append(f"--pos-zone {saved['pos']}")
        print("Run with: " + " ".join(parts) if parts else "Nothing saved.")
        return 0
    if config.define_pos_zone:
        from storepose.queue.zone_editor import define_zones
        saved = define_zones(config.source, pos_path=config.pos_zone, pos_only=True)
        print(f"Run with: --pos-zone {saved['pos']}" if "pos" in saved else "Nothing saved.")
        return 0
    try:
        Runner(config).run()
    except CameraOpenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
