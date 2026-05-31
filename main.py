"""storePose CLI entrypoint."""

from __future__ import annotations

import sys

from storepose.config import from_args
from storepose.runner import Runner
from storepose.video_source import CameraOpenError


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the realtime pose loop, return a process exit code."""
    config = from_args(argv)
    try:
        Runner(config).run()
    except CameraOpenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
