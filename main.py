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
        saved = define_zones(config.source, config.zone, config.pos_zone,
                             config.alt_zone, config.blur_zone)
        flags = {"line": "--zone", "pos": "--pos-zone", "alt": "--alt-zone",
                 "blur": "--blur-zone"}
        parts = [f"{flags[t]} {saved[t]}" for t in ("line", "pos", "alt", "blur")
                 if t in saved]
        print("Run with: " + " ".join(parts) if parts else "Nothing saved.")
        return 0
    if config.define_pos_zone:
        from storepose.queue.zone_editor import define_zones
        saved = define_zones(config.source, pos_path=config.pos_zone, only="pos")
        print(f"Run with: --pos-zone {saved['pos']}" if "pos" in saved else "Nothing saved.")
        return 0
    if config.define_alt_zone:
        from storepose.queue.zone_editor import define_zones
        saved = define_zones(config.source, alt_path=config.alt_zone, only="alt")
        print(f"Run with: --alt-zone {saved['alt']}" if "alt" in saved else "Nothing saved.")
        return 0
    if config.define_blur_zone:
        from storepose.queue.zone_editor import define_zones
        saved = define_zones(config.source, blur_path=config.blur_zone, only="blur")
        print(f"Run with: --blur-zone {saved['blur']}" if "blur" in saved else "Nothing saved.")
        return 0
    if config.define_ignore_zone:
        from storepose.queue.zone_editor import define_zone, default_ignore_zone_path
        out_path = config.ignore_zone or default_ignore_zone_path(config.source)
        saved = define_zone(config.source, out_path)
        print(f"Run with: --ignore-zone {saved}")
        return 0
    if config.calibrate:
        from storepose.busy.calibrate import calibrate
        try:
            calibrate(config)
        except (CameraOpenError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    try:
        Runner(config).run()
    except CameraOpenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
