"""CSV I/O for the busy layer: read a wait log, write/read a busy report."""

from __future__ import annotations

import csv
from pathlib import Path

from ..queue.types import CompletedWait
from .types import BusyLevel, BusyWindow

WAIT_FIELDS = ("id", "entered_s", "exited_s", "wait_seconds")

BUSY_FIELDS = (
    "window_index",
    "start_s",
    "end_s",
    "level",
    "metric",
    "metric_value",
    "mean_occupancy",
    "median_occupancy",
    "p90_occupancy",
    "max_occupancy",
    "throughput",
    "mean_wait_seconds",
    "arrivals",
    "sample_seconds",
)


def read_waits(path: str | Path) -> list[CompletedWait]:
    """Load completed waits written by ``--wait-log`` (header required)."""
    waits: list[CompletedWait] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            waits.append(
                CompletedWait(
                    id=int(row["id"]),
                    entered_s=float(row["entered_s"]),
                    exited_s=float(row["exited_s"]),
                    wait_seconds=float(row["wait_seconds"]),
                )
            )
    return waits


def write_busy(path: str | Path, windows: list[BusyWindow]) -> None:
    """Write one row per window with the label and its supporting features."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BUSY_FIELDS)
        for bw in windows:
            ft = bw.features
            w.writerow(
                [
                    bw.index,
                    f"{bw.start_s:.2f}",
                    f"{bw.end_s:.2f}",
                    bw.level.label,
                    bw.metric,
                    f"{bw.metric_value:.4f}",
                    f"{ft.mean_occupancy:.4f}",
                    f"{ft.median_occupancy:.4f}",
                    f"{ft.p90_occupancy:.4f}",
                    f"{ft.max_occupancy:.4f}",
                    ft.throughput,
                    f"{ft.mean_wait_seconds:.4f}",
                    ft.arrivals,
                    f"{ft.sample_seconds:.2f}",
                ]
            )


def write_levels(path: str | Path, levels: dict[int, BusyLevel]) -> None:
    """Write a minimal ``window_index,level`` CSV (ground-truth label format)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_index", "level"])
        for idx in sorted(levels):
            w.writerow([idx, levels[idx].label])


def read_busy_levels(path: str | Path) -> dict[int, BusyLevel]:
    """Read a busy report (predicted or ground truth) into ``{window_index:
    level}``. Accepts level names case-insensitively (Low/Medium/High)."""
    out: dict[int, BusyLevel] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["window_index"])
            out[idx] = BusyLevel[row["level"].strip().upper()]
    return out
