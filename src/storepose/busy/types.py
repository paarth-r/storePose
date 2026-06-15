"""Types for the busy-signal layer: noisy per-frame occupancy -> stable label.

The deliverable is a Low / Medium / High "busy" label for every fixed window
(10 minutes by default). The per-frame occupancy (people waiting in the zone)
flickers, so each window is summarised by a *robust* statistic and mapped to a
level through configurable thresholds, with hysteresis across windows so the
output does not oscillate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class BusyLevel(IntEnum):
    """Ordinal busy level. Integer values let evaluation treat it as ordinal."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2

    @property
    def label(self) -> str:
        return self.name.capitalize()


# Which window feature drives the label. Keep names stable; they are also the
# accepted ``--busy-metric`` CLI values and map to WindowFeatures attributes.
METRICS = (
    "occupancy_p90",
    "occupancy_mean",
    "occupancy_median",
    "occupancy_max",
    "mean_wait",
)

_METRIC_ATTR = {
    "occupancy_p90": "p90_occupancy",
    "occupancy_mean": "mean_occupancy",
    "occupancy_median": "median_occupancy",
    "occupancy_max": "max_occupancy",
    "mean_wait": "mean_wait_seconds",
}

# Calibration band strategies (see busy.calibrate). Also the accepted
# ``--busy-strategy`` CLI values.
BUSY_STRATEGIES = ("skewed", "thirds", "peak")
DEFAULT_BUSY_STRATEGY = "skewed"


@dataclass(frozen=True)
class BusyThresholds:
    """Maps a window's summary statistic to a :class:`BusyLevel`.

    ``value <= low_max`` -> LOW; ``value <= medium_max`` -> MEDIUM; else HIGH.

    Defaults are *placeholders*: the right cut points depend on the store and on
    what Low/Medium/High should mean, and must be calibrated on real footage
    against a labeled evaluation set. See ``docs/problem-definition.md``.

    Attributes:
        metric: Which window feature drives the label (one of :data:`METRICS`).
        low_max: Upper bound (inclusive) of the LOW band, in the metric's units.
        medium_max: Upper bound (inclusive) of the MEDIUM band.
        hysteresis: Deadband, in the metric's units, a value must clear *beyond*
            a boundary before the level is allowed to change vs. the previous
            window. ``0`` disables hysteresis.
    """

    metric: str = "occupancy_p90"
    low_max: float = 1.0
    medium_max: float = 3.0
    hysteresis: float = 0.0

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {self.metric!r}")
        if self.medium_max < self.low_max:
            raise ValueError(
                f"medium_max ({self.medium_max}) must be >= low_max ({self.low_max})"
            )
        if self.hysteresis < 0:
            raise ValueError(f"hysteresis must be >= 0, got {self.hysteresis}")

    def metric_attr(self) -> str:
        return _METRIC_ATTR[self.metric]


@dataclass
class WindowFeatures:
    """Aggregated statistics for one window.

    Occupancy stats are time-weighted (each sample weighted by its frame ``dt``),
    so they are not biased by variable frame rate. Wait-derived extras come from
    completed waits attributed to the window.
    """

    mean_occupancy: float = 0.0
    median_occupancy: float = 0.0
    p90_occupancy: float = 0.0
    max_occupancy: float = 0.0
    sample_seconds: float = 0.0
    throughput: int = 0          # waits that completed (exited) in this window
    mean_wait_seconds: float = 0.0
    arrivals: int = 0            # waits that started (entered) in this window


@dataclass
class BusyWindow:
    """A finalized busy label for one window."""

    index: int
    start_s: float
    end_s: float
    level: BusyLevel
    metric: str
    metric_value: float
    features: WindowFeatures
