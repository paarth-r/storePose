"""Aggregate noisy per-frame occupancy into stable per-window busy labels."""

from __future__ import annotations

from statistics import median

from ..queue.types import CompletedWait
from .types import BusyLevel, BusyThresholds, BusyWindow, WindowFeatures


def weighted_percentile(
    values: list[float], weights: list[float], q: float
) -> float:
    """Step-function weighted percentile (no interpolation).

    ``q`` in ``[0, 1]``. Returns the smallest value whose cumulative weight
    reaches ``q`` of the total. Robust to flicker because brief spikes carry
    little time weight. Empty input -> 0.0.
    """
    if not values:
        return 0.0
    pairs = sorted(zip(values, weights), key=lambda p: p[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return float(pairs[-1][0])
    target = q * total
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= target:
            return float(v)
    return float(pairs[-1][0])


# Each occupancy metric reduces a set of (occupancy, weight) samples to a scalar.
def _occ_stat(occ: list[float], wts: list[float], metric: str) -> float:
    if not occ:
        return 0.0
    if metric == "occupancy_p90":
        return weighted_percentile(occ, wts, 0.9)
    if metric == "occupancy_median":
        return weighted_percentile(occ, wts, 0.5)
    if metric == "occupancy_max":
        return max(occ)
    if metric == "occupancy_mean":
        total = sum(wts)
        return sum(o * w for o, w in zip(occ, wts)) / total if total > 0 else 0.0
    raise ValueError(f"not an occupancy metric: {metric!r}")


def classify(
    value: float, thresholds: BusyThresholds, prev: BusyLevel | None = None
) -> BusyLevel:
    """Map a metric value to a level, with optional hysteresis vs. ``prev``.

    Hysteresis applies a deadband around the boundary being crossed: the level
    only changes once ``value`` clears the relevant boundary by ``hysteresis``,
    which prevents windows hovering near a cut point from flipping.
    """
    low, med, h = thresholds.low_max, thresholds.medium_max, thresholds.hysteresis

    def raw(v: float) -> BusyLevel:
        if v <= low:
            return BusyLevel.LOW
        if v <= med:
            return BusyLevel.MEDIUM
        return BusyLevel.HIGH

    level = raw(value)
    if prev is None or h == 0 or level == prev:
        return level

    if level > prev:  # moving up: must clear the boundary just above prev by h
        boundary = low if prev == BusyLevel.LOW else med
        return level if value > boundary + h else prev
    # moving down: must drop below the boundary just below prev by h
    boundary = med if prev == BusyLevel.HIGH else low
    return level if value < boundary - h else prev


class BusyAggregator:
    """Collects per-frame occupancy samples and completed waits, then emits a
    :class:`BusyWindow` per fixed-length window.

    Online usage (live runner): call :meth:`observe` every frame and
    :meth:`add_wait` as waits complete, then :meth:`windows` at the end. Offline
    usage feeds reconstructed occupancy samples the same way.

    ``sub_window_seconds`` enables **two-level smoothing**: within each window the
    occupancy metric is computed per short sub-window, then the *median* of those
    sub-window values drives the label. This both suppresses second-scale flicker
    (the inner percentile) and stops one busy minute from dominating the window
    (the outer median) — a principled way to a stable label, per the brief.
    """

    def __init__(
        self,
        thresholds: BusyThresholds | None = None,
        window_seconds: float = 600.0,
        sub_window_seconds: float | None = None,
    ):
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        if sub_window_seconds is not None and sub_window_seconds <= 0:
            raise ValueError(
                f"sub_window_seconds must be > 0, got {sub_window_seconds}"
            )
        self.thresholds = thresholds or BusyThresholds()
        self.window_seconds = window_seconds
        self.sub_window_seconds = sub_window_seconds
        # window index -> list of (t, occupancy, dt) samples
        self._samples: dict[int, list[tuple[float, float, float]]] = {}
        self._throughput: dict[int, int] = {}
        self._wait_sum: dict[int, float] = {}
        self._arrivals: dict[int, int] = {}

    def _win(self, t: float) -> int:
        return int(t // self.window_seconds)

    def observe(self, t: float, occupancy: float, dt: float) -> None:
        """Record one occupancy sample at time ``t`` carrying weight ``dt``."""
        if dt <= 0:
            return
        self._samples.setdefault(self._win(t), []).append(
            (float(t), float(occupancy), float(dt))
        )

    def add_wait(self, wait: CompletedWait) -> None:
        """Attribute a completed wait: throughput/mean-wait to its exit window,
        arrival to its enter window."""
        ex = self._win(wait.exited_s)
        self._throughput[ex] = self._throughput.get(ex, 0) + 1
        self._wait_sum[ex] = self._wait_sum.get(ex, 0.0) + wait.wait_seconds
        en = self._win(wait.entered_s)
        self._arrivals[en] = self._arrivals.get(en, 0) + 1

    def _features(self, w: int) -> WindowFeatures:
        samples = self._samples.get(w, [])
        occ = [o for _, o, _ in samples]
        wts = [d for _, _, d in samples]
        tp = self._throughput.get(w, 0)
        return WindowFeatures(
            mean_occupancy=_occ_stat(occ, wts, "occupancy_mean"),
            median_occupancy=_occ_stat(occ, wts, "occupancy_median"),
            p90_occupancy=_occ_stat(occ, wts, "occupancy_p90"),
            max_occupancy=_occ_stat(occ, wts, "occupancy_max"),
            sample_seconds=sum(wts),
            throughput=tp,
            mean_wait_seconds=(self._wait_sum.get(w, 0.0) / tp) if tp else 0.0,
            arrivals=self._arrivals.get(w, 0),
        )

    def _metric_value(self, w: int, feats: WindowFeatures) -> float:
        """The scalar the label is based on, applying sub-window smoothing for
        occupancy metrics when enabled. Non-occupancy metrics (mean_wait) read the
        window feature directly."""
        metric = self.thresholds.metric
        if metric == "mean_wait":
            return feats.mean_wait_seconds
        if not self.sub_window_seconds:
            return _occ_stat(
                [o for _, o, _ in self._samples.get(w, [])],
                [d for _, _, d in self._samples.get(w, [])],
                metric,
            )
        # two-level: per sub-window stat, then median across sub-windows
        sub = self.sub_window_seconds
        buckets: dict[int, tuple[list[float], list[float]]] = {}
        win_start = w * self.window_seconds
        for t, o, d in self._samples.get(w, []):
            k = int((t - win_start) // sub)
            occ, wts = buckets.setdefault(k, ([], []))
            occ.append(o)
            wts.append(d)
        sub_values = [
            _occ_stat(occ, wts, metric) for occ, wts in buckets.values()
        ]
        return median(sub_values) if sub_values else 0.0

    def estimate(self, t: float) -> tuple[BusyLevel, float]:
        """Live ``(level, metric_value)`` for the in-progress window at ``t``.

        No hysteresis — this is an at-a-glance running estimate for the overlay,
        not a finalized label."""
        w = self._win(t)
        value = self._metric_value(w, self._features(w))
        return classify(value, self.thresholds), value

    def estimate_recent(
        self, t: float, lookback: float
    ) -> tuple[BusyLevel, float]:
        """Live ``(level, value)`` from only the trailing ``lookback`` seconds.

        Unlike :meth:`estimate` (which summarises the whole in-progress window and
        so barely moves once the window fills), this reflects *recent* activity, so
        the live badge tracks what is happening now. No hysteresis. For the
        ``mean_wait`` metric there is no trailing-occupancy analogue, so this falls
        back to :meth:`estimate`.
        """
        if lookback <= 0 or self.thresholds.metric == "mean_wait":
            return self.estimate(t)
        t0 = t - lookback
        occ: list[float] = []
        wts: list[float] = []
        for w in range(self._win(max(0.0, t0)), self._win(t) + 1):
            for ts, o, d in self._samples.get(w, []):
                if t0 <= ts <= t:
                    occ.append(o)
                    wts.append(d)
        value = _occ_stat(occ, wts, self.thresholds.metric)
        return classify(value, self.thresholds), value

    def subwindow_values(self) -> list[float]:
        """The metric's value for every sub-window across the whole clip.

        Buckets all collected samples by global sub-window index
        (``t // sub_window_seconds``), independent of the 10-minute window
        boundaries, and applies the configured occupancy metric per bucket. This
        is the sample distribution used to *calibrate* the busy thresholds.
        Requires ``sub_window_seconds`` to be set and an occupancy metric.
        """
        sub = self.sub_window_seconds
        if not sub:
            raise ValueError("subwindow_values requires sub_window_seconds")
        metric = self.thresholds.metric
        if metric == "mean_wait":
            raise ValueError("subwindow_values needs an occupancy metric, not mean_wait")
        buckets: dict[int, tuple[list[float], list[float]]] = {}
        for samples in self._samples.values():
            for t, o, d in samples:
                k = int(t // sub)
                occ, wts = buckets.setdefault(k, ([], []))
                occ.append(o)
                wts.append(d)
        return [_occ_stat(occ, wts, metric) for _, (occ, wts) in sorted(buckets.items())]

    def _window_indices(self) -> list[int]:
        keys = set(self._samples) | set(self._throughput) | set(self._arrivals)
        return sorted(keys)

    def windows(self) -> list[BusyWindow]:
        """Finalize every window seen so far, in time order, with hysteresis."""
        out: list[BusyWindow] = []
        prev: BusyLevel | None = None
        for w in self._window_indices():
            feats = self._features(w)
            value = float(self._metric_value(w, feats))
            level = classify(value, self.thresholds, prev)
            prev = level
            out.append(
                BusyWindow(
                    index=w,
                    start_s=w * self.window_seconds,
                    end_s=(w + 1) * self.window_seconds,
                    level=level,
                    metric=self.thresholds.metric,
                    metric_value=value,
                    features=feats,
                )
            )
        return out
