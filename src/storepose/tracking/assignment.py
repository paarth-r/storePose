"""IoU-based detection-to-track association (Hungarian)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two ``xyxy`` boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(dets: list[np.ndarray], tracks: list[np.ndarray]) -> np.ndarray:
    """``(len(dets), len(tracks))`` matrix of pairwise IoU."""
    m = np.zeros((len(dets), len(tracks)), dtype=np.float32)
    for i, d in enumerate(dets):
        for j, t in enumerate(tracks):
            m[i, j] = iou(d, t)
    return m


def match(
    dets: list[np.ndarray], tracks: list[np.ndarray], iou_thr: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Associate detections to tracks by maximum IoU, gated by ``iou_thr``.

    Returns ``(matches, unmatched_dets, unmatched_tracks)`` where ``matches`` is
    a list of ``(det_index, track_index)`` pairs.
    """
    if len(dets) == 0 or len(tracks) == 0:
        return [], list(range(len(dets))), list(range(len(tracks)))

    m = iou_matrix(dets, tracks)
    rows, cols = linear_sum_assignment(-m)  # maximize total IoU

    matches: list[tuple[int, int]] = []
    unmatched_dets = set(range(len(dets)))
    unmatched_tracks = set(range(len(tracks)))
    for r, c in zip(rows, cols):
        if m[r, c] >= iou_thr:
            matches.append((int(r), int(c)))
            unmatched_dets.discard(int(r))
            unmatched_tracks.discard(int(c))
    return matches, sorted(unmatched_dets), sorted(unmatched_tracks)
