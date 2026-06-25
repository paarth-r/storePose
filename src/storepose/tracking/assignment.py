"""IoU-based detection-to-track association (Hungarian)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# BoT-SORT gated-minimum fusion (Aharon 2022): appearance is admitted only when
# a pair is both visually close (cosine distance < gate) AND spatially close
# (IoU > 1 - gate); admitted appearance is halved, then cost = min(d_iou, d_cos).
# So a weak embedding can only ever lower the cost of a geometrically-plausible
# match -- it can never override geometry to merge two distant look-alikes.
_BOTSORT_COS_GATE = 0.25   # admit appearance only when cosine distance < this (sim > 0.75)
_BOTSORT_IOU_GATE = 0.5    # ...and d_iou < this (IoU > 0.5)
_BOTSORT_APP_SCALE = 0.5   # admitted appearance distance is halved


def _botsort_cost(m: np.ndarray, appsim: np.ndarray | None) -> np.ndarray:
    """BoT-SORT gated-minimum cost from an IoU matrix and cosine ``appsim``."""
    d_iou = 1.0 - m
    if appsim is None:
        return d_iou
    d_cos = 1.0 - np.clip(appsim, -1.0, 1.0)          # cosine distance in [0, 2]
    admit = (d_cos < _BOTSORT_COS_GATE) & (d_iou < _BOTSORT_IOU_GATE)  # NaN -> False
    d_cos_hat = np.where(admit, _BOTSORT_APP_SCALE * d_cos, 1.0)
    return np.minimum(d_iou, d_cos_hat)


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
    dets: list[np.ndarray], tracks: list[np.ndarray], iou_thr: float,
    *, appsim: np.ndarray | None = None, app_weight: float = 0.0,
    app_floor: float = 0.0, motsim: np.ndarray | None = None,
    mot_weight: float = 0.0, fusion: str = "sum",
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Associate detections to tracks, gated by ``iou_thr``.

    The assignment maximizes a blend of three cues, any of which can be off:
    - **IoU** (box overlap) — always.
    - **Appearance** (``appsim``, cosine in ``[-1, 1]``, ``NaN`` where unknown;
      weight ``app_weight``) — StrongSORT/BoT-SORT style "looks like". A pair
      below ``app_floor`` is rejected even at high IoU, so a person passing a
      fixed prop isn't matched onto it.
    - **Motion direction** (``motsim``, cosine of the angle between a track's
      heading and the direction to the detection, ``NaN`` where undefined;
      weight ``mot_weight``) — OC-SORT-style "went the right way", which breaks
      a crossing tie when geometry and appearance can't.

    Each cue's ``NaN`` entries fall back to IoU for that pair (neutral). All
    weights zero / matrices ``None`` reproduces plain IoU matching.

    ``fusion`` selects how the cues combine:
    - ``"sum"`` (default): the weighted blend above (additive; appearance can
      pull a match up or, via ``app_floor``, veto it).
    - ``"botsort"``: BoT-SORT gated-minimum ``cost = min(d_iou, gated d_cos)``,
      where appearance is admitted only when both spatially and visually close
      (see ``_botsort_cost``). Appearance can only *help* a plausible match,
      never override geometry. ``motsim`` is unused in this mode.

    The ``iou_thr`` and ``app_floor`` post-filters apply in both modes. Returns
    ``(matches, unmatched_dets, unmatched_tracks)``.
    """
    if len(dets) == 0 or len(tracks) == 0:
        return [], list(range(len(dets))), list(range(len(tracks)))

    m = iou_matrix(dets, tracks)
    if fusion == "botsort":
        rows, cols = linear_sum_assignment(_botsort_cost(m, appsim))  # minimize cost
    else:
        app_w = app_weight if appsim is not None else 0.0
        mot_w = mot_weight if motsim is not None else 0.0
        if app_w > 0 or mot_w > 0:
            iou_w = max(0.0, 1.0 - app_w - mot_w)
            score = iou_w * m
            if app_w > 0:
                an = (np.clip(appsim, -1.0, 1.0) + 1.0) / 2.0  # cosine -> [0, 1]
                score = score + app_w * np.where(np.isnan(an), m, an)
            if mot_w > 0:
                mn = (np.clip(motsim, -1.0, 1.0) + 1.0) / 2.0
                score = score + mot_w * np.where(np.isnan(mn), m, mn)
        else:
            score = m
        rows, cols = linear_sum_assignment(-score)  # maximize the (blended) score

    matches: list[tuple[int, int]] = []
    unmatched_dets = set(range(len(dets)))
    unmatched_tracks = set(range(len(tracks)))
    for r, c in zip(rows, cols):
        if m[r, c] < iou_thr:
            continue  # proximity sanity always applies
        if appsim is not None and not np.isnan(appsim[r, c]) and appsim[r, c] < app_floor:
            continue  # appearance veto: looks nothing like this track
        matches.append((int(r), int(c)))
        unmatched_dets.discard(int(r))
        unmatched_tracks.discard(int(c))
    return matches, sorted(unmatched_dets), sorted(unmatched_tracks)
