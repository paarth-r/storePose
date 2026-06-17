# POS occlusion-tolerant re-assignment (non-Mashgin first)

## Problem

A serve at a checkout is keyed by the tracker's track id. When the served
person is occluded and the tracker drops the id (appearance re-id fails), the
`QueueAnalyzer` sees the track vanish, finalizes the visit after the re-id
grace, and a later apparition mints a *fresh* serve. One real serve fragments
into two, corrupting serving-time stats (and the busy/throughput signals built
on them).

## Goal

Distinguish a *clear exit* from an *occlusion vanish*. On an occlusion vanish at
a participating checkout, do not finalize the visit — **park** it, and let the
next apparition entering that checkout **adopt** it and continue the timer.

Implement for the non-Mashgin (`"other"`) checkout first, structured so enabling
Mashgin is a one-line / one-flag change.

## Definitions and decisions

1. **Clear exit** — the track was present and accrued `out_streak >=
   exit_seconds` (we watched them step out of every checkout zone), or it
   switched to the other checkout. These finalize exactly as today.
2. **Occlusion vanish** — the visit was in `serving` state for a participating
   checkout and the track disappeared (`absent_seconds` reached
   `reid_grace_seconds`) without a clear exit. This is parked instead of
   finalized.
3. **Park pool** — keyed by checkout string. A `_ParkedVisit` keeps
   `serving_seconds`, `waiting_seconds`, `entered_s`, the last-seen box center,
   and an age. It **keeps accruing serving time** while parked (continued timer,
   matching today's re-id-grace accrual).
4. **Adoption** — at any `-> serving("other")` transition for a *fresh* visit
   (no prior serving history on that id), pick the nearest parked entry for that
   checkout whose last center is within a spatial radius and whose age is below
   the hold timeout; transfer its accumulated time into the new id's visit state
   and consume the parked entry.
5. **Hold timeout** — new config `pos_reassign_seconds` (default `20.0`; `<= 0`
   disables the feature). On expiry with no adoption, the parked visit finalizes
   with its accrued time as `served_other`.
6. **Participating checkouts** — an analyzer set, default `{"other"}`. A
   `--pos-reassign-mashgin` flag adds `"mashgin"`. The mechanism is
   checkout-agnostic; nothing else changes to extend it.
7. **Spatial gate** — adoption requires the new apparition's entry center to lie
   within a radius of the parked entry's last center. Radius is a fraction of
   the frame diagonal; with multiple parked entries the nearest within radius
   wins. (Mirrors the tracker's re-attach spatial gate in spirit.)
8. **Appearance** — NOT used in v1 (per decision: zone + space + time only). The
   `_ParkedVisit` and adoption path are shaped so a relaxed appearance gate can
   be added later without restructuring.

## Live-count behavior

Parked (occluded) people have no current box, so they are not emitted in the
per-frame `statuses` and the live "at REG" (`serving_other_count`) may dip while
someone is occluded. The preserved quantity is the *measured serving time*,
which is the point of the feature. Acceptable for the stated goal.

## Where it lives

All in `queue/analyzer.py`:
- a `_ParkedVisit` dataclass and a `self._parked: dict[str, list[_ParkedVisit]]`
  pool on `QueueAnalyzer`;
- park instead of finalize at the vanish path (the absent >= grace branch) when
  the state is `serving` on a participating checkout;
- attempt adoption at the `-> serving` transitions when `which` is a
  participating checkout and the visit is fresh;
- age parked entries each `update`, expiring (finalizing) past the hold timeout.

New config in `config.py`: `pos_reassign_seconds: float = 20.0` and
`pos_reassign_mashgin: bool = False`, wired through `build_analyzer` in
`runner.py` into the analyzer's `reassign_checkouts` set + `reassign_seconds`.

A frame diagonal / size is needed for the spatial radius. The analyzer does not
currently know frame size; the radius will be computed from the zone/box scale
already available (box height-based, consistent with the transit filter's
normalization) rather than introducing a frame-size dependency. Concretely: the
spatial gate uses an absolute pixel radius derived from a configurable fraction
of the parked person's last box diagonal, falling back to a fixed default.

## Testing (`tests/queue/test_analyzer.py`)

1. Vanish at `other` while serving -> visit parked, not finalized.
2. New apparition near the parked center adopts it -> a single `served_other`
   completion with summed serving time, emitted when that visit ends.
3. Hold timeout expiry with no adoption -> finalize with the accrued time.
4. A clear walk-out (out_streak >= exit_seconds while present) still finalizes
   normally; nothing is parked.
5. Mashgin serves are unaffected by default (not parked, not adopted).
6. With `--pos-reassign-mashgin`, a Mashgin vanish parks/adopts symmetrically.
7. Spatial gate: an apparition far from the vanish point does NOT adopt; it
   starts a fresh serve and the parked entry expires on its own.
8. Feature disabled (`pos_reassign_seconds <= 0`): behaves exactly as today.

## Accepted trade-off

Without appearance, a different person stepping into the non-Mashgin POS near a
recent occlusion point within the hold window inherits the timer. The spatial
gate plus the short window bound this; revisit with the lowered-appearance gate
if it proves to be a problem.
