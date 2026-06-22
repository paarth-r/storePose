# storePose param-tuner — design

**Date:** 2026-06-22
**Status:** approved (design), pending implementation plan

## Goal

A system that tunes the parameters of the storePose computer-vision pipeline:
it understands what each parameter does, searches for better values, benchmarks
candidates against ground-truth data, and adjusts its search based on observed
failure modes — instead of a human hand-twiddling `det_conf`/`iou_thr`/busy-band
values one run at a time.

This is a **hybrid**: a deterministic Python search harness (`storepose.tune`)
that does the actual sweeping and scoring, plus a Claude Code subagent
(`cv-param-tuner`) that drives it, reads the failure modes, and decides what to
search next.

## Scope (v1)

- **In:** tuning against the *existing* busy-level ground truth (the
  `window_index,level` truth CSVs that `busy_report eval` already reads); Optuna
  search; three-tier eval cache; a pluggable scorer with `BusyLevelScorer` wired
  up; ground-truth discovery; the driving subagent.
- **Out (deferred until ground truth exists):** queue/serve-timing objective,
  tracking/ReID-quality objective, throughput/FPS objective. The scorer
  interface is designed so each is *one new class* later — but none ship in v1.
  The user will annotate the additional ground truth before those are built.

## Non-obvious facts this design depends on

- The scoring path already exists: `busy_report aggregate` turns a wait-log CSV
  + band thresholds into per-window busy labels; `busy_report eval pred.csv
  truth.csv` scores them via `storepose.eval.metrics.evaluate`, returning an
  `EvalReport` (accuracy, within-1 accuracy, ordinal MAE, confusion matrix,
  per-class precision/recall).
- A normal `Runner.run()` **always opens a cv2 window** (`imshow`/`waitKey`);
  only `--calibrate` is headless and it does not emit a wait-log. Therefore the
  harness does **not** shell out to `main.py` for full runs — it drives the
  pipeline stages programmatically in its own headless loop (see Tier 2).
- `--wait-log PATH` and the busy-band params already exist on `AppConfig`.

## Architecture

Two pieces with a clean seam:

1. **`storepose.tune` — Python harness.** Deterministic, LLM-free, importable
   and runnable standalone (`uv run python -m storepose.tune ...`). Owns the
   param registry, the three-tier eval cache, the Optuna search loop, the
   pluggable scorer, ground-truth discovery, and a results store.
2. **`cv-param-tuner` — Claude Code subagent** (`.claude/agents/cv-param-tuner.md`).
   The judgment layer: scopes a search to the user's goal, launches studies via
   Bash, reads the confusion matrix / leaderboard, diagnoses failure modes,
   refines the search space, and finally applies + explains the winning config.

## Components

### `tune/space.py` — parameter registry

One declarative table; the single source of truth for what is tunable. Each
entry: `name`, `type` (`float`/`int`/`bool`/`categorical`), bounds or choices,
and **tier** (0/1/2). A test asserts every registry `name` is a real
`AppConfig` field so the registry cannot drift silently from `config.py`.

A "search request" is a subset of registry names plus optional per-param range
overrides. Both Optuna and the subagent consume the registry; neither may search
a param not in it.

Tier assignment (drives eval cost — see cache):

- **Tier 0 (busy bands):** `busy_metric`, `busy_low_max`, `busy_medium_max`,
  `busy_hysteresis`, `busy_window`.
- **Tier 1 (tracking/zone/queue):** `iou_thr`, `max_overlap`, `min_hits`,
  `hold_seconds`, `reid_thr`, `reid_seconds`, `smooth_cutoff`, `smooth_beta`,
  `wait_enter_frames`, `pos_enter_frames`, `transit_speed`, `wait_exit_seconds`,
  `zone_coverage`, `zone_foot_band`, `pos_reassign_seconds`, etc.
- **Tier 2 (detection):** `det_conf`, `det_overlap`, `kpt_thr`, `mode`.

### `tune/runner.py` + `tune/cache.py` — three-tier eval cache

The harness inspects the search space, finds the highest tier any chosen param
touches, and pays only that cost:

- **Tier 2 — detection.** Drive the pipeline stages (`detector` → `pose` →
  `pipeline`) in a headless loop (no cv2 window), persisting **raw per-frame
  detections + poses** to disk, keyed by `(clip, detection-param hash)`.
- **Tier 1 — tracking/zone/queue.** Replay the cached raw outputs through
  tracking → queue → wait-log (no pose cost).
- **Tier 0 — busy bands.** Replay a fixed wait-log through `busy_report
  aggregate`. Sub-second.

Cache keys are param hashes, so repeated combos are free. A study touching only
Tier-0 params never re-runs pose; a study touching a Tier-2 param refreshes the
raw cache once per detection-config and reuses it across all downstream combos.

### `tune/scoring.py` — pluggable scorer

```
class Scorer(Protocol):
    def score(self, run_outputs, ground_truth) -> ScoreResult: ...

@dataclass
class ScoreResult:
    primary: float          # maximize
    details: dict           # confusion matrix, per-class P/R, MAE, ...
```

**v1 ships only `BusyLevelScorer`**, wrapping `eval.metrics.evaluate`. Default
`primary` is a composite that rewards accuracy and penalizes gross ordinal
errors (Low↔High) via MAE — exact weighting decided in the plan, exposed as a
config knob. Adding a queue/tracking/throughput objective later is one new
`Scorer` class with no harness changes.

### `tune/dataset.py` — ground-truth discovery

Scans the repo for truth CSVs matching the `window_index,level` schema and pairs
each with its source video by name/dir convention, yielding `(clip, truth)`
pairs. Multi-clip studies score on the **pooled** confusion matrix so the tuner
optimizes across clips rather than overfitting one video. If discovery finds no
usable GT, the harness fails loudly with guidance (point at the `label`
command).

### `tune/search.py` — Optuna search loop

An Optuna study over the requested sub-space; each trial = sample params → run
at the appropriate tier → score → record. Results (params, `primary`, `details`)
persist to a store under `runs/tune/` (Optuna SQLite + a JSON leaderboard).
Supports resume, an n-trials or wall-clock budget, and emits a ranked
leaderboard plus the best trial's confusion matrix. Optuna is added to
`pyproject.toml`.

### `.claude/agents/cv-param-tuner.md` — driving subagent

Loop it follows:

1. Read the registry and discover ground truth.
2. Propose an initial search space scoped to the user's stated goal.
3. Launch a study via Bash.
4. Read the leaderboard + best confusion matrix and **diagnose failure modes**
   (e.g. "systematically over-predicting HIGH → MEDIUM band too low *or*
   `det_conf` too permissive"; "Low↔High confusions dominate MAE → check
   `busy_metric` choice").
5. Refine the space (widen/narrow ranges, swap params) and re-run, or stop when
   gains plateau / budget is hit.
6. Apply the winning params — write a config snippet / the exact CLI flags — and
   summarize what changed and why, honestly flagging when GT is too thin to draw
   a conclusion.

Guardrails: never invents params outside the registry; never claims an
improvement it didn't measure; reports the GT size alongside any score.

## Testing (TDD)

Pure logic gets real unit tests; the expensive pose pass is mocked at the cache
boundary so the suite stays fast:

- registry ↔ `AppConfig` consistency (every name is a real field; tiers valid).
- tier selection from a given search space.
- cache-key hashing (same params → same key; different → different).
- scorer math on synthetic confusion matrices (composite `primary`, MAE,
  within-1).
- GT discovery/pairing on a fixture tree (schema match, video pairing, pooling).
- the Optuna loop against a tiny mock objective (converges, respects budget,
  resumes).

## Open items for the implementation plan

- Exact composite-score weighting and how it is exposed.
- Concrete per-param bounds in the registry.
- Cache on-disk format for raw detections+poses.
- Where the headless processing loop lives (`tune/runner.py` reusing pipeline
  modules vs. a thin shared helper extracted from `Runner`).
