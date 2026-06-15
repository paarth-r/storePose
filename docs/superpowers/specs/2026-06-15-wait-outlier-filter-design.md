# Wait-time outlier filter — design

**Date:** 2026-06-15
**Status:** approved (brainstorming) → ready for implementation

## Problem

A flaky detection can produce a phantom "visit" that appears and "checks out" in
~2s, when real sales cluster near ~23s. These short phantoms pollute the wait log
and drag down the live busy signal (throughput up, mean-wait down). We want to
reject implausibly short completed visits.

## Decisions (from brainstorming)

- **Duration tested:** total time present, `exited − entered`.
- **Per outcome:** judge each visit against the typical duration for its **own**
  outcome (`served` / `served_other` / `abandoned`), since their normals differ.
- **Rule:** floor + relative median — reject if
  `duration < max(reject_floor, reject_frac × median[outcome])`.
- **Where:** online, between the analyzer and its consumers; per-outcome running
  median; **floor only** until `reject_warmup` samples exist for that outcome.
  Built as a reusable component so the same logic can scrub a CSV offline.
- **Disposition:** flag, don't destroy. Keep the visit (with its true outcome) in
  the wait log marked `rejected`; exclude it from the busy aggregator + dashboard;
  rejected durations do **not** update the median (phantoms can't drag it down).
- **Default off:** active only with `--reject-short`, so current behavior is
  unchanged unless opted in.

## Architecture

### `OutlierFilter` — `src/storepose/queue/outliers.py` (pure, reusable)

- State: `windows: dict[str, deque[float]]` — accepted durations per outcome,
  `maxlen ≈ 200` (trailing, bounded; adapts and caps memory).
- Construction: `OutlierFilter(floor, frac, warmup)`.
- `judge(wait: CompletedWait) -> CompletedWait`:
  - `duration = wait.exited_s − wait.entered_s`.
  - `win = windows[wait.outcome]`.
  - `threshold = max(floor, frac × median(win))` if `len(win) >= warmup` else `floor`.
  - If `duration < threshold`: return `replace(wait, rejected=True)` — **do not**
    update the window.
  - Else: `win.append(duration)`; return the wait unchanged (`rejected=False`).
- No global mutable state beyond the per-outcome windows; deterministic given the
  stream order.

### `CompletedWait` — `src/storepose/queue/types.py`

Add `rejected: bool = False` (keeps the true `outcome` intact — cleaner than
clobbering it with a synthetic `false_short`).

### Runner wiring — `runner.py`

- Build the filter once when `config.reject_short` (else `None`).
- Per completed wait `c`: `c = filt.judge(c)` when the filter is active.
- Wait log: always write `c`, now including a `rejected` column.
- `busy.add_wait(c)` and `dashboard.add_visit(...)`: only when **not**
  `c.rejected` (combined with the existing `outcome in (served, served_other)`
  gate for busy throughput).

### Wait-log format — `runner.py` writer + `busy/report.py` reader

- Writer header gains `rejected` (`"1"`/`"0"`).
- `read_waits` reads `rejected` with a default of `False` (backward compatible
  with older logs lacking the column).

## Config (`config.py`)

- `reject_short: bool = False` + `--reject-short`.
- `reject_floor: float = 2.0` + `--reject-floor` (seconds). Validate `>= 0`.
- `reject_frac: float = 0.25` + `--reject-frac` (fraction of median).
  Validate `0 <= frac <= 1`.
- `reject_warmup: int = 10` + `--reject-warmup` (min samples per outcome before
  the relative term applies). Validate `>= 0`.

## Data flow

```
analyzer.completed[c]
  -> (if --reject-short) c = filter.judge(c)
  -> wait_log.write(c)                      # always, incl. rejected column
  -> if not c.rejected:
        busy.add_wait(c)  (served/served_other only, as today)
        dashboard.add_visit(c)
```

## Testing

`tests/queue/test_outliers.py` (pure):
- Floor-only before warm-up: `duration < floor` rejected; `>= floor` accepted
  regardless of (insufficient) samples.
- Relative after warm-up: feed `warmup` ~20s `served` visits, then 2s → rejected
  (`2 < max(2, 0.25·20)=5`), 6s → accepted.
- Per-outcome independence: a long `served` median doesn't change `abandoned`
  judging.
- Rejected durations don't update the window (feed phantoms; median holds).
- `replace`-based immutability: input wait object not mutated.

`tests/` runner/integration: a rejected visit is not fed to the busy aggregator;
`--reject-short` off → all visits pass through unchanged. `read_waits` tolerates
logs with and without the `rejected` column.

## Out of scope (v1)

- An offline `busy_report` subcommand to scrub recorded CSVs (filter is built
  reusable; wiring a CLI is later).
- Upper-bound / too-long outlier rejection.
- Cross-outcome or per-view-calibrated thresholds.
