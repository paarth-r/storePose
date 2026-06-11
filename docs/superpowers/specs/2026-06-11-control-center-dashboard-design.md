# Control Center Dashboard — Design Spec

*2026-06-11 · a multi-project, in-iTerm control center that mirrors Claude Code work*

## Purpose

A full-screen terminal dashboard (launched on demand, k9s/lazygit style) that
gives a single live view across **all** of Paarth's projects: curated roadmaps
and goals, plus live Claude Code session activity, background subagents, and git
state. Any Claude session reports into it automatically; the dashboard reflects
state. View-only first, with the data model built so bidirectional **control**
can be added later without a rewrite.

## Decisions (locked)

- **Scope:** general, multi-project (storePose, Hyperform, Jarvis, dumE, FRC, …),
  not one project.
- **Form factor:** full-screen Textual TUI, launched on demand. Framework:
  Textual (matches the Python/uv stack).
- **Direction:** view-only v1; data model designed so control is additive later.
- **Reporting model:** split. **Hooks** auto-report liveness (session start/stop,
  current task, subagents) with zero effort; a **skill** handles deliberate
  curation (roadmap items, goals, milestones).
- **Tracked entities:** curated roadmap + goals, live Claude sessions, background
  subagents, live git/branch state. *(Scheduled routines/crons intentionally out
  of v1.)*
- **Storage:** central **SQLite** (WAL mode) + an append-only event log. Chosen
  over flat files specifically for safe concurrent writers (many sessions + hooks
  writing at once) and trivial cross-project queries.

## Architecture

One DB, one write-path, one reader. The core discipline: **all SQL lives behind
a single thin CLI** so hooks, the skill, and future control share one narrow
interface.

```
  hooks (lifecycle)  ─┐
  skill (curation)   ─┼─►  cc-control CLI  ─►  control.db (SQLite, WAL)
                      │     (the only writer)         ▲
                      │                               │ read (poll/refresh)
                                          Textual TUI ┘  + live git shell-outs
```

### Components (each one job, clean interface)

1. **`control.db` schema module** — owns table definitions + idempotent
   migrations. The contract everything depends on.
2. **`cc-control` CLI** — the *only* code that writes SQL. Subcommands for every
   event. Thin, fast, near-zero deps so hooks add no latency. **Must never error
   a session**: a locked/missing/corrupt DB → log and exit 0.
3. **Hooks** — glue in `settings.json` (SessionStart, Stop/UserPromptSubmit for
   heartbeat, SubagentStop, SessionEnd) that call `cc-control`. Minimal logic.
4. **`control-center` skill** — loaded by any session. Teaches Claude when/how to
   register the project and push deliberate roadmap/goal/milestone updates.
5. **Textual TUI** — read-only renderer, launched on demand. Polls the DB, lays
   out projects → roadmap/goals/sessions/subagents.
6. **Git reader** — small module the TUI uses for branch / dirty / ahead-behind
   per project path, read **live** (not stored — repo state is already a source
   of truth; mirroring it just invites staleness).

## Data model

```sql
projects(id, name, repo_path UNIQUE, created_at)

roadmap_items(id, project_id, title, body, status,        -- todo|in_progress|done|blocked
              priority, created_at, updated_at)

goals(id, project_id, text, kind, status, target_date, created_at)  -- kind: northstar|milestone

sessions(id PK=claude_session_id, project_id, cwd, branch,
         current_task, status, started_at, last_heartbeat)   -- status: active|idle|ended

subagents(id, session_id, description, status, started_at, ended_at) -- running|done|failed

events(id, project_id, session_id, type, payload_json, ts)   -- append-only activity log

commands(id, target_session_id, command, status, created_at, consumed_at) -- RESERVED for control v2; unused in v1
```

- **Liveness** = `now - last_heartbeat`. SessionEnd sets `status='ended'`; a
  heartbeat older than ~90s with no end event renders as *stale/dead*. No daemon
  needed to reap.
- **`events`** is the history that makes "what happened on this project this week"
  answerable later; append-only, so it never contends with mutable rows.
- **`commands`** ships empty in v1 — the seam for control later: a session's hook
  polls for rows targeting its id. Reserving it now makes v2 additive, not a
  migration.

## Event flows — who writes what

| Trigger | Mechanism | `cc-control` call | Writes |
|---|---|---|---|
| Session opens on a project | `SessionStart` hook | `session-start --session $ID --cwd $PWD` | upsert `projects` (by repo root), insert `sessions`, `events(session_start)` |
| Periodic liveness + current task | `UserPromptSubmit`/`Stop` hook | `heartbeat --session $ID --task "<summary>"` | update `sessions.last_heartbeat`, `current_task` |
| Background subagent finishes | `SubagentStop` hook | `subagent --session $ID --status done` | upsert `subagents`, `events(subagent)` |
| Session closes | `SessionEnd` hook | `session-end --session $ID` | `sessions.status='ended'`, `events(session_end)` |
| Curate roadmap/goals | **skill** → Claude calls CLI | `roadmap add/done`, `goal set` | mutate `roadmap_items`/`goals` + `events` |
| Git branch/dirty/ahead-behind | TUI, live | — (no write) | read-only shell-out |

Liveness comes for free from hooks that already fire each turn — no polling
daemon, no manual updates.

## Error handling — the reporter must never break the reported

- Every `cc-control` write is wrapped so a locked/missing/corrupt DB → log to
  `~/.claude/control-center/cc-control.log` and **exit 0**. A hook can never fail
  a Claude turn.
- DB + schema **auto-created** on first call (idempotent migration); no setup
  step.
- WAL mode + short transactions + a busy-timeout for concurrent writers.
- TUI tolerates an empty/locked DB (renders "no data"); stale sessions reaped
  purely by heartbeat age, so a crashed session that never fired SessionEnd still
  ages out.

## Testing

- **CLI (the heart):** each subcommand against a temp DB, assert resulting rows.
- **Schema/migration:** fresh-create + idempotent re-run leave identical schema.
- **Liveness:** pure function of `(now, last_heartbeat)` → unit-test directly.
- **Git reader:** against a throwaway temp repo.
- **TUI:** test the query/data layer headless; a light Textual `pilot` smoke test
  that it renders without crashing. No pixel snapshots.
- **Never-break rule:** a write against a locked/garbage DB still exits 0.

## Phasing — sequential, tangible slices

> Build in small, individually-verifiable steps. Each slice produces something
> real before the next begins — no laying the whole thing out at once.

- **Phase 1 — backbone (no UI):** schema + `cc-control` CLI + hooks + skill. End
  state: real state flowing into the DB, inspectable via `sqlite3`. Independently
  useful and fully testable before any TUI exists.
- **Phase 2 — the TUI:** Textual reader + git module on top of a DB already full
  of real data.
- **Phase 3 (later, not now):** the `commands` seam → bidirectional control.

Splitting at the DB boundary means Phase 2 builds against real data, and any bug
is obviously *either* a writer or a reader problem — never both.

## Open / deferred

- Spec currently lives in the storePose repo (where work happened); the dashboard
  is its own project and will likely get its own repo when Phase 1 starts.
- Scheduled routines/crons as a tracked entity — deferred, not designed.
- Bidirectional control (the `commands` table) — Phase 3, schema-reserved only.
