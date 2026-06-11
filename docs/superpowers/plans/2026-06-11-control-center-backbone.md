# Control Center — Phase 1 (Backbone) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the reporting backbone for the multi-project control center — a SQLite store plus a `cc-control` CLI that hooks and a skill call to record live Claude session state and curated roadmap/goals. No UI yet; success = real state flowing into a DB you can inspect with `sqlite3`.

**Architecture:** One SQLite DB (WAL mode) behind a single thin CLI that is the only writer. Hook subcommands read Claude Code's hook JSON from stdin; curation subcommands take flags and are called by a skill. Every write is wrapped so a locked/missing/corrupt DB logs and exits 0 — the reporter must never break the reported.

**Tech Stack:** Python 3.12, uv, stdlib `argparse` + `sqlite3` (zero runtime deps), pytest. New repo at `~/Code/control-center`.

Spec: `storePose/docs/superpowers/specs/2026-06-11-control-center-dashboard-design.md`.

---

## File structure (new repo `~/Code/control-center`)

```
pyproject.toml                  # uv project; console_script cc-control; pytest config
src/control_center/
  __init__.py
  db.py                         # DB path resolution + connect() (WAL, busy_timeout, auto-init)
  schema.py                     # SCHEMA_SQL + init_db(conn) (idempotent)
  writes.py                     # all mutation functions (the only place that writes rows)
  cli.py                        # argparse: hook-* (stdin) + curation subcommands; never-break wrapper; git helpers
tests/
  test_schema.py
  test_db.py
  test_writes.py
  test_cli.py
skill/control-center/SKILL.md   # copied to ~/.claude/skills/control-center/ in Task 7
```

`writes.py` holds one function per mutation; `cli.py` only parses input and calls them. This keeps SQL in one auditable place and lets every write be unit-tested without a CLI.

---

### Task 1: Scaffold the repo and CLI entry point

**Files:**
- Create: `~/Code/control-center/pyproject.toml`
- Create: `~/Code/control-center/src/control_center/__init__.py`
- Create: `~/Code/control-center/src/control_center/cli.py`
- Test: `~/Code/control-center/tests/test_cli.py`

- [ ] **Step 1: Create the repo and write `pyproject.toml`**

```bash
mkdir -p ~/Code/control-center/src/control_center ~/Code/control-center/tests && cd ~/Code/control-center && git init
```

`pyproject.toml`:

```toml
[project]
name = "control-center"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[project.scripts]
cc-control = "control_center.cli:main"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing test**

`tests/test_cli.py`:

```python
from control_center.cli import main


def test_main_help_exits_zero(capsys):
    # argparse prints help and raises SystemExit(0)
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
    out = capsys.readouterr().out
    assert "cc-control" in out
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/Code/control-center && uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'control_center'` (or `main` undefined).

- [ ] **Step 4: Write minimal `__init__.py` and `cli.py`**

`src/control_center/__init__.py`: empty file.

`src/control_center/cli.py`:

```python
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cc-control")
    parser.add_subparsers(dest="command")
    return parser


def main(argv=None):
    parser = build_parser()
    parser.parse_args(argv)
    return 0
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS. Also verify the console script: `uv run cc-control --help` prints usage.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: scaffold control-center repo and cc-control entry point"
```

---

### Task 2: Schema module

**Files:**
- Create: `src/control_center/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

`tests/test_schema.py`:

```python
import sqlite3

from control_center.schema import init_db


def _tables(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def test_init_db_creates_all_tables():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    assert _tables(conn) >= {
        "projects", "roadmap_items", "goals",
        "sessions", "subagents", "events", "commands",
    }


def test_init_db_is_idempotent():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    init_db(conn)  # must not raise
    assert "projects" in _tables(conn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schema.py -v`
Expected: FAIL — `No module named 'control_center.schema'`.

- [ ] **Step 3: Write `schema.py`**

```python
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    repo_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS roadmap_items (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    title TEXT NOT NULL,
    body TEXT,
    status TEXT NOT NULL DEFAULT 'todo',
    priority INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    text TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'milestone',
    status TEXT NOT NULL DEFAULT 'open',
    target_date TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    cwd TEXT,
    branch TEXT,
    current_task TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    started_at TEXT NOT NULL,
    last_heartbeat TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subagents (
    id INTEGER PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    description TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    project_id INTEGER,
    session_id TEXT,
    type TEXT NOT NULL,
    payload_json TEXT,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY,
    target_session_id TEXT,
    command TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    consumed_at TEXT
);
"""


def init_db(conn):
    conn.executescript(SCHEMA_SQL)
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_schema.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: control-center sqlite schema with idempotent init"
```

---

### Task 3: DB connection helper

**Files:**
- Create: `src/control_center/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

`tests/test_db.py`:

```python
from control_center.db import connect, db_path


def test_db_path_honors_env(monkeypatch, tmp_path):
    target = tmp_path / "custom.db"
    monkeypatch.setenv("CC_CONTROL_DB", str(target))
    assert db_path() == target


def test_connect_creates_file_and_schema(tmp_path):
    p = tmp_path / "nested" / "control.db"
    conn = connect(p)
    # schema auto-created
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "sessions" in names
    # WAL mode active
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    assert p.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL — `No module named 'control_center.db'`.

- [ ] **Step 3: Write `db.py`**

```python
import os
import sqlite3
from pathlib import Path

from control_center.schema import init_db


def db_path() -> Path:
    override = os.environ.get("CC_CONTROL_DB")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "control-center" / "control.db"


def connect(path=None) -> sqlite3.Connection:
    p = Path(path) if path is not None else db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: control-center db connect helper (WAL, auto-init)"
```

---

### Task 4: Session lifecycle writes

**Files:**
- Create: `src/control_center/writes.py`
- Test: `tests/test_writes.py`

- [ ] **Step 1: Write the failing test**

`tests/test_writes.py`:

```python
from control_center.db import connect
from control_center import writes


def _conn(tmp_path):
    return connect(tmp_path / "t.db")


def test_session_start_creates_project_and_session(tmp_path):
    conn = _conn(tmp_path)
    writes.session_start(conn, "sess-1", "/repo/foo", "foo", "/repo/foo", "main")
    proj = conn.execute("SELECT * FROM projects").fetchone()
    sess = conn.execute("SELECT * FROM sessions WHERE id='sess-1'").fetchone()
    assert proj["repo_path"] == "/repo/foo"
    assert sess["project_id"] == proj["id"]
    assert sess["status"] == "active"
    assert sess["branch"] == "main"
    ev = conn.execute("SELECT * FROM events WHERE type='session_start'").fetchone()
    assert ev["session_id"] == "sess-1"


def test_session_start_is_idempotent_on_project(tmp_path):
    conn = _conn(tmp_path)
    writes.session_start(conn, "sess-1", "/repo/foo", "foo", "/repo/foo", "main")
    writes.session_start(conn, "sess-2", "/repo/foo", "foo", "/repo/foo", "dev")
    assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2


def test_heartbeat_updates_timestamp_and_task(tmp_path):
    conn = _conn(tmp_path)
    writes.session_start(conn, "s", "/r", "r", "/r", "main")
    before = conn.execute("SELECT last_heartbeat FROM sessions WHERE id='s'").fetchone()[0]
    writes.heartbeat(conn, "s", task="doing the thing")
    row = conn.execute("SELECT last_heartbeat, current_task FROM sessions WHERE id='s'").fetchone()
    assert row["current_task"] == "doing the thing"
    assert row["last_heartbeat"] >= before


def test_session_end_marks_ended(tmp_path):
    conn = _conn(tmp_path)
    writes.session_start(conn, "s", "/r", "r", "/r", "main")
    writes.session_end(conn, "s")
    assert conn.execute("SELECT status FROM sessions WHERE id='s'").fetchone()[0] == "ended"
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE type='session_end'").fetchone()[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_writes.py -v`
Expected: FAIL — `No module named 'control_center.writes'`.

- [ ] **Step 3: Write `writes.py` (lifecycle portion)**

```python
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(conn, project_id, session_id, type_, payload):
    conn.execute(
        "INSERT INTO events(project_id, session_id, type, payload_json, ts) "
        "VALUES(?,?,?,?,?)",
        (project_id, session_id, type_, payload, _now()),
    )


def upsert_project(conn, repo_path, name) -> int:
    conn.execute(
        "INSERT INTO projects(name, repo_path, created_at) VALUES(?,?,?) "
        "ON CONFLICT(repo_path) DO NOTHING",
        (name, repo_path, _now()),
    )
    row = conn.execute(
        "SELECT id FROM projects WHERE repo_path=?", (repo_path,)
    ).fetchone()
    return row["id"]


def session_start(conn, session_id, project_ref, name, cwd, branch):
    pid = upsert_project(conn, project_ref, name)
    now = _now()
    conn.execute(
        "INSERT INTO sessions(id, project_id, cwd, branch, status, started_at, last_heartbeat) "
        "VALUES(?,?,?,?, 'active', ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET project_id=excluded.project_id, "
        "cwd=excluded.cwd, branch=excluded.branch, status='active', "
        "last_heartbeat=excluded.last_heartbeat",
        (session_id, pid, cwd, branch, now, now),
    )
    _event(conn, pid, session_id, "session_start", None)
    conn.commit()


def heartbeat(conn, session_id, task=None):
    now = _now()
    if task is None:
        conn.execute(
            "UPDATE sessions SET last_heartbeat=?, status='active' WHERE id=?",
            (now, session_id),
        )
    else:
        conn.execute(
            "UPDATE sessions SET last_heartbeat=?, status='active', current_task=? WHERE id=?",
            (now, task, session_id),
        )
    conn.commit()


def session_end(conn, session_id):
    row = conn.execute(
        "SELECT project_id FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    pid = row["project_id"] if row else None
    conn.execute(
        "UPDATE sessions SET status='ended', last_heartbeat=? WHERE id=?",
        (_now(), session_id),
    )
    _event(conn, pid, session_id, "session_end", None)
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_writes.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: session lifecycle writes (start/heartbeat/end)"
```

---

### Task 5: Subagent + roadmap + goal writes

**Files:**
- Modify: `src/control_center/writes.py` (append functions)
- Modify: `tests/test_writes.py` (append tests)

- [ ] **Step 1: Write the failing tests (append to `tests/test_writes.py`)**

```python
def test_subagent_stop_records_row_and_event(tmp_path):
    conn = _conn(tmp_path)
    writes.session_start(conn, "s", "/r", "r", "/r", "main")
    writes.subagent_stop(conn, "s", description="explore the codebase", status="done")
    row = conn.execute("SELECT * FROM subagents WHERE session_id='s'").fetchone()
    assert row["description"] == "explore the codebase"
    assert row["status"] == "done"
    assert row["ended_at"] is not None
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE type='subagent'").fetchone()[0] == 1


def test_roadmap_add_then_set_status(tmp_path):
    conn = _conn(tmp_path)
    pid = writes.upsert_project(conn, "/r", "r")
    item_id = writes.roadmap_add(conn, pid, "Build the TUI", body="phase 2", priority=5)
    row = conn.execute("SELECT * FROM roadmap_items WHERE id=?", (item_id,)).fetchone()
    assert row["title"] == "Build the TUI"
    assert row["status"] == "todo"
    assert row["priority"] == 5
    writes.roadmap_set_status(conn, item_id, "done")
    assert conn.execute(
        "SELECT status FROM roadmap_items WHERE id=?", (item_id,)).fetchone()[0] == "done"


def test_goal_set(tmp_path):
    conn = _conn(tmp_path)
    pid = writes.upsert_project(conn, "/r", "r")
    gid = writes.goal_set(conn, pid, "Ship view-only v1", kind="northstar")
    row = conn.execute("SELECT * FROM goals WHERE id=?", (gid,)).fetchone()
    assert row["text"] == "Ship view-only v1"
    assert row["kind"] == "northstar"
    assert row["status"] == "open"


def test_resolve_project_by_path_or_name(tmp_path):
    conn = _conn(tmp_path)
    pid = writes.upsert_project(conn, "/repo/foo", "foo")
    assert writes.resolve_project(conn, "/repo/foo") == pid
    assert writes.resolve_project(conn, "foo") == pid
    assert writes.resolve_project(conn, "nope") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_writes.py -v`
Expected: FAIL — `module 'control_center.writes' has no attribute 'subagent_stop'`.

- [ ] **Step 3: Append functions to `writes.py`**

```python
def subagent_stop(conn, session_id, description=None, status="done"):
    row = conn.execute(
        "SELECT project_id FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    pid = row["project_id"] if row else None
    now = _now()
    conn.execute(
        "INSERT INTO subagents(session_id, description, status, started_at, ended_at) "
        "VALUES(?,?,?,?,?)",
        (session_id, description, status, now, now),
    )
    _event(conn, pid, session_id, "subagent", description)
    conn.commit()


def resolve_project(conn, ref):
    row = conn.execute(
        "SELECT id FROM projects WHERE repo_path=? OR name=?", (ref, ref)
    ).fetchone()
    return row["id"] if row else None


def roadmap_add(conn, project_id, title, body=None, priority=0) -> int:
    now = _now()
    cur = conn.execute(
        "INSERT INTO roadmap_items(project_id, title, body, status, priority, created_at, updated_at) "
        "VALUES(?,?,?, 'todo', ?,?,?)",
        (project_id, title, body, priority, now, now),
    )
    _event(conn, project_id, None, "roadmap_add", title)
    conn.commit()
    return cur.lastrowid


def roadmap_set_status(conn, item_id, status):
    conn.execute(
        "UPDATE roadmap_items SET status=?, updated_at=? WHERE id=?",
        (status, _now(), item_id),
    )
    conn.commit()


def goal_set(conn, project_id, text, kind="milestone", target_date=None) -> int:
    cur = conn.execute(
        "INSERT INTO goals(project_id, text, kind, status, target_date, created_at) "
        "VALUES(?,?,?, 'open', ?,?)",
        (project_id, text, kind, target_date, _now()),
    )
    _event(conn, project_id, None, "goal_set", text)
    conn.commit()
    return cur.lastrowid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_writes.py -v`
Expected: PASS (all tests, old and new).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: subagent, roadmap, and goal writes"
```

---

### Task 6: CLI subcommands + never-break wrapper

**Files:**
- Modify: `src/control_center/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests (append to `tests/test_cli.py`)**

```python
import json

from control_center.db import connect
from control_center import writes


def test_hook_session_start_from_stdin(tmp_path, monkeypatch, capsys):
    db = tmp_path / "t.db"
    monkeypatch.setenv("CC_CONTROL_DB", str(db))
    payload = json.dumps({"session_id": "abc", "cwd": str(tmp_path)})
    monkeypatch.setattr("sys.stdin.read", lambda: payload)
    rc = main(["hook-session-start"])
    assert rc == 0
    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) FROM sessions WHERE id='abc'").fetchone()[0] == 1


def test_hook_heartbeat_sets_current_task_from_prompt(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("CC_CONTROL_DB", str(db))
    conn = connect(db)
    writes.session_start(conn, "abc", str(tmp_path), "x", str(tmp_path), "main")
    payload = json.dumps({"session_id": "abc", "prompt": "fix the bug in foo"})
    monkeypatch.setattr("sys.stdin.read", lambda: payload)
    assert main(["hook-heartbeat"]) == 0
    conn2 = connect(db)
    assert conn2.execute(
        "SELECT current_task FROM sessions WHERE id='abc'").fetchone()[0] == "fix the bug in foo"


def test_roadmap_add_command(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("CC_CONTROL_DB", str(db))
    conn = connect(db)
    writes.upsert_project(conn, "/repo/foo", "foo")
    conn.commit()
    rc = main(["roadmap", "add", "--project", "foo", "--title", "Do the thing"])
    assert rc == 0
    conn2 = connect(db)
    assert conn2.execute(
        "SELECT COUNT(*) FROM roadmap_items WHERE title='Do the thing'").fetchone()[0] == 1


def test_write_never_raises_on_bad_db(tmp_path, monkeypatch):
    # Point at a path that cannot be a DB (a directory) -> must still exit 0.
    bad = tmp_path / "iam_a_dir"
    bad.mkdir()
    monkeypatch.setenv("CC_CONTROL_DB", str(bad))
    monkeypatch.setattr("sys.stdin.read", lambda: '{"session_id": "z", "cwd": "/tmp"}')
    assert main(["hook-session-start"]) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — unknown subcommand `hook-session-start` / `main` ignores it.

- [ ] **Step 3: Rewrite `cli.py`**

```python
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from control_center.db import connect
from control_center import writes


def _log(msg):
    try:
        p = Path.home() / ".claude" / "control-center" / "cc-control.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except Exception:
        pass


def _read_hook_payload():
    try:
        raw = sys.stdin.read()
    except Exception:
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _git_info(cwd):
    def run(args):
        try:
            return subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=3
            ).stdout.strip()
        except Exception:
            return ""

    root = run(["rev-parse", "--show-toplevel"]) or cwd
    branch = run(["rev-parse", "--abbrev-ref", "HEAD"]) or ""
    return root, branch


# ---- command handlers (each returns an int rc) ----

def _cmd_hook_session_start(_args):
    p = _read_hook_payload()
    sid = p.get("session_id")
    cwd = p.get("cwd") or os.getcwd()
    if not sid:
        return 0
    root, branch = _git_info(cwd)
    name = os.path.basename(root.rstrip("/")) or root
    conn = connect()
    writes.session_start(conn, sid, root, name, cwd, branch)
    return 0


def _cmd_hook_heartbeat(_args):
    p = _read_hook_payload()
    sid = p.get("session_id")
    if not sid:
        return 0
    task = p.get("prompt")
    if isinstance(task, str):
        task = task.strip()[:200] or None
    conn = connect()
    writes.heartbeat(conn, sid, task=task)
    return 0


def _cmd_hook_subagent_stop(_args):
    p = _read_hook_payload()
    sid = p.get("session_id")
    if not sid:
        return 0
    conn = connect()
    writes.subagent_stop(conn, sid, description=p.get("description"), status="done")
    return 0


def _cmd_hook_session_end(_args):
    p = _read_hook_payload()
    sid = p.get("session_id")
    if not sid:
        return 0
    conn = connect()
    writes.session_end(conn, sid)
    return 0


def _cmd_roadmap_add(args):
    conn = connect()
    pid = writes.resolve_project(conn, args.project)
    if pid is None:
        print(f"unknown project: {args.project}", file=sys.stderr)
        return 1
    writes.roadmap_add(conn, pid, args.title, body=args.body, priority=args.priority)
    return 0


def _cmd_roadmap_done(args):
    conn = connect()
    writes.roadmap_set_status(conn, args.id, "done")
    return 0


def _cmd_goal_set(args):
    conn = connect()
    pid = writes.resolve_project(conn, args.project)
    if pid is None:
        print(f"unknown project: {args.project}", file=sys.stderr)
        return 1
    writes.goal_set(conn, pid, args.text, kind=args.kind, target_date=args.target_date)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cc-control")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("hook-session-start").set_defaults(func=_cmd_hook_session_start)
    sub.add_parser("hook-heartbeat").set_defaults(func=_cmd_hook_heartbeat)
    sub.add_parser("hook-subagent-stop").set_defaults(func=_cmd_hook_subagent_stop)
    sub.add_parser("hook-session-end").set_defaults(func=_cmd_hook_session_end)

    rm = sub.add_parser("roadmap").add_subparsers(dest="roadmap_cmd")
    add = rm.add_parser("add")
    add.add_argument("--project", required=True)
    add.add_argument("--title", required=True)
    add.add_argument("--body", default=None)
    add.add_argument("--priority", type=int, default=0)
    add.set_defaults(func=_cmd_roadmap_add)
    done = rm.add_parser("done")
    done.add_argument("--id", type=int, required=True)
    done.set_defaults(func=_cmd_roadmap_done)

    goal = sub.add_parser("goal").add_subparsers(dest="goal_cmd")
    gset = goal.add_parser("set")
    gset.add_argument("--project", required=True)
    gset.add_argument("--text", required=True)
    gset.add_argument("--kind", default="milestone")
    gset.add_argument("--target-date", dest="target_date", default=None)
    gset.set_defaults(func=_cmd_goal_set)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    is_hook = isinstance(args.command, str) and args.command.startswith("hook-")
    try:
        return func(args)
    except Exception as e:  # the reporter must never break the reported
        _log(f"{getattr(args, 'command', '?')}: {e!r}")
        return 0 if is_hook else 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: PASS (all tests across all files).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: cc-control CLI (hook + curation subcommands, never-break)"
```

---

### Task 7: Install the tool, wire hooks, ship the skill

This task is configuration + manual verification (no unit tests — it touches global `~/.claude` config and a real session).

**Files:**
- Create: `skill/control-center/SKILL.md` (then copied to `~/.claude/skills/control-center/SKILL.md`)
- Modify: `~/.claude/settings.json` (hooks block)

- [ ] **Step 1: Install `cc-control` on PATH**

```bash
cd ~/Code/control-center && uv tool install --editable .
```

Verify: `cc-control --help` runs from any directory (e.g. `cd ~ && cc-control --help`).

- [ ] **Step 2: Smoke-test the CLI end to end against the real DB**

```bash
echo '{"session_id":"manual-test","cwd":"'"$HOME/Code/storePose"'"}' | cc-control hook-session-start
sqlite3 ~/.claude/control-center/control.db "SELECT id, project_id, branch, status FROM sessions;"
```

Expected: a `manual-test` row with the storePose project id and current branch. Then clean it:

```bash
sqlite3 ~/.claude/control-center/control.db "DELETE FROM sessions WHERE id='manual-test';"
```

- [ ] **Step 3: Write the skill**

`skill/control-center/SKILL.md`:

```markdown
---
name: control-center
description: Use when the user asks to record or update a project's roadmap, goals, or milestones in the control center, e.g. "add a roadmap item", "mark that milestone done", "set a goal for this project". Reports curated project state into the control-center dashboard DB via the cc-control CLI.
---

# Control Center — curating roadmap & goals

The control center mirrors live Claude work automatically via hooks. This skill
is for the *deliberate* updates only: roadmap items, goals, and milestone status.

All writes go through the `cc-control` CLI (already on PATH). The current
project is identified by its git repo root path or its name (the repo's folder
name). Resolve the name from the current working directory's repo root.

## Add a roadmap item
`cc-control roadmap add --project <name-or-repo-path> --title "<title>" [--body "<detail>"] [--priority <int>]`

## Mark a roadmap item done
First find its id, then:
`cc-control roadmap done --id <id>`
(List ids with: `sqlite3 ~/.claude/control-center/control.db "SELECT id,title,status FROM roadmap_items;"`)

## Set a goal
`cc-control goal set --project <name-or-repo-path> --text "<goal>" [--kind northstar|milestone] [--target-date YYYY-MM-DD]`

The CLI is safe to call freely: a failed write logs to
`~/.claude/control-center/cc-control.log` and never errors the session.
```

Then install it:

```bash
mkdir -p ~/.claude/skills/control-center && cp skill/control-center/SKILL.md ~/.claude/skills/control-center/SKILL.md
```

- [ ] **Step 4: Wire the hooks into `~/.claude/settings.json`**

Add these entries to the `hooks` object (merge with any existing hooks — do not clobber). Each command reads the hook JSON from stdin:

```json
{
  "hooks": {
    "SessionStart": [
      { "hooks": [ { "type": "command", "command": "cc-control hook-session-start" } ] }
    ],
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "cc-control hook-heartbeat" } ] }
    ],
    "SubagentStop": [
      { "hooks": [ { "type": "command", "command": "cc-control hook-subagent-stop" } ] }
    ],
    "SessionEnd": [
      { "hooks": [ { "type": "command", "command": "cc-control hook-session-end" } ] }
    ]
  }
}
```

- [ ] **Step 5: Verify end to end in a real session**

Open a fresh Claude Code session in `~/Code/storePose`, send one prompt, then in another terminal:

```bash
sqlite3 ~/.claude/control-center/control.db "SELECT id, project_id, current_task, status, last_heartbeat FROM sessions ORDER BY last_heartbeat DESC LIMIT 5;"
```

Expected: a row for the live session with `current_task` = the prompt text, `status='active'`, and a fresh `last_heartbeat`. This is the Phase 1 success criterion — live state in the DB, inspectable.

- [ ] **Step 6: Commit the repo (skill source lives in the repo)**

```bash
cd ~/Code/control-center && git add -A && git commit -m "feat: control-center skill + hook wiring docs"
```

---

## Self-review notes

- **Spec coverage:** schema (all 7 tables incl. reserved `commands`) ✓; single-writer CLI ✓; hooks for session-start/heartbeat/subagent/session-end ✓; skill for roadmap/goal curation ✓; never-break (exit 0 on bad DB) ✓ (Task 6 test); WAL + busy_timeout ✓; git state read live — the CLI captures branch at session-start; full dirty/ahead-behind is a **TUI (Phase 2)** concern, not backbone. Liveness via `last_heartbeat` ✓ (stale detection itself is a Phase 2 read concern).
- **Out of scope (correctly):** TUI, stale-session reaping/rendering, the `commands` polling loop — all later phases.
- **Type consistency:** `session_start(conn, session_id, project_ref, name, cwd, branch)`, `heartbeat(conn, session_id, task=None)`, `subagent_stop(conn, session_id, description, status)`, `roadmap_add(conn, project_id, title, body, priority)`, `roadmap_set_status(conn, item_id, status)`, `goal_set(conn, project_id, text, kind, target_date)`, `resolve_project(conn, ref)` — names used consistently across Tasks 4–6 and the CLI.
```
