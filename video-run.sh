#!/usr/bin/env bash
# video-run.sh — arrow-driven launcher for saved camera views.
# Lists every viewscripts/*.sh in an interactive table: scroll with ↑/↓, move
# across flag columns with ←/→, toggle with space, Enter to run. Any extra args
# pass through to the launched run. Falls back to a numbered prompt when not a TTY.
#
# Usage:
#   ./video-run.sh                 # interactive launcher
#   ./video-run.sh --no-dashboard  # launcher, flags forwarded to the run
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0;; esac

# Build the polished web dashboard (web/out) when it is missing or stale so the
# launcher serves the current UI. Falls back silently to the legacy page if npm
# is unavailable or the build fails (main.py serves PAGE_HTML when web/out is absent).
build_web() {
  [ -d web ] || return 0
  command -v npm >/dev/null 2>&1 || { echo "note: npm not found; using the legacy dashboard page." >&2; return 0; }
  local idx="web/out/index.html"
  local sources=(web/app web/components web/lib web/package.json web/next.config.mjs)
  if [ -f "$idx" ] && [ -z "$(find "${sources[@]}" -newer "$idx" 2>/dev/null)" ]; then
    return 0  # up to date
  fi
  echo "Building web dashboard (one-time / after changes)…" >&2
  if ( cd web && { [ -d node_modules ] || npm install; } && npm run build ) >/tmp/storepose-web-build.log 2>&1; then
    echo "Web dashboard ready." >&2
  else
    echo "warning: web build failed (see /tmp/storepose-web-build.log); serving the legacy page." >&2
  fi
}
build_web

exec uv run python -m storepose.launcher "$@"
