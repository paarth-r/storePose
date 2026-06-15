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

exec uv run python -m storepose.launcher "$@"
