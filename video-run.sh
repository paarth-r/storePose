#!/usr/bin/env bash
# video-run.sh — pick a saved camera view script and run it.
# Lists every viewscripts/*.sh, lets you select one (fzf if installed, else a
# numbered menu), and runs it. Any extra flags pass through to the chosen run.
#
# Usage:
#   ./video-run.sh                 # interactive selector
#   ./video-run.sh --no-dashboard  # selector, flags forwarded to the run
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0;; esac

DIR="viewscripts"
shopt -s nullglob
scripts=("$DIR"/*.sh)
if [[ ${#scripts[@]} -eq 0 ]]; then
  echo "No view scripts in $DIR/. Create one first:  ./view-setup.sh -v <video>" >&2
  exit 1
fi

names=("${scripts[@]##*/}")              # strip the directory, keep <stem>.sh
names=("${names[@]%.sh}")                # strip the .sh for display

if command -v fzf >/dev/null 2>&1; then
  choice="$(printf '%s\n' "${names[@]}" | fzf --prompt='view > ' --height=40% --reverse)" || exit 0
else
  echo "Select a view to run:"
  PS3="> "
  select choice in "${names[@]}"; do
    [[ -n "${choice:-}" ]] && break
    echo "invalid choice — enter a number from the list"
  done
fi

[[ -n "${choice:-}" ]] || { echo "nothing selected" >&2; exit 1; }
echo "==> running: $choice"
exec "$DIR/${choice}.sh" "$@"
