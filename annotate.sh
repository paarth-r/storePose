#!/usr/bin/env bash
# annotate.sh — browse videos/ and launch the CVAT annotation pipeline.
#
# Usage: ./annotate.sh [videos-dir]
# Default directory: videos/
#
# Arrow keys navigate, Enter launches cvat-annotate.sh, q/Esc quits.
set -uo pipefail
cd "$(dirname "$0")"

VIDEOS_DIR="${1:-videos}"

if [[ ! -d "$VIDEOS_DIR" ]]; then
  printf "error: directory not found: %s\n" "$VIDEOS_DIR" >&2
  exit 1
fi

# collect .mp4 files (sorted, preserves spaces in filenames)
FILES=()
while IFS= read -r f; do
  FILES+=("$f")
done < <(find "$VIDEOS_DIR" -name "*.mp4" | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
  printf "No .mp4 files found in %s/\n" "$VIDEOS_DIR" >&2
  exit 1
fi

N=${#FILES[@]}
SEL=0
OFFSET=0
MAX_VIS=16  # max rows visible at once

# ---- terminal helpers ----
_t() { tput "$@" 2>/dev/null || true; }

_restore() {
  _t cnorm   # show cursor
  _t rmcup   # restore original screen
}
trap '_restore' EXIT INT TERM

_t smcup    # switch to alternate screen
_t civis    # hide cursor

_draw() {
  _t clear
  # header
  printf "\033[1mCVAT Annotation Launcher\033[0m\n"
  printf "%s/  (%d files)   " "$VIDEOS_DIR" "$N"
  printf "\033[2m arrows  Enter=annotate  q=quit\033[0m\n\n"

  local limit=$(( OFFSET + MAX_VIS < N ? OFFSET + MAX_VIS : N ))
  local i
  for (( i=OFFSET; i<limit; i++ )); do
    if [[ $i -eq $SEL ]]; then
      printf "  \033[7m %-70s \033[0m\n" "${FILES[$i]}"
    else
      printf "    %s\n" "${FILES[$i]}"
    fi
  done

  # scroll indicators
  if (( OFFSET > 0 )); then
    _t cup 3 0
    printf "\033[2m  ^ %d more above\033[0m" "$OFFSET"
  fi
  local below=$(( N - OFFSET - MAX_VIS ))
  if (( below > 0 )); then
    _t cup $(( MAX_VIS + 4 )) 0
    printf "\033[2m  v %d more below\033[0m" "$below"
  fi

  # footer: show index
  local rows
  rows=$(_t lines 2>/dev/null) || rows=24
  _t cup $(( rows - 1 )) 0
  printf "\033[2m  [%d / %d]\033[0m" "$(( SEL + 1 ))" "$N"
}

# ---- main loop ----
while true; do
  _draw
  IFS= read -rs -n1 KEY
  case "$KEY" in
    q|Q|$'\x1b')
      # distinguish Esc alone (quit) from Esc+[ (arrow key)
      IFS= read -rs -n2 -t 0.05 ESC || ESC=""
      case "$ESC" in
        '[A')  # up arrow
          (( SEL > 0 )) && (( SEL-- )) || true
          (( SEL < OFFSET )) && OFFSET=$SEL || true
          ;;
        '[B')  # down arrow
          (( SEL < N - 1 )) && (( SEL++ )) || true
          (( SEL >= OFFSET + MAX_VIS )) && (( OFFSET++ )) || true
          ;;
        '')    # bare Esc or q
          exit 0 ;;
      esac
      ;;
    $'\n'|$'\r')
      break ;;
  esac
done

# clean up before handing off to cvat-annotate.sh
trap - EXIT INT TERM
_restore

VIDEO="${FILES[$SEL]}"
printf "Launching CVAT pipeline for:\n  %s\n\n" "$VIDEO"
exec ./viewscripts/cvat-annotate.sh --video "$VIDEO"
