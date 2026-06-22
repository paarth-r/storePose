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
MAX_VIS=16

# ---- terminal helpers ----
_t() { tput "$@" 2>/dev/null || true; }

_restore() {
  _t cnorm
  _t rmcup
}
trap '_restore' EXIT INT TERM

_t smcup
_t civis

_draw() {
  _t clear
  printf "\033[1mCVAT Annotation Launcher\033[0m — %s/  (%d files)\n" "$VIDEOS_DIR" "$N"
  printf "\033[2m  arrows to move   Enter to annotate   q to quit\033[0m\n\n"

  local limit=$(( OFFSET + MAX_VIS < N ? OFFSET + MAX_VIS : N ))
  local i
  for (( i=OFFSET; i<limit; i++ )); do
    if [[ $i -eq $SEL ]]; then
      printf "  \033[7m %-70s \033[0m\n" "${FILES[$i]}"
    else
      printf "    %s\n" "${FILES[$i]}"
    fi
  done

  printf "\n"
  (( OFFSET > 0 )) && printf "  \033[2m^ %d more above\033[0m\n" "$OFFSET" || true
  local below=$(( N - OFFSET - MAX_VIS ))
  (( below > 0 )) && printf "  \033[2mv %d more below\033[0m\n" "$below" || true
  printf "  \033[2m[%d / %d]\033[0m\n" "$(( SEL + 1 ))" "$N"
}

# ---- main loop ----
while true; do
  _draw
  IFS= read -rs -n1 KEY
  case "$KEY" in
    q|Q) exit 0 ;;
    $'\n'|$'\r') break ;;
    $'\x1b')
      # read the two follow-on bytes one at a time so a partial read can't
      # reset the whole sequence (read -n2 || ESC="" would lose the first byte)
      C1="" C2=""
      IFS= read -rs -n1 -t 0.15 C1 || true
      IFS= read -rs -n1 -t 0.05 C2 || true
      case "${C1}${C2}" in
        '[A')  # up
          (( SEL > 0 )) && (( SEL-- )) || true
          (( SEL < OFFSET )) && OFFSET=$SEL || true
          ;;
        '[B')  # down
          (( SEL < N - 1 )) && (( SEL++ )) || true
          (( SEL >= OFFSET + MAX_VIS )) && (( OFFSET++ )) || true
          ;;
        '') exit 0 ;;  # bare Esc
      esac
      ;;
  esac
done

trap - EXIT INT TERM
_restore

VIDEO="${FILES[$SEL]}"
printf "Launching CVAT pipeline for:\n  %s\n\n" "$VIDEO"
exec ./viewscripts/cvat-annotate.sh --video "$VIDEO"
