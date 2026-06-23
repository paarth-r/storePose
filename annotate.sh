#!/usr/bin/env bash
# annotate.sh — browse videos/ and launch the CVAT annotation pipeline.
#
# Usage: ./annotate.sh [videos-dir]
# Default directory: videos/
#
# Arrow keys navigate, Enter launches cvat-annotate.sh, q/Esc quits.
set -uo pipefail
cd "$(dirname "$0")"

[[ -t 0 ]] || { printf "error: stdin is not a terminal\n" >&2; exit 1; }

VIDEOS_DIR="${1:-videos}"

if [[ ! -d "$VIDEOS_DIR" ]]; then
  printf "error: directory not found: %s\n" "$VIDEOS_DIR" >&2
  exit 1
fi

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

# ---- terminal setup ----
# stty cbreak -echo: read one char at a time, suppress echo.
# Without this, bash's read -s doesn't fully own the terminal and
# unconsumed escape sequence bytes leak into the shell prompt on exit.
OLD_STTY=$(stty -g)
_t() { tput "$@" 2>/dev/null || true; }

_restore() {
  stty "$OLD_STTY" 2>/dev/null || true
  _t cnorm
  _t rmcup
}
trap '_restore' EXIT INT TERM

stty cbreak -echo
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
  IFS= read -rn1 KEY || KEY=""
  case "$KEY" in
    q|Q) exit 0 ;;
    ""|$'\n'|$'\r') break ;;
    $'\x1b')
      # read the bracket, then the command letter, with short timeouts
      IFS= read -rn1 -t 1 C1 || C1=""
      if [[ "$C1" == "[" ]]; then
        IFS= read -rn1 -t 1 C2 || C2=""
        case "$C2" in
          A)  # up arrow
            (( SEL > 0 )) && (( SEL-- )) || true
            (( SEL < OFFSET )) && OFFSET=$SEL || true
            ;;
          B)  # down arrow
            (( SEL < N - 1 )) && (( SEL++ )) || true
            (( SEL >= OFFSET + MAX_VIS )) && (( OFFSET++ )) || true
            ;;
        esac
      elif [[ -z "$C1" ]]; then
        exit 0  # bare Esc
      fi
      # right/left/F-keys: sequence consumed, nothing happens
      ;;
  esac
done

trap - EXIT INT TERM
_restore

VIDEO="${FILES[$SEL]}"
printf "Launching CVAT pipeline for:\n  %s\n\n" "$VIDEO"
exec ./cvat-annotate.sh --video "$VIDEO"
