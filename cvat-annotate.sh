#!/usr/bin/env bash
# cvat-annotate.sh — full CVAT annotation pipeline for a video clip.
#
# Usage (direct):
#   ./cvat-annotate.sh --video videos/clip.mp4 [--fps 30] [--out clip-cvat.xml]
#
# Usage (via the picker):
#   ./annotate.sh   # arrow to a clip, Enter; execs this script with --video
#   set CVAT_VIDEO_PATH before launching to skip the interactive prompt.
#
# Flags:
#   --video PATH    video file to annotate (or set CVAT_VIDEO_PATH env)
#   --fps N         frame rate (default: auto-detect via ffprobe/OpenCV, else 30)
#   --out PATH      output XML path (default: <video-stem>-cvat.xml)
#   --no-cache      regenerate the pre-annotation even if a cached one exists
#   --help          print this header
#
# Environment:
#   CVAT_DIR        path to cloned cvat repo  (default: ~/cvat)
#   CVAT_URL        CVAT base URL             (default: http://localhost:8080)
#   CVAT_USER       superuser username        (default: admin)
#   CVAT_PASS       superuser password        (default: admin)
#   CVAT_VIDEO_PATH video path (alternative to --video flag)
#   CVAT_PREANNO_MAXFRAMES  limit pre-annotation to the first N frames
#   CVAT_NO_PREANNO         skip pre-annotation; review starts from a blank task
#   CVAT_NO_CACHE           same as --no-cache
#
# Pre-annotation upload/import/export use the official cvat-cli (run via uvx),
# which speaks CVAT 2.x's upload-framework API; raw curl endpoints were removed.
# Unknown flags are silently ignored (tolerant of extra args from the picker).
#
# Steps:
#   1. Start CVAT docker containers if not running
#   2. Authenticate and obtain an API token
#   3. Create a CVAT task with the storePose label schema (person/rectangle/track,
#      membership select: in_line | bystander)
#   4. Upload the video and wait for CVAT to process it
#   5. Pre-annotate with the storePose pipeline and upload the boxes for review
#   6. Open the annotation UI in the browser; wait for you to finish, export XML
#
# After this script finishes, run:
#   uv run python busy_report.py import-cvat <out.xml> --fps <N> --step 1 -o gt.csv
set -uo pipefail
cd "$(dirname "$0")"

# ---- configurable defaults ----
CVAT_DIR="${CVAT_DIR:-$HOME/cvat}"
CVAT_URL="${CVAT_URL:-http://localhost:8080}"
CVAT_USER="${CVAT_USER:-admin}"
CVAT_PASS="${CVAT_PASS:-admin}"

VIDEO_PATH="${CVAT_VIDEO_PATH:-}"
FPS_EXPLICIT=""
OUT_PATH=""
NO_CACHE="${CVAT_NO_CACHE:-}"

# ---- flag parsing (unknown flags ignored for TUI compatibility) ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --video|-v)  VIDEO_PATH="$2"; shift 2 ;;
    --fps)       FPS_EXPLICIT="$2"; shift 2 ;;
    --out|-o)    OUT_PATH="$2"; shift 2 ;;
    --no-cache)  NO_CACHE=1; shift ;;
    --help|-h)   awk 'NR>1{if(/^[^#]/)exit; sub(/^# ?/,""); print}' "$0"; exit 0 ;;
    *)           shift ;;
  esac
done

# ---- cvat-cli wrapper (version-correct upload/import/export for CVAT 2.x) ----
# Split CVAT_URL (e.g. http://localhost:8080) into host + port for cvat-cli.
_CVAT_PORT="${CVAT_URL##*:}"; [[ "$_CVAT_PORT" =~ ^[0-9]+$ ]] || _CVAT_PORT=8080
_CVAT_HOST="${CVAT_URL%:*}";  [[ "$_CVAT_HOST" =~ ^https?:// ]] || _CVAT_HOST="http://localhost"
_cli() {
  uvx cvat-cli --auth "${CVAT_USER}:${CVAT_PASS}" \
    --server-host "$_CVAT_HOST" --server-port "$_CVAT_PORT" "$@"
}

# ---- video path: flag > env > interactive prompt ----
if [[ -z "$VIDEO_PATH" ]]; then
  printf "Video path (or set CVAT_VIDEO_PATH): "
  read -r VIDEO_PATH
fi
[[ -f "$VIDEO_PATH" ]] || { printf "error: not found: %s\n" "$VIDEO_PATH" >&2; exit 1; }

# ---- derive output path and task name ----
STEM="${VIDEO_PATH%.*}"
[[ -z "$OUT_PATH" ]] && OUT_PATH="${STEM}-cvat.xml"
TASK_NAME="$(basename "$STEM")-$(date +%Y%m%d-%H%M%S)"

# ---- auto-detect FPS (ffprobe; else OpenCV; else explicit flag / 30) ----
FPS="${FPS_EXPLICIT:-30}"
if [[ -z "$FPS_EXPLICIT" ]]; then
  _D=""
  if command -v ffprobe >/dev/null 2>&1; then
    _D=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate \
         -of csv=p=0 "$VIDEO_PATH" 2>/dev/null | \
         python3 -c "import sys; f=sys.stdin.read().strip(); p=f.split('/'); \
                     print(round(float(p[0])/float(p[1]),4) if '/' in f else float(f))" \
         2>/dev/null || true)
  fi
  # No ffprobe on this machine (common on macOS) -> read fps via OpenCV.
  if [[ -z "$_D" ]]; then
    _D=$(uv run python -c "import cv2,sys; c=cv2.VideoCapture(sys.argv[1]); \
                           print(round(c.get(cv2.CAP_PROP_FPS),4) or '')" \
         "$VIDEO_PATH" 2>/dev/null || true)
  fi
  [[ -n "$_D" && "$_D" != "0.0" ]] && FPS="$_D"
fi

EXPORT_TMP=$(mktemp /tmp/cvat-export-XXXXXX)
trap 'rm -f "$EXPORT_TMP"' EXIT

printf "\n=== CVAT annotation pipeline ===\n"
printf "Video  : %s\n" "$VIDEO_PATH"
printf "FPS    : %s\n" "$FPS"
printf "Output : %s\n" "$OUT_PATH"
printf "Task   : %s\n\n" "$TASK_NAME"

# ================================================================
# 1. Ensure CVAT containers are running
# ================================================================
printf "[1/6] Checking CVAT containers...\n"
if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "cvat_server"; then
  if [[ ! -d "$CVAT_DIR" ]]; then
    printf "\nerror: CVAT clone not found at %s\n" "$CVAT_DIR" >&2
    printf "Set it up once:\n" >&2
    printf "  git clone https://github.com/cvat-ai/cvat %s\n" "$CVAT_DIR" >&2
    printf "  cd %s && docker compose up -d\n" "$CVAT_DIR" >&2
    printf "  docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'\n" >&2
    printf "Then re-run this script.\n" >&2
    exit 1
  fi
  printf "      Starting containers (first time may take ~2 min)...\n"
  (cd "$CVAT_DIR" && docker compose up -d 2>&1) | grep -E "Started|Running|done|Warning|Error" || true
fi

printf "      Waiting for server"
for i in $(seq 1 90); do
  if curl -sf "${CVAT_URL}/api/server/about" >/dev/null 2>&1; then
    printf " ready.\n"; break
  fi
  printf "."
  sleep 2
  [[ $i -eq 90 ]] && { printf "\nerror: server not ready after 180s\n" >&2; exit 1; }
done

# ================================================================
# 2. Authenticate and get API token
# ================================================================
printf "[2/6] Authenticating...\n"
_auth_resp=$(curl -s -X POST "${CVAT_URL}/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"${CVAT_USER}\",\"password\":\"${CVAT_PASS}\"}")
TOKEN=$(printf '%s' "$_auth_resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['key'])" 2>/dev/null) || true
[[ -n "$TOKEN" ]] || {
  printf "error: authentication failed for user '%s' (check CVAT_USER / CVAT_PASS)\n" "$CVAT_USER" >&2
  printf "       server said: %s\n" "$_auth_resp" >&2
  exit 1
}

_api() {
  curl -sf "${CVAT_URL}/api/$1" -H "Authorization: Token ${TOKEN}" "${@:2}"
}

# ================================================================
# 3. Create task with person/points/track label schema
# ================================================================
printf "[3/6] Creating task: %s\n" "$TASK_NAME"
TASK_PAYLOAD=$(python3 -c '
import json, sys
print(json.dumps({
    "name": sys.argv[1],
    "labels": [{
        "name": "person",
        "type": "rectangle",
        "attributes": [
            {"name": "membership", "input_type": "select", "mutable": True,
             "values": ["in_line", "bystander"], "default_value": "in_line"},
            {"name": "state", "input_type": "text", "mutable": True,
             "values": [], "default_value": ""},
            {"name": "intent", "input_type": "text", "mutable": True,
             "values": [], "default_value": ""},
        ]
    }]
}))
' "$TASK_NAME")

TASK_ID=$(_api "tasks" -X POST \
  -H "Content-Type: application/json" -d "$TASK_PAYLOAD" | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
printf "      Task ID: %s\n" "$TASK_ID"

# ================================================================
# 4. Upload video and wait for CVAT to finish processing
# ================================================================
printf "[4/6] Uploading %s...\n" "$(basename "$VIDEO_PATH")"
_api "tasks/${TASK_ID}/data" -X POST \
  -F "image_quality=70" \
  -F "use_zip_chunks=False" \
  -F "chunk_type=video" \
  -F "client_files[0]=@${VIDEO_PATH}" >/dev/null

printf "      Processing"
for i in $(seq 1 60); do
  STATE=$(_api "tasks/${TASK_ID}/status" | \
    python3 -c "import sys,json; print(json.load(sys.stdin).get('state','Unknown'))" \
    2>/dev/null || printf "Unknown")
  case "$STATE" in
    Completed|Finished) printf " done.\n"; break ;;
    Failed) printf "\nerror: CVAT video processing failed\n" >&2; exit 1 ;;
    *) printf "."; sleep 3 ;;
  esac
  [[ $i -eq 60 ]] && { printf "\nerror: processing timed out after 3 min\n" >&2; exit 1; }
done

# ================================================================
# 5. Pre-annotate with the storePose pipeline and upload for review
# ================================================================
printf "[5/6] Generating box pre-annotations with the pipeline...\n"
printf "      (CVAT_PREANNO_MAXFRAMES=N to limit, --no-cache to regen, CVAT_NO_PREANNO=1 to skip)\n"
if [[ -z "${CVAT_NO_PREANNO:-}" ]]; then
  # Cache the pipeline output per (clip, max-frames) so re-runs of the same
  # video skip the (slow) full-clip inference. --no-cache / CVAT_NO_CACHE regens.
  mkdir -p runs/preanno
  _MF="${CVAT_PREANNO_MAXFRAMES:-all}"
  PREANNO_FILE="runs/preanno/$(basename "$STEM")__mf${_MF}.xml"
  PREANNO_OK=1
  if [[ -z "$NO_CACHE" && -f "$PREANNO_FILE" && "$PREANNO_FILE" -nt "$VIDEO_PATH" ]]; then
    printf "      Using cached pre-annotation: %s (--no-cache to regenerate)\n" "$PREANNO_FILE"
  else
    PREANNO_ARGS=()
    [[ -n "${CVAT_PREANNO_MAXFRAMES:-}" ]] && PREANNO_ARGS+=(--max-frames "$CVAT_PREANNO_MAXFRAMES")
    if ! uv run python busy_report.py export-cvat "$VIDEO_PATH" -o "$PREANNO_FILE" \
         "${PREANNO_ARGS[@]+"${PREANNO_ARGS[@]}"}"; then
      printf "      warning: pre-annotation failed; review starts from a blank task.\n"
      PREANNO_OK=""
    fi
  fi
  if [[ -n "$PREANNO_OK" ]]; then
    printf "      Importing pre-annotations into task %s via cvat-cli...\n" "$TASK_ID"
    if _cli task import-dataset "$TASK_ID" "$PREANNO_FILE" --format "CVAT 1.1"; then
      printf "      Pre-annotations loaded.\n"
    else
      printf "      warning: cvat-cli import failed; review starts from a blank task.\n"
    fi
  fi
fi

# ================================================================
# 6. Open annotation UI, wait for completion, export XML
# ================================================================
JOB_URL="${CVAT_URL}/tasks/${TASK_ID}/jobs"
printf "\n=== Review the pre-annotations ===\n"
printf "URL    : %s\n" "$JOB_URL"
printf "Labels : person (rectangle, track mode)\n"
printf "         membership = in_line  (stationary, queuing intent)\n"
printf "         membership = bystander  (transiting or non-queuing)\n"
printf "Review : the pipeline pre-filled a box track per person. Delete false\n"
printf "         positives (e.g. a merchandise stand), drag boxes that drifted,\n"
printf "         and flip membership where the in_line default is wrong.\n\n"
open "$JOB_URL" 2>/dev/null || xdg-open "$JOB_URL" 2>/dev/null || true

printf "Press Enter when annotation is complete: "
read -r

printf "\n[6/6] Exporting annotations via cvat-cli...\n"
rm -f "$EXPORT_TMP"   # cvat-cli export-dataset refuses to overwrite an existing file
if ! _cli task export-dataset "$TASK_ID" "$EXPORT_TMP" \
     --format "CVAT for video 1.1" --with-images False; then
  printf "error: cvat-cli export failed\n" >&2
  exit 1
fi

# export-dataset writes a zip; extract annotations.xml from it
if unzip -l "$EXPORT_TMP" >/dev/null 2>&1; then
  XML_NAME=$(unzip -Z1 "$EXPORT_TMP" | grep -m1 '\.xml$')
  unzip -p "$EXPORT_TMP" "$XML_NAME" > "$OUT_PATH"
else
  cp "$EXPORT_TMP" "$OUT_PATH"
fi

printf "\nSaved: %s\n\n" "$OUT_PATH"
printf "Next steps:\n"
printf "  1. Convert annotations to a GT occupancy timeline:\n"
printf "     uv run python busy_report.py import-cvat \"%s\" --fps %s --step 1 -o gt_occupancy.csv\n" "$OUT_PATH" "$FPS"
printf "\n  2. Run the pipeline on the same clip (if not done already):\n"
printf "     uv run python main.py --source \"%s\" --zone zones/<clip>.json --wait-log waits.csv\n" "$VIDEO_PATH"
printf "\n  3. Score:\n"
printf "     uv run python busy_report.py eval-occupancy gt_occupancy.csv waits.csv --step 1\n\n"
