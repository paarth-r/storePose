#!/usr/bin/env bash
# cvat-annotate.sh — full CVAT annotation pipeline for a video clip.
#
# Usage (direct):
#   ./viewscripts/cvat-annotate.sh --video videos/clip.mp4 [--fps 30] [--out clip-cvat.xml]
#
# Usage (via launcher TUI):
#   select "cvat-annotate" from the view list and press Enter.
#   set CVAT_VIDEO_PATH before launching to skip the interactive prompt.
#
# Flags:
#   --video PATH    video file to annotate (or set CVAT_VIDEO_PATH env)
#   --fps N         frame rate (default: auto-detect via ffprobe, else 30)
#   --out PATH      output XML path (default: <video-stem>-cvat.xml)
#   --help          print this header
#
# Environment:
#   CVAT_DIR        path to cloned cvat repo  (default: ~/cvat)
#   CVAT_URL        CVAT base URL             (default: http://localhost:8080)
#   CVAT_USER       superuser username        (default: admin)
#   CVAT_PASS       superuser password        (default: admin)
#   CVAT_VIDEO_PATH video path (alternative to --video flag)
#
# Unknown flags (--no-dashboard, --debug, --save-mp4, etc.) passed by the
# launcher TUI are silently ignored.
#
# Steps:
#   1. Start CVAT docker containers if not running
#   2. Authenticate and obtain an API token
#   3. Create a CVAT task with the storePose label schema (person/points/track,
#      membership select: in_line | bystander)
#   4. Upload the video and wait for CVAT to process it
#   5. Open the annotation UI in the browser; wait for you to finish
#   6. Export annotations as "CVAT for video 1.1" XML
#
# After this script finishes, run:
#   uv run python busy_report.py import-cvat <out.xml> --fps <N> --step 1 -o gt.csv
set -uo pipefail
cd "$(dirname "$0")/.."

# ---- configurable defaults ----
CVAT_DIR="${CVAT_DIR:-$HOME/cvat}"
CVAT_URL="${CVAT_URL:-http://localhost:8080}"
CVAT_USER="${CVAT_USER:-admin}"
CVAT_PASS="${CVAT_PASS:-admin}"

VIDEO_PATH="${CVAT_VIDEO_PATH:-}"
FPS_EXPLICIT=""
OUT_PATH=""

# ---- flag parsing (unknown flags ignored for TUI compatibility) ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --video|-v)  VIDEO_PATH="$2"; shift 2 ;;
    --fps)       FPS_EXPLICIT="$2"; shift 2 ;;
    --out|-o)    OUT_PATH="$2"; shift 2 ;;
    --help|-h)   awk 'NR>1{if(/^[^#]/)exit; sub(/^# ?/,""); print}' "$0"; exit 0 ;;
    *)           shift ;;
  esac
done

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

# ---- auto-detect FPS (ffprobe; fallback to 30 or explicit flag) ----
FPS="${FPS_EXPLICIT:-30}"
if [[ -z "$FPS_EXPLICIT" ]] && command -v ffprobe >/dev/null 2>&1; then
  _D=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate \
       -of csv=p=0 "$VIDEO_PATH" 2>/dev/null | \
       python3 -c "import sys; f=sys.stdin.read().strip(); p=f.split('/'); \
                   print(round(float(p[0])/float(p[1]),4) if '/' in f else float(f))" \
       2>/dev/null || true)
  [[ -n "$_D" ]] && FPS="$_D"
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
printf "[1/5] Checking CVAT containers...\n"
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
for i in $(seq 1 30); do
  if curl -sf "${CVAT_URL}/api/server/about" -u "${CVAT_USER}:${CVAT_PASS}" >/dev/null 2>&1; then
    printf " ready.\n"; break
  fi
  printf "."
  sleep 2
  [[ $i -eq 30 ]] && { printf "\nerror: server not ready after 60s\n" >&2; exit 1; }
done

# ================================================================
# 2. Authenticate and get API token
# ================================================================
printf "[2/5] Authenticating...\n"
TOKEN=$(curl -sf -X POST "${CVAT_URL}/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"${CVAT_USER}\",\"password\":\"${CVAT_PASS}\"}" | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['key'])")

_api() {
  curl -sf "${CVAT_URL}/api/$1" -H "Authorization: Token ${TOKEN}" "${@:2}"
}

# ================================================================
# 3. Create task with person/points/track label schema
# ================================================================
printf "[3/5] Creating task: %s\n" "$TASK_NAME"
TASK_PAYLOAD=$(python3 -c '
import json, sys
print(json.dumps({
    "name": sys.argv[1],
    "labels": [{
        "name": "person",
        "type": "points",
        "attributes": [
            {"name": "membership", "input_type": "select", "mutable": True,
             "values": ["in_line", "bystander"], "default_value": "in_line"},
            {"name": "state", "input_type": "text", "mutable": True,
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
printf "[4/5] Uploading %s...\n" "$(basename "$VIDEO_PATH")"
_api "tasks/${TASK_ID}/data" -X POST \
  -F "image_quality=70" \
  -F "use_zip_chunks=False" \
  -F "chunk_type=video" \
  -F "client_files=@${VIDEO_PATH}" >/dev/null

printf "      Processing"
for i in $(seq 1 60); do
  STATE=$(_api "tasks/${TASK_ID}/status" | \
    python3 -c "import sys,json; print(json.load(sys.stdin).get('state','Unknown'))" \
    2>/dev/null || printf "Unknown")
  case "$STATE" in
    Completed) printf " done.\n"; break ;;
    Failed) printf "\nerror: CVAT video processing failed\n" >&2; exit 1 ;;
    *) printf "."; sleep 3 ;;
  esac
  [[ $i -eq 60 ]] && { printf "\nerror: processing timed out after 3 min\n" >&2; exit 1; }
done

# ================================================================
# 5. Open annotation UI, wait for completion, export XML
# ================================================================
JOB_URL="${CVAT_URL}/tasks/${TASK_ID}/jobs"
printf "\n=== Annotate the video ===\n"
printf "URL    : %s\n" "$JOB_URL"
printf "Labels : person (points, track mode)\n"
printf "         membership = in_line  (stationary, queuing intent)\n"
printf "         membership = bystander  (transiting or non-queuing)\n"
printf "Tip    : mark entry with a point, exit with outside=1; add keyframes\n"
printf "         only when position or membership changes.\n\n"
open "$JOB_URL" 2>/dev/null || xdg-open "$JOB_URL" 2>/dev/null || true

printf "Press Enter when annotation is complete: "
read -r

printf "\n[5/5] Exporting annotations...\n"
EXPORT_URL="${CVAT_URL}/api/tasks/${TASK_ID}/annotations?format=CVAT%20for%20video%201.1"
for i in $(seq 1 30); do
  HTTP_CODE=$(curl -s -o "$EXPORT_TMP" -w "%{http_code}" \
    -H "Authorization: Token ${TOKEN}" "$EXPORT_URL")
  case "$HTTP_CODE" in
    200|201) printf "\n"; break ;;
    202) printf "."; sleep 3 ;;
    *) printf "\nerror: export returned HTTP %s\n" "$HTTP_CODE" >&2; exit 1 ;;
  esac
  [[ $i -eq 30 ]] && { printf "\nerror: export timed out\n" >&2; exit 1; }
done

# CVAT returns a zip archive; extract the XML from it
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
