#!/usr/bin/env bash
# build-web.sh — build the Next.js dashboard export (web/out) when it is missing
# or stale, so any launch path serves the current UI. Idempotent and safe to call
# before every run: it is a no-op when web/out is up to date.
#
# If npm is unavailable or the build fails it warns and exits 0 — main.py then
# serves the self-contained legacy HTML page instead.
set -euo pipefail
cd "$(dirname "$0")"

[ -d web ] || exit 0
command -v npm >/dev/null 2>&1 || { echo "note: npm not found; using the legacy dashboard page." >&2; exit 0; }

idx="web/out/index.html"
sources=(web/app web/components web/lib web/package.json web/next.config.mjs)
# up to date when the export exists and no source is newer than it
if [ -f "$idx" ] && [ -z "$(find "${sources[@]}" -newer "$idx" 2>/dev/null)" ]; then
  exit 0
fi

echo "Building web dashboard (one-time / after changes)…" >&2
if ( cd web && { [ -d node_modules ] || npm install; } && npm run build ) >/tmp/storepose-web-build.log 2>&1; then
  echo "Web dashboard ready." >&2
else
  echo "warning: web build failed (see /tmp/storepose-web-build.log); serving the legacy page." >&2
fi
