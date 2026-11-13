#!/usr/bin/env bash
# Verify downloaded weights against the sha256 HuggingFace publishes.
#
# Necessary because a size check cannot detect corruption: the earlier
# downloader wrote every chunk at the wrong offset and produced files with
# exactly the right length. HF exposes the expected digest as the ETag of a
# ranged HEAD request (x-linked-etag for LFS files).
#
# usage: verify_models.sh [org/repo] [--fix]
set -u
REPO="${1:?usage: verify_models.sh <org/repo> [--fix]}"
FIX=0
[ "${2:-}" = "--fix" ] && FIX=1
ROOT="$(cd "$(dirname "$0")" && pwd)/models/hf/$REPO"
HERE="$(cd "$(dirname "$0")" && pwd)"

bad=0
checked=0
unverified=0
# The file list comes from the hub API, which drops connections under load.
# An empty or failed listing must fail the run: reporting "ALL FILES VERIFIED"
# after verifying nothing is worse than reporting an error.
LIST=$(python3 - "$REPO" <<'PY'
import json, sys, time, urllib.request
repo = sys.argv[1]
for attempt in range(5):
    try:
        req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}?blobs=true")
        data = json.load(urllib.request.urlopen(req, timeout=60))
        break
    except Exception:
        if attempt == 4:
            sys.exit(1)
        time.sleep(2 ** attempt)
for s in data.get("siblings", []):
    name = s["rfilename"]
    if name.endswith((".safetensors", ".json", ".txt", ".model")):
        print(f'{name}\t{s.get("size") or 0}')
PY
) || { echo "ERROR: could not list $REPO from the hub"; exit 2; }
if [ -z "$LIST" ]; then echo "ERROR: empty file list for $REPO"; exit 2; fi

while IFS=$'\t' read -r path size; do
  [ -z "$path" ] && continue
  checked=$((checked+1))
  target="$ROOT/$path"
  if [ ! -f "$target" ]; then
    echo "MISSING  $path"; bad=$((bad+1)); continue
  fi
  actual_size=$(stat -c%s "$target")
  if [ "$actual_size" != "$size" ]; then
    echo "SIZE     $path  $actual_size != $size"; bad=$((bad+1)); continue
  fi
  # Only LFS files carry a sha256; small JSONs have a plain content ETag.
  [ "$size" -lt 33554432 ] && continue
  expected=$(curl -sIL --retry 3 "https://huggingface.co/$REPO/resolve/main/$path" 2>/dev/null \
    | tr -d '\r' | awk 'tolower($1)=="x-linked-etag:"{gsub(/"/,"",$2); print $2}' | tail -1)
  if [ -z "$expected" ]; then
    # A file we could not check is not a file we verified. Counting it as a
    # pass is how a run reports "ALL VERIFIED" having proved nothing.
    echo "NODIGEST $path (not verified)"; unverified=$((unverified+1)); continue
  fi
  actual=$(sha256sum "$target" | cut -d' ' -f1)
  if [ "$actual" != "$expected" ]; then
    echo "CORRUPT  $path"
    echo "         expected ${expected:0:16}... got ${actual:0:16}..."
    bad=$((bad+1))
    if [ "$FIX" = "1" ]; then
      rm -f "$target"
      bash "$HERE/dl_ranges.sh" "https://huggingface.co/$REPO/resolve/main/$path" "$target" "$size" 12 \
        && echo "         repaired"
    fi
  else
    echo "ok       $path"
  fi
done <<< "$LIST"
echo "---"
if [ "$bad" -gt 0 ]; then
  echo "$bad file(s) need attention"; exit 1
fi
if [ "$unverified" -gt 0 ]; then
  echo "$unverified file(s) could not be checked (no published digest);"
  echo "inspect these manually before treating the weights as verified"
  exit 3
fi
echo "ALL FILES VERIFIED ($REPO) — $checked files, digest match"
