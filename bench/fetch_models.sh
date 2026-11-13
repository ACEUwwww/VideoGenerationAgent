#!/usr/bin/env bash
# Fetch a HuggingFace repo's weights with parallel ranged curl requests.
#
# huggingface_hub + hf_transfer stalled on this host: six shards opened, the
# process slept, throughput fell to 41 B/s. Plain urllib stalled the same way.
# Measured here: one curl stream gets ~700 KB/s and twelve get ~8 MB/s, so the
# cap is per-connection. Ranged parallel curl is what actually works.
#
# Usage: bench/fetch_models.sh zai-org/CogVideoX-5b [workers]
set -u
REPO="${1:?usage: fetch_models.sh <org/repo> [workers]}"
WORKERS="${2:-12}"
ROOT="$(cd "$(dirname "$0")" && pwd)/models/hf"
OUT="$ROOT/$REPO"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Every file the pipeline needs, including the SentencePiece vocab
# (tokenizer/spiece.model) — without it the T5 encoder produces zero
# embeddings and the model renders a pure black clip. Filtering to
# safetensors/json/txt silently skipped it.
python3 - "$REPO" <<'PY' > /tmp/bench_filelist.txt
import json, sys, urllib.request
repo = sys.argv[1]
req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}?blobs=true")
data = json.load(urllib.request.urlopen(req, timeout=60))
for s in data.get("siblings", []):
    name = s["rfilename"]
    if name.endswith((".safetensors", ".json", ".txt", ".model")):
        print(f'{name}\t{s.get("size") or 0}')
PY

while IFS=$'\t' read -r path size; do
  url="https://huggingface.co/$REPO/resolve/main/$path"
  target="$OUT/$path"
  if [ -f "$target" ] && [ "$(stat -c%s "$target")" = "$size" ]; then
    echo "skip $path"; continue
  fi
  if [ "$size" -lt 33554432 ]; then
    mkdir -p "$(dirname "$target")"
    curl -sL --retry 5 --max-time 600 -o "$target" "$url" || { echo "FAIL $path"; exit 1; }
    echo "got  $path"
  else
    echo "=== $path ($((size/1048576)) MB) ==="
    bash "$HERE/dl_ranges.sh" "$url" "$target" "$size" "$WORKERS" || exit 1
  fi
done < /tmp/bench_filelist.txt
echo "ALL DONE $REPO -> $OUT"
