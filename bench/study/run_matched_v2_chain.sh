#!/usr/bin/env bash
# Chain the remaining steps of the matched-v2 comparison on the single GPU.
#
# 1. wait for our-side render to exit
# 2. re-run OpenMontage with the video-prompt translation in place
# 3. score both episodes with the same judge
#
# Sequential on purpose: every step uses the same GPU, and overlapping them
# would make each slower and the timings meaningless.
set -uo pipefail
ROOT=/mdata/shiweisong/VideoGenerationAgent
cd "$ROOT" || exit 1

PY=bench/.venv/bin/python

echo "[chain] waiting for our-side render ..."
while pgrep -f "bench/study/ours_matched_arm.py" >/dev/null; do sleep 30; done
echo "[chain] our side done: $(ls bench/results/om_matched_v2/ours/clips/ours/videos/*.mp4 2>/dev/null | wc -l) clips"

# Fresh demo token: the studio restarted, and the old bearer is rejected.
TOK=$(curl -s -X POST http://127.0.0.1:8787/api/auth/demo \
      -H 'Content-Type: application/json' -d '{}' \
      | "$PY" -c "import sys,json; print(json.load(sys.stdin)['data']['token'])")
printf '%s' "$TOK" > bench/_vendor/OpenMontage/bench_run/.token
echo "[chain] token refreshed len=${#TOK}"

echo "[chain] re-running OpenMontage with prompt translation ..."
"$PY" -u bench/study/om_full.py \
  --briefs bench/study/briefs_matched.json --seeds 0 --max-shots 12 \
  --out bench/results/om_matched_v2/om > /tmp/om_matched_v4.log 2>&1
echo "[chain] om exit=$? ; tail:"
tail -c 400 /tmp/om_matched_v4.log | tr '\r' '\n' | tail -5

OM_PROJECT=$("$PY" -c "
import json, pathlib
p = pathlib.Path('bench/results/om_matched_v2/om/episodes.json')
rows = json.loads(p.read_text()) if p.exists() else []
print(rows[-1].get('project_id', '')) if rows else print('')
")
echo "[chain] OM project from run record: $OM_PROJECT"

echo "[chain] scoring both sides ..."
"$PY" -u bench/study/score_matched_v2.py \
  --om-project "$OM_PROJECT" \
  --ours-dir bench/results/om_matched_v2/ours/clips/ours/videos \
  --max-shots 12 > /tmp/score_matched_v2.log 2>&1
echo "[chain] score exit=$?"
cat /tmp/score_matched_v2.log
echo "[chain] done"
