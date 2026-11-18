#!/usr/bin/env bash
# Study A, free half: render every local open-source baseline on the stratified
# prompt set. No Ark call, no money, no judge. Clips land under
# bench/results/study/ and the runner reuses them on a later invocation, so a
# re-run cost nothing and only fills in seeds that are still missing.
set -euo pipefail
ROOT="/mdata/shiweisong/VideoGenerationAgent"
cd "$ROOT"

export BENCH_SAMPLE_FPS=2
export BENCH_MAX_EDGE=640

exec "$ROOT/bench/.venv/bin/python" -m bench.runner \
  --stage generate \
  --models cogvideox5b wan21_1_3b longlive_1_3b \
  --prompt-file "$ROOT/bench/study/prompts.json" \
  --seeds 0 1 2 3 4 --seconds 5.0 \
  --run-dir "$ROOT/bench/results/study" 2>&1 | grep -vE "Loading (weights|checkpoint|pipeline)"
