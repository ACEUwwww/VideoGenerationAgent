#!/usr/bin/env bash
# Finish the repeated-measure study once rendering is done.
#
# Rendering runs for hours, so scoring must not depend on someone watching it.
# This waits for the renderer to exit, then scores every episode, recomputes the
# statistics and redraws the figures. Each step is idempotent: scoring skips
# episodes already in its JSON, and the analysis and figure scripts read only
# what those files contain.
set -u
cd /mdata/shiweisong/VideoGenerationAgent || exit 1
PY=bench/.venv/bin/python
LOG=/tmp/repeats_pipeline.log
: > "$LOG"

echo "[pipeline] waiting for renderer $(date -u +%H:%M)" | tee -a "$LOG"
while pgrep -f "tier3a_repeats.py --models" >/dev/null 2>&1; do
    sleep 60
done
echo "[pipeline] renderer exited $(date -u +%H:%M)" | tee -a "$LOG"
echo "[pipeline] clips: $(find bench/results/tier3a_repeats -name '*.mp4' | wc -l)" | tee -a "$LOG"

for model in wan21_1_3b cogvideox5b; do
    echo "[pipeline] scoring $model" | tee -a "$LOG"
    "$PY" bench/study/score_repeats.py --model "$model" >> "$LOG" 2>&1
done

echo "[pipeline] analysis" | tee -a "$LOG"
"$PY" bench/study/repeats_analysis.py >> "$LOG" 2>&1
echo "[pipeline] figures" | tee -a "$LOG"
"$PY" bench/study/repeats_figures.py >> "$LOG" 2>&1
echo "[pipeline] filling the report" | tee -a "$LOG"
"$PY" bench/study/fill_report.py >> "$LOG" 2>&1
echo "[pipeline] done $(date -u +%H:%M)" | tee -a "$LOG"
