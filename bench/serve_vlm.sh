#!/usr/bin/env bash
# Start a second local judge on another port, so two models can be compared on
# the same frames without either one displacing the other. The bench points at
# whichever port wins; nothing about the harness depends on the port number.
#
# Usage: ./serve_vlm.sh <model-dir-name> <name> <port>
#   ./serve_vlm.sh Qwen2.5-VL-7B-Instruct  qwen2.5-vl-7b-local   8790
#   ./serve_vlm.sh Qwen2.5-VL-32B-Instruct qwen2.5-vl-32b-local  8791
set -euo pipefail

ROOT="/mdata/shiweisong/VideoGenerationAgent"
DIR_NAME="${1:?model directory name under bench/models/vlm}"
SERVED="${2:?served model name}"
PORT="${3:?port}"

MODEL_DIR="$ROOT/bench/models/vlm/$DIR_NAME"
[ -d "$MODEL_DIR" ] || { echo "missing model dir: $MODEL_DIR" >&2; exit 1; }
mkdir -p "$ROOT/bench/logs"

exec env VLM_MODEL_DIR="$MODEL_DIR" \
  VLM_NAME="$SERVED" \
  VLM_PORT="$PORT" \
  "$ROOT/bench/.venv/bin/python" -u "$ROOT/bench/vlm_server.py"
