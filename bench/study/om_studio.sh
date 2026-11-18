#!/usr/bin/env bash
# Launch the OpenMontage Drama Studio for the benchmark's full-workflow baseline.
#
# Separate from the vendored bench_run/start_studio.sh on purpose: that script is
# part of the untracked vendor checkout and points the studio's shot QA at
# 127.0.0.1:8790 (the 7B judge), which is not the judge this study uses and is
# not even running. A baseline whose own internal QA consults a different, dead
# model is not a controlled comparison, so the QA endpoint is pinned to the same
# live 32B judge that scores both workflows.
#
# Everything else matches the vendor launcher: local mode, no Postgres/Redis/
# MinIO, Seedance + Seedream pinned, credentials from the project .env.
set -euo pipefail

ROOT="/mdata/shiweisong/VideoGenerationAgent"
OM="$ROOT/bench/_vendor/OpenMontage"
RUN="$OM/bench_run"
JUDGE_BASE="${BENCH_JUDGE_BASE_URL:-http://127.0.0.1:8791/v1}"
JUDGE_MODEL="${BENCH_JUDGE_MODEL:-qwen2.5-vl-32b-local}"

mkdir -p "$RUN/projects" "$RUN/logs" "$ROOT/bench/logs"

set -a
. "$ROOT/.env"
set +a

export PROJECTS_ROOT="$RUN/projects"
export OPENMONTAGE_ROOT="$OM"
export STUDIO_DIST="$OM/studio/dist"
export HOST=127.0.0.1
export PORT=8787
export STUDIO_AUTH_REQUIRED=true
export STUDIO_DEMO_ENABLED=true
export STUDIO_DEMO_AUTO_ENTER=false
export STUDIO_PUBLIC_URL=http://127.0.0.1:8787
export STUDIO_CORS_ORIGINS=http://127.0.0.1:8787
# Two at a time: Seedance bills per task and the account's quota is the binding
# constraint, not local GPU. Raising this risks 429s that look like failures.
export STUDIO_MEDIA_WORKERS=2
export STUDIO_VIDEO_PARALLEL=2

# Both media backends pinned, so a run cannot silently depend on a ComfyUI probe.
export STUDIO_VIDEO_BACKEND=seedance
export STUDIO_IMAGE_BACKEND=seedream
export STUDIO_FREE_FIRST=0
export SEEDREAM_MODEL="${ARK_IMAGE_MODEL}"
export SEEDANCE_MODEL="${ARK_VIDEO_MODEL}"
export SEEDANCE_API_URL="${ARK_BASE_URL}/contents/generations/tasks"

# One QA model for both workflows, on the port that is actually serving.
export OPENAI_API_KEY="local"
export OPENAI_BASE_URL="$JUDGE_BASE"
export OPENAI_MODEL="$JUDGE_MODEL"
export STUDIO_QA_VISION_MODEL="$JUDGE_MODEL"
# Keep the hosted vision model out of the fallback chain: a silent switch to a
# different scorer mid-run mixes two judges in one result set.
unset ARK_MODEL

# Bounded retries. The default ladder can re-render a segment up to ten times,
# each one billed, when the local judge is unhappy with a prompt fit. Two is
# enough to show the loop works without turning one bad segment into a budget.
export STUDIO_VIDEO_QA_MAX_ATTEMPTS=2
export STUDIO_ENTITY_QA_MAX_ATTEMPTS=2

unset DATABASE_URL STUDIO_DATABASE_URL REDIS_URL STUDIO_REDIS_URL MINIO_ENDPOINT

export EDGE_TTS_PROXY="${EDGE_TTS_PROXY:-${HTTPS_PROXY:-}}"
export PATH="$RUN/bin:$PATH"

cd "$OM"
echo "$$" > "$RUN/studio.pid"
exec "$OM/.venv-studio/bin/python" -u bench_run/serve.py
