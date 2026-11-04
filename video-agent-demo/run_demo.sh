#!/usr/bin/env bash
# One-command demo: install check, tests, then serve API + UI.
# Usage:
#   ./run_demo.sh check     # 环境与 provider 自检（不花钱）
#   ./run_demo.sh test      # 跑测试
#   ./run_demo.sh serve     # 启动后端 + 前端开发服务器
#   ./run_demo.sh simulate  # 本地 mock 跑一部 3 镜短剧（零成本）
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
UVICORN=".venv/bin/uvicorn"

need_venv() {
  if [[ ! -x "$PY" ]]; then
    echo "缺少虚拟环境，先执行：python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
  fi
}

case "${1:-check}" in
  check)
    need_venv
    [[ -f .env ]] || { echo "缺少 .env，执行 cp .env.example .env 并填入 key"; exit 1; }
    "$PY" scripts/doctor.py
    ;;
  test)
    need_venv
    "$PY" -m pytest tests -q
    ;;
  simulate)
    need_venv
    echo "== 本地 mock 跑 3 镜（零成本，用于验证流程完好）=="
    IMAGE_PROVIDER=mock VIDEO_PROVIDER=mock "$PY" - <<'PYEOF'
import glob, json, time
from pathlib import Path
from backend.services import project_store
from agent.graph import run_agent
from agent.settings import reset_settings
reset_settings()
t0 = time.time()
project = project_store.create_project("深夜便利店，店员在监控里看到慢半拍的自己", "电影感竖屏短剧", 3, genre="悬疑")
state = project_store.load_state(project["project_id"])
state["status"] = "running"
final = run_agent(state)
root = project_store.project_dir(project["project_id"])
print("耗时", round(time.time() - t0, 1), "s |", final["status"])
print("成片", root / final["final_video_path"])
print("综合评分", final["episode_report"]["score"])
PYEOF
    ;;
  serve)
    need_venv
    echo "后端 http://localhost:8000  前端另开终端: cd frontend && npm run dev"
    exec "$UVICORN" backend.main:app --host 0.0.0.0 --port 8000
    ;;
  *)
    echo "用法: ./run_demo.sh [check|test|simulate|serve]"
    exit 1
    ;;
esac
