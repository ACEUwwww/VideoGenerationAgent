"""Zero-cost end-to-end run using the mock providers.

Verifies the whole chain — planning, rendering, scoring, subtitles, composition
— without spending anything on paid models.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.graph import run_agent  # noqa: E402
from agent.settings import reset_settings  # noqa: E402
from backend.services import project_store  # noqa: E402


def main() -> int:
    import os

    os.environ.setdefault("IMAGE_PROVIDER", "mock")
    os.environ.setdefault("VIDEO_PROVIDER", "mock")
    os.environ.setdefault("TTS_PROVIDER", "mock")
    os.environ["IMAGE_PROVIDER"] = "mock"
    os.environ["VIDEO_PROVIDER"] = "mock"
    os.environ["TTS_PROVIDER"] = "mock"
    reset_settings()

    started = time.time()
    project = project_store.create_project(
        "深夜便利店，店员在监控里看到慢半拍的自己", "电影感竖屏短剧", 3, genre="悬疑"
    )
    state = project_store.load_state(project["project_id"])
    state["status"] = "running"
    final = run_agent(state)

    root = project_store.project_dir(project["project_id"])
    print(f"耗时 {time.time() - started:.1f}s | 状态 {final['status']}")
    for shot in final.get("shots", []):
        print(f"  {shot['id']} {shot['shot_size']} 多维 {shot.get('multidim_score')}")
    print(f"成片 {root / final['final_video_path']}")
    print(f"综合评分 {final['episode_report']['score']}")
    return 0 if final["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
