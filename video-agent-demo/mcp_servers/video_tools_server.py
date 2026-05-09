"""MCP-like video tools.

The functions keep a model-agnostic interface so the Agent can call tools
without depending on a specific image or video provider.
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1] / "workspace"
PROJECTS_ROOT = WORKSPACE_ROOT / "projects"

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _project_dir(project_id: str) -> Path:
    return PROJECTS_ROOT / project_id


def generate_image(prompt: str, project_id: str, shot_id: str) -> dict:
    image_path = _project_dir(project_id) / "images" / f"{shot_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(_TINY_PNG)

    meta_path = image_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps({"prompt": prompt, "kind": "mock_image"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"image_path": str(image_path.relative_to(_project_dir(project_id)))}


def evaluate_image(image_path: str, shot_description: str, style: str) -> dict:
    prompt_quality = min(len(shot_description) / 80, 0.25)
    style_bonus = 0.15 if style else 0
    score = round(0.62 + prompt_quality + style_bonus, 2)
    passed = score >= 0.72
    feedback = "画面与分镜和风格基本一致。" if passed else "画面信息不足，需要强化主体、风格和构图。"
    return {"score": score, "passed": passed, "feedback": feedback}


def generate_video(image_path: str, motion_prompt: str, project_id: str, shot_id: str) -> dict:
    video_path = _project_dir(project_id) / "videos" / f"{shot_id}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_text(
        f"MOCK VIDEO\nsource_image={image_path}\nmotion_prompt={motion_prompt}\n",
        encoding="utf-8",
    )
    return {"video_path": str(video_path.relative_to(_project_dir(project_id)))}


def evaluate_video(video_path: str, shot_description: str) -> dict:
    score = round(0.7 + min(len(shot_description) / 120, 0.2), 2)
    passed = score >= 0.75
    feedback = "运动节奏满足分镜要求。" if passed else "运动描述偏弱，需要增加镜头运动和情绪节奏。"
    return {"score": score, "passed": passed, "feedback": feedback}


def compose_video(video_paths: list[str], project_id: str) -> dict:
    project_dir = _project_dir(project_id)
    final_path = project_dir / "final" / "final_video.mp4"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(
        "MOCK FINAL VIDEO\n" + "\n".join(video_paths),
        encoding="utf-8",
    )

    manifest_path = final_path.with_suffix(".json")
    manifest_path.write_text(
        json.dumps({"video_paths": video_paths}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"final_video_path": str(final_path.relative_to(project_dir))}


def copy_sample_if_exists(sample_path: str, target_path: Path) -> bool:
    source = Path(sample_path)
    if not source.exists():
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target_path)
    return True
