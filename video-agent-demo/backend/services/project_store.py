from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[2] / "workspace"
PROJECTS_ROOT = WORKSPACE_ROOT / "projects"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_dir(project_id: str) -> Path:
    return PROJECTS_ROOT / project_id


def ensure_project_dirs(project_id: str) -> Path:
    root = project_dir(project_id)
    for name in ["shots", "images", "videos", "final"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def create_project(user_prompt: str, style: str, shot_count: int) -> dict[str, Any]:
    project_id = f"project_{uuid.uuid4().hex[:8]}"
    root = ensure_project_dirs(project_id)
    project = {
        "project_id": project_id,
        "status": "created",
        "user_prompt": user_prompt,
        "style": style,
        "shot_count": shot_count,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    write_json(root / "project.json", project)
    write_json(root / "state.json", initial_state(project))
    (root / "events.jsonl").write_text("", encoding="utf-8")
    return project


def initial_state(project: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": project["project_id"],
        "user_prompt": project["user_prompt"],
        "style": project["style"],
        "shot_count": project["shot_count"],
        "status": project["status"],
        "story_outline": "",
        "shots": [],
        "current_shot_index": 0,
        "evaluation_score": 0.0,
        "retry_count": 0,
        "agent_events": [],
        "final_video_path": "",
    }


def get_project(project_id: str) -> dict[str, Any]:
    root = project_dir(project_id)
    if not root.exists():
        raise FileNotFoundError(project_id)
    project = read_json(root / "project.json")
    state = read_json(root / "state.json")
    return {**project, "state": state}


def load_state(project_id: str) -> dict[str, Any]:
    return read_json(project_dir(project_id) / "state.json")


def save_state(project_id: str, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json(project_dir(project_id) / "state.json", state)

    project_path = project_dir(project_id) / "project.json"
    project = read_json(project_path)
    project["status"] = state.get("status", project.get("status", "running"))
    project["updated_at"] = now_iso()
    write_json(project_path, project)


def save_shot(project_id: str, shot: dict[str, Any]) -> None:
    shot_path = project_dir(project_id) / "shots" / f"{shot['id']}.json"
    write_json(shot_path, shot)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
