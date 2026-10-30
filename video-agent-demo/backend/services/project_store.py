from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.settings import PROJECTS_ROOT


_RUN_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_dir(project_id: str) -> Path:
    return PROJECTS_ROOT / project_id


def ensure_project_dirs(project_id: str) -> Path:
    root = project_dir(project_id)
    for name in ["shots", "images", "videos", "audio", "subtitles", "keyframes", "final"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def create_project(
    user_prompt: str,
    style: str,
    shot_count: int,
    *,
    script: str = "",
    genre: str = "",
) -> dict[str, Any]:
    project_id = f"project_{uuid.uuid4().hex[:8]}"
    root = ensure_project_dirs(project_id)
    project = {
        "project_id": project_id,
        "status": "created",
        "user_prompt": user_prompt,
        "script": script,
        "genre": genre,
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
        "script": project.get("script", ""),
        "genre": project.get("genre", ""),
        "style": project["style"],
        "shot_count": project["shot_count"],
        "status": project["status"],
        "logline": "",
        "tone": "",
        "setting": "",
        "outline": "",
        "hook": "",
        "ending": "",
        "characters": [],
        "shots": [],
        "current_shot_index": 0,
        "evaluation_score": 0.0,
        "retry_count": 0,
        "keyframe_memory": {"anchor": "", "entries": []},
        "episode_report": {},
        "cover_image_path": "",
        "subtitle_path": "",
        "event_count": 0,
        "final_video_path": "",
        "usage": {},
    }


def list_projects(limit: int = 30) -> list[dict[str, Any]]:
    projects = []
    if not PROJECTS_ROOT.exists():
        return projects
    for root in PROJECTS_ROOT.iterdir():
        if not root.is_dir() or not (root / "project.json").exists():
            continue
        project = read_json(root / "project.json")
        projects.append(
            {
                "project_id": project["project_id"],
                "status": project.get("status", ""),
                "user_prompt": project.get("user_prompt", ""),
                "logline": project.get("logline", ""),
                "created_at": project.get("created_at", ""),
                "updated_at": project.get("updated_at", ""),
            }
        )
    projects.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return projects[:limit]


def claim_run(project_id: str) -> bool:
    """Atomically move a project from created/completed into running.

    Returns False when another run already holds it. The check and the write
    happen under one lock, so two concurrent /run requests cannot both start.
    """
    root = project_dir(project_id)
    if not root.exists():
        raise FileNotFoundError(project_id)
    with _RUN_LOCK:
        state = load_state(project_id)
        if state.get("status") == "running":
            return False
        state["status"] = "running"
        save_state(project_id, state)
        return True


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
    for key in ("logline", "genre"):
        if state.get(key):
            project[key] = state[key]
    project["updated_at"] = now_iso()
    write_json(project_path, project)


def save_shot(project_id: str, shot: dict[str, Any]) -> None:
    shape = {key: value for key, value in shot.items() if key != "dims_detail"}
    write_json(project_dir(project_id) / "shots" / f"{shot['id']}.json", shape)


def save_bible(project_id: str, bible: dict[str, Any]) -> None:
    write_json(project_dir(project_id) / "bible.json", bible)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
