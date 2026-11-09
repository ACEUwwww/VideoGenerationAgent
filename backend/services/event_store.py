from __future__ import annotations

import json
from typing import Any

from backend.services.project_store import now_iso, project_dir


def append_event(project_id: str, event_type: str, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    event = {
        "ts": now_iso(),
        "type": event_type,
        "message": message,
        "data": data or {},
    }
    events_path = project_dir(project_id) / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def list_events(project_id: str, since: int = 0) -> dict[str, Any]:
    events_path = project_dir(project_id) / "events.jsonl"
    if not events_path.exists():
        return {"offset": 0, "events": []}

    events = []
    lines = events_path.read_text(encoding="utf-8").splitlines()
    for line in lines[since:]:
        if line.strip():
            events.append(json.loads(line))
    return {"offset": len(lines), "events": events}
