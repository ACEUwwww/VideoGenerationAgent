from __future__ import annotations

import asyncio
import traceback

from agent.graph import run_agent
from backend.services import event_store, project_store


async def run_project_agent(project_id: str) -> None:
    await asyncio.to_thread(_run_project_agent_sync, project_id)


def _run_project_agent_sync(project_id: str) -> None:
    try:
        state = project_store.load_state(project_id)
        state["status"] = "running"
        project_store.save_state(project_id, state)
        event_store.append_event(project_id, "Thought", "Agent 后台任务启动。")
        run_agent(state)
    except Exception as exc:  # pragma: no cover - surfaced through events for the demo UI.
        state = project_store.load_state(project_id)
        state["status"] = "failed"
        state["error"] = str(exc)
        project_store.save_state(project_id, state)
        event_store.append_event(
            project_id,
            "Observation",
            "Agent 运行失败。",
            {"error": str(exc), "traceback": traceback.format_exc()},
        )
