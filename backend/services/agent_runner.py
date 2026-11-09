from __future__ import annotations

import asyncio
import threading
import traceback

from agent.graph import run_agent
from backend.services import event_store, project_store

# Long single steps (a Seedance render is 60-90s, a full episode several
# minutes) mean node-boundary heartbeats alone are not enough to distinguish
# "slow" from "dead". A timer keeps the marker fresh while the graph works.
HEARTBEAT_INTERVAL_S = 45.0


class _Heartbeat:
    def __init__(self, project_id: str) -> None:
        self._project_id = project_id
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_S):
            try:
                project_store.heartbeat(self._project_id)
            except Exception:  # a missed beat must never kill the run
                pass

    def __enter__(self) -> "_Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


async def run_project_agent(project_id: str) -> None:
    await asyncio.to_thread(_run_project_agent_sync, project_id)


def _run_project_agent_sync(project_id: str) -> None:
    try:
        state = project_store.load_state(project_id)
        event_store.append_event(project_id, "Thought", "Agent 后台任务启动。")
        with _Heartbeat(project_id):
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
