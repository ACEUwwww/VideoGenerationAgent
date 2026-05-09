from __future__ import annotations

from typing import Any, TypedDict

from backend.services import event_store, project_store


class AgentState(TypedDict, total=False):
    project_id: str
    user_prompt: str
    style: str
    shot_count: int
    status: str
    story_outline: str
    shots: list[dict[str, Any]]
    current_shot_index: int
    generated_image_path: str
    generated_video_path: str
    evaluation_score: float
    retry_count: int
    agent_events: list[dict[str, Any]]
    final_video_path: str


def current_shot(state: AgentState) -> dict[str, Any]:
    return state["shots"][state.get("current_shot_index", 0)]


def update_current_shot(state: AgentState, updates: dict[str, Any]) -> AgentState:
    shots = [dict(shot) for shot in state.get("shots", [])]
    index = state.get("current_shot_index", 0)
    shots[index] = {**shots[index], **updates}
    state["shots"] = shots
    project_store.save_shot(state["project_id"], shots[index])
    project_store.save_state(state["project_id"], dict(state))
    return state


def add_event(state: AgentState, event_type: str, message: str, data: dict[str, Any] | None = None) -> AgentState:
    event = event_store.append_event(state["project_id"], event_type, message, data)
    state["agent_events"] = [*state.get("agent_events", []), event]
    project_store.save_state(state["project_id"], dict(state))
    return state


def persist(state: AgentState) -> AgentState:
    project_store.save_state(state["project_id"], dict(state))
    return state
