"""Graph state and persistence helpers.

The state is the single source of truth for one project: plan, cast, shots and
media paths. Every node mutates it and persists through ``project_store`` so the
frontend can poll a crash-safe snapshot.
"""

from __future__ import annotations

from typing import Any, TypedDict

from backend.services import event_store, project_store


class AgentState(TypedDict, total=False):
    project_id: str
    user_prompt: str
    script: str
    style: str
    genre: str
    shot_count: int
    # Requested episode length. LangGraph only carries keys declared here, so an
    # undeclared field is silently dropped at the first node boundary.
    target_seconds: int
    status: str
    error: str

    logline: str
    tone: str
    setting: str
    outline: str
    hook: str
    ending: str
    # One entry per shot, produced by plan_story and consumed by plan_shots.
    beats: list[str]
    characters: list[dict[str, Any]]

    shots: list[dict[str, Any]]
    current_shot_index: int

    generated_image_path: str
    generated_video_path: str
    evaluation_score: float
    retry_count: int

    keyframe_memory: dict[str, Any]
    episode_report: dict[str, Any]
    cover_image_path: str
    final_video_path: str
    subtitle_path: str
    event_count: int
    usage: dict[str, Any]


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


def add_event(
    state: AgentState,
    event_type: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> AgentState:
    """Append to events.jsonl and keep only a counter in state.

    Events are read incrementally from the jsonl by the frontend; duplicating
    the whole list in state.json made the file grow to megabytes and rewrote it
    on every node.
    """
    event_store.append_event(state["project_id"], event_type, message, data)
    state["event_count"] = state.get("event_count", 0) + 1
    return state


def track_usage(state: AgentState, stage: str, reply: Any) -> None:
    """Accumulate LLM token usage so a run can be costed after the fact."""
    usage = dict(state.get("usage") or {})
    stages = dict(usage.get("stages") or {})
    entry = stages.get(stage, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0})
    entry["calls"] += 1
    entry["prompt_tokens"] += getattr(reply, "prompt_tokens", 0)
    entry["completion_tokens"] += getattr(reply, "completion_tokens", 0)
    entry["latency_s"] = round(entry["latency_s"] + getattr(reply, "latency_s", 0.0), 2)
    stages[stage] = entry
    usage["stages"] = stages
    usage["prompt_tokens"] = sum(v["prompt_tokens"] for v in stages.values())
    usage["completion_tokens"] = sum(v["completion_tokens"] for v in stages.values())
    usage["calls"] = sum(v["calls"] for v in stages.values())
    state["usage"] = usage


def track_media_usage(state: AgentState, stage: str, provider: str, usage: dict[str, Any] | None) -> None:
    """Accumulate paid media usage so a finished episode can be costed."""
    usage_summary = dict(state.get("usage") or {})
    media = dict(usage_summary.get("media") or {})
    entry = media.get(stage, {"provider": provider, "calls": 0, "tokens": 0})
    entry["calls"] += 1
    entry["tokens"] += int((usage or {}).get("total_tokens", 0) or 0)
    media[stage] = entry
    usage_summary["media"] = media
    usage_summary["media_calls"] = sum(item["calls"] for item in media.values())
    usage_summary["media_tokens"] = sum(item["tokens"] for item in media.values())
    state["usage"] = usage_summary


def persist(state: AgentState) -> AgentState:
    project_store.save_state(state["project_id"], dict(state))
    return state
