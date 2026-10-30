"""Image nodes: prompt construction, generation, QA, retry decision."""

from __future__ import annotations

import json

from agent.llm_client import get_llm
from agent.prompts import IMAGE_PROMPT_SYSTEM
from agent.state import (
    AgentState,
    add_event,
    current_shot,
    track_media_usage,
    track_usage,
    update_current_shot,
)
from agent.settings import get_settings
from mcp_servers import keyframe_memory
from mcp_servers import video_tools_server as tools
from mcp_servers.evaluators import evaluate_image as qa_image



def shot_anchor(state: AgentState, shot: dict) -> str:
    """Concatenate the appearance anchors of the characters present in a shot."""
    wanted = set(shot.get("characters") or [])
    characters = state.get("characters") or []
    selected = [c for c in characters if not wanted or c.get("id") in wanted] or characters[:1]
    return "；".join(
        f"{c.get('name', c.get('id'))}: {c.get('appearance', '')} (wardrobe: {c.get('wardrobe', '')})"
        for c in selected
    )


def generate_image_prompt(state: AgentState) -> AgentState:
    shot = current_shot(state)
    anchor = shot_anchor(state, shot)
    add_event(state, "Thought", f"为 {shot['id']} 设计关键帧提示词，沿用角色外观锚点。")

    llm = get_llm()
    request = (
        f"分镜：{json.dumps(shot, ensure_ascii=False)}\n"
        f"场景设定：{state.get('setting', '')}\n"
        f"整体调性：{state.get('tone', '')}\n"
        f"风格要求：{state['style']}\n"
        f"角色外观锚点（必须逐字保留）：{anchor}"
    )
    data, reply = llm.generate_json(IMAGE_PROMPT_SYSTEM, request, required_keys=("image_prompt",))
    track_usage(state, f"{shot['id']}/image_prompt", reply)

    prompt = data["image_prompt"]
    if anchor and anchor.split(":")[0] not in prompt:
        prompt = f"{prompt} Character appearance: {anchor}."
    add_event(state, "Decision", data.get("rationale", "使用生成的关键帧提示词。"), {"prompt": prompt})
    return update_current_shot(
        state,
        {
            "status": "image_prompt_ready",
            "image_prompt": prompt,
            "negative_prompt": data.get("negative_prompt", ""),
        },
    )


def _anchor_reference(state: AgentState, root, media) -> str:
    """Path of the keyframe-memory anchor, or "" when there is nothing to anchor to.

    The first shot has no anchor by definition; it establishes one when its clip
    is scored. Later shots are conditioned on it so faces, wardrobe and colour
    grade carry over instead of being re-invented per shot.
    """
    if not media.image_reference or media.image_provider == "mock":
        return ""
    entry = keyframe_memory.anchor_entry(root)
    if not entry:
        return ""
    path = root / entry["path"]
    return entry["path"] if path.exists() else ""


def call_generate_image(state: AgentState) -> AgentState:
    shot = current_shot(state)
    media = get_settings().media
    root = tools.ensure_project_dirs(state["project_id"])
    reference = _anchor_reference(state, root, media)
    add_event(
        state,
        "Action",
        "MCP.generate_image",
        {"shot_id": shot["id"], "provider": media.image_provider, "reference": reference or "none"},
    )
    result = tools.generate_image(
        shot["image_prompt"],
        state["project_id"],
        shot["id"],
        negative_prompt=shot.get("negative_prompt", ""),
        caption=f"{shot.get('shot_size', '')} · {shot.get('scene', '')}",
        seed=shot["index"] + 1,
        reference_path=reference,
    )
    if media.image_provider != "mock":
        track_media_usage(state, "generate_image", media.image_provider, result.get("usage"))
    add_event(
        state,
        "Observation",
        "图片工具返回结果。",
        {"image_path": result["image_path"], "provider": result["provider"], "used_reference": bool(reference)},
    )
    state["generated_image_path"] = result["image_path"]
    return update_current_shot(state, {"status": "image_generated", "image_path": result["image_path"]})


def call_evaluate_image(state: AgentState) -> AgentState:
    shot = current_shot(state)
    root = tools.ensure_project_dirs(state["project_id"])
    media = get_settings().media
    add_event(state, "Action", "QA.evaluate_image", {"image_path": shot["image_path"]})

    result = qa_image(
        root / shot["image_path"],
        width=media.width,
        height=media.height,
        prompt=shot.get("image_prompt", ""),
        anchor=shot_anchor(state, shot),
        provider=media.image_provider,
    )
    add_event(state, "Observation", "关键帧质检完成。", {"score": result["score"], "issues": result["issues"]})
    state["evaluation_score"] = result["score"]
    return update_current_shot(
        state,
        {
            "status": "image_passed" if result["passed"] else "image_failed",
            "image_score": result["score"],
            "feedback": result["feedback"],
            "image_passed": result["passed"],
            "image_mock": result.get("mock", False),
        },
    )


def should_retry_image(state: AgentState) -> str:
    shot = current_shot(state)
    if shot.get("image_mock"):
        add_event(state, "Decision", f"{shot['id']} 使用 mock 关键帧，返工无法改善，直接进入视频生成。")
        return "video"
    if shot.get("image_passed"):
        add_event(state, "Decision", f"{shot['id']} 关键帧通过质检，进入视频生成。")
        return "video"
    if shot.get("image_retry_count", 0) < get_settings().media.image_retries:
        add_event(state, "Decision", f"{shot['id']} 关键帧未通过，重写提示词后重试。")
        return "retry"
    add_event(state, "Decision", f"{shot['id']} 达到最大重试次数，保留当前关键帧继续。")
    return "video"
