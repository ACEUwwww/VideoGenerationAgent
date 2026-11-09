"""Repair nodes: LLM-driven prompt rewriting from QA feedback."""

from __future__ import annotations

import json

from agent.llm_client import get_llm
from agent.nodes.image_nodes import shot_anchor
from agent.prompts import REWRITE_IMAGE_SYSTEM, REWRITE_MOTION_SYSTEM
from agent.state import AgentState, add_event, current_shot, track_usage, update_current_shot


def rewrite_image_prompt(state: AgentState) -> AgentState:
    shot = current_shot(state)
    retry_count = shot.get("image_retry_count", 0) + 1
    add_event(state, "Thought", f"根据质检反馈修复 {shot['id']} 的关键帧提示词。")

    llm = get_llm()
    request = (
        f"原提示词：{shot.get('image_prompt', '')}\n"
        f"原负面提示词：{shot.get('negative_prompt', '')}\n"
        f"质检问题：{json.dumps(shot.get('feedback', ''), ensure_ascii=False)}\n"
        f"分镜：{json.dumps(shot, ensure_ascii=False)}\n"
        f"角色外观锚点（逐字保留）：{shot_anchor(state, shot)}"
    )
    data, reply = llm.generate_json(REWRITE_IMAGE_SYSTEM, request, required_keys=("image_prompt",))
    track_usage(state, f"{shot['id']}/repair_image", reply)
    add_event(state, "Decision", data.get("change_log", "按质检反馈重写提示词。"), {"retry_count": retry_count})
    return update_current_shot(
        state,
        {
            "image_prompt": data["image_prompt"],
            "negative_prompt": data.get("negative_prompt", shot.get("negative_prompt", "")),
            "image_retry_count": retry_count,
        },
    )


def rewrite_video_prompt(state: AgentState) -> AgentState:
    shot = current_shot(state)
    retry_count = shot.get("video_retry_count", 0) + 1
    add_event(state, "Thought", f"根据质检反馈修复 {shot['id']} 的运动提示词。")

    llm = get_llm()
    request = (
        f"原运动提示词：{shot.get('motion_prompt', '')}\n"
        f"质检问题：{json.dumps(shot.get('feedback', ''), ensure_ascii=False)}\n"
        f"分镜：{json.dumps(shot, ensure_ascii=False)}\n"
        f"时长：{shot['duration_s']} 秒"
    )
    data, reply = llm.generate_json(REWRITE_MOTION_SYSTEM, request, required_keys=("motion_prompt",))
    track_usage(state, f"{shot['id']}/repair_video", reply)
    add_event(state, "Decision", data.get("change_log", "按质检反馈重写运动提示词。"), {"retry_count": retry_count})
    return update_current_shot(
        state,
        {"motion_prompt": data["motion_prompt"], "video_retry_count": retry_count},
    )
