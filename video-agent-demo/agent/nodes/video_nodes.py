from __future__ import annotations

from agent.nodes.image_nodes import MAX_RETRIES
from agent.state import AgentState, add_event, current_shot, persist, update_current_shot
from mcp_servers import video_tools_server as tools


def generate_video_prompt(state: AgentState) -> AgentState:
    shot = current_shot(state)
    prompt = f"{shot['description']} 镜头缓慢推进，保持 {state['style']} 风格，强调情绪递进。"
    add_event(state, "Thought", f"为 {shot['id']} 设计 motion prompt。")
    add_event(state, "Decision", "用已通过或已接受的关键帧生成短视频片段。")
    return update_current_shot(state, {"status": "video_prompt_ready", "motion_prompt": prompt})


def call_generate_video(state: AgentState) -> AgentState:
    shot = current_shot(state)
    add_event(
        state,
        "Action",
        "MCP.generate_video",
        {"image_path": shot["image_path"], "motion_prompt": shot["motion_prompt"], "shot_id": shot["id"]},
    )
    result = tools.generate_video(shot["image_path"], shot["motion_prompt"], state["project_id"], shot["id"])
    add_event(state, "Observation", "视频工具返回结果。", result)
    state["generated_video_path"] = result["video_path"]
    return update_current_shot(state, {"status": "video_generated", "video_path": result["video_path"]})


def call_evaluate_video(state: AgentState) -> AgentState:
    shot = current_shot(state)
    add_event(state, "Action", "MCP.evaluate_video", {"video_path": shot["video_path"]})
    result = tools.evaluate_video(shot["video_path"], shot["description"])
    add_event(state, "Observation", "视频评估完成。", result)
    status = "video_passed" if result["passed"] else "video_failed"
    state["evaluation_score"] = result["score"]
    return update_current_shot(
        state,
        {
            "status": status,
            "video_score": result["score"],
            "feedback": result["feedback"],
            "video_passed": result["passed"],
        },
    )


def should_retry_video(state: AgentState) -> str:
    shot = current_shot(state)
    if shot.get("video_passed"):
        add_event(state, "Decision", f"{shot['id']} 视频通过评估。")
        return "next"
    if shot.get("video_retry_count", 0) < MAX_RETRIES:
        add_event(state, "Decision", f"{shot['id']} 视频未通过，重写 motion prompt 后重试。")
        return "retry"
    add_event(state, "Decision", f"{shot['id']} 视频达到最大重试次数，接受当前结果。")
    return "next"


def advance_shot(state: AgentState) -> AgentState:
    next_index = state.get("current_shot_index", 0) + 1
    state["current_shot_index"] = next_index
    if next_index < len(state.get("shots", [])):
        add_event(state, "Decision", f"进入下一个镜头：shot_{next_index + 1:03d}。")
    else:
        add_event(state, "Decision", "所有镜头处理完成，进入合成阶段。")
    return persist(state)


def has_more_shots(state: AgentState) -> str:
    return "more" if state.get("current_shot_index", 0) < len(state.get("shots", [])) else "compose"
