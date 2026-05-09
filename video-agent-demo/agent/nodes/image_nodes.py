from __future__ import annotations

from agent.state import AgentState, add_event, current_shot, update_current_shot
from mcp_servers import video_tools_server as tools

MAX_RETRIES = 2


def generate_image_prompt(state: AgentState) -> AgentState:
    shot = current_shot(state)
    prompt = (
        f"{shot['description']} 风格：{state['style']}。"
        "画面需要清晰主体、环境细节、电影感光线。"
    )
    add_event(state, "Thought", f"为 {shot['id']} 生成图片 prompt。")
    add_event(state, "Decision", "先生成静态关键帧，再做图像评估。")
    return update_current_shot(state, {"status": "image_prompt_ready", "image_prompt": prompt})


def call_generate_image(state: AgentState) -> AgentState:
    shot = current_shot(state)
    add_event(state, "Action", "MCP.generate_image", {"prompt": shot["image_prompt"], "shot_id": shot["id"]})
    result = tools.generate_image(shot["image_prompt"], state["project_id"], shot["id"])
    add_event(state, "Observation", "图片工具返回结果。", result)
    state["generated_image_path"] = result["image_path"]
    return update_current_shot(state, {"status": "image_generated", "image_path": result["image_path"]})


def call_evaluate_image(state: AgentState) -> AgentState:
    shot = current_shot(state)
    add_event(state, "Action", "MCP.evaluate_image", {"image_path": shot["image_path"]})
    result = tools.evaluate_image(shot["image_path"], shot["description"], state["style"])
    add_event(state, "Observation", "图片评估完成。", result)
    status = "image_passed" if result["passed"] else "image_failed"
    state["evaluation_score"] = result["score"]
    return update_current_shot(
        state,
        {
            "status": status,
            "image_score": result["score"],
            "feedback": result["feedback"],
            "image_passed": result["passed"],
        },
    )


def should_retry_image(state: AgentState) -> str:
    shot = current_shot(state)
    if shot.get("image_passed"):
        add_event(state, "Decision", f"{shot['id']} 图片通过评估，进入视频生成。")
        return "video"
    if shot.get("image_retry_count", 0) < MAX_RETRIES:
        add_event(state, "Decision", f"{shot['id']} 图片未通过，重写 prompt 后重试。")
        return "retry"
    add_event(state, "Decision", f"{shot['id']} 图片达到最大重试次数，继续进入视频生成。")
    return "video"
