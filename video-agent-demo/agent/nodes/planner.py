from __future__ import annotations

from agent.state import AgentState, add_event, persist
from backend.services import project_store


def analyze_user_goal(state: AgentState) -> AgentState:
    state["status"] = "running"
    add_event(state, "Thought", "分析用户故事需求，确定主题、风格和镜头数量。")
    add_event(
        state,
        "Decision",
        f"将生成 {state['shot_count']} 个镜头，整体风格为 {state['style']}。",
    )
    return persist(state)


def plan_story(state: AgentState) -> AgentState:
    prompt = state["user_prompt"]
    outline = f"围绕「{prompt}」展开一个具有起承转合的短视频故事，视觉风格为 {state['style']}。"
    state["story_outline"] = outline
    add_event(state, "Thought", "把用户需求整理为可执行的故事大纲。")
    add_event(state, "Observation", outline)
    return persist(state)


def plan_shots(state: AgentState) -> AgentState:
    shots = []
    for index in range(state["shot_count"]):
        shot_id = f"shot_{index + 1:03d}"
        shot = {
            "id": shot_id,
            "index": index,
            "title": f"镜头 {index + 1}",
            "description": f"{state['story_outline']} 第 {index + 1} 个关键画面，突出故事推进和情绪变化。",
            "status": "planned",
            "image_prompt": "",
            "motion_prompt": "",
            "image_path": "",
            "video_path": "",
            "image_score": 0,
            "video_score": 0,
            "feedback": "",
            "image_retry_count": 0,
            "video_retry_count": 0,
        }
        shots.append(shot)
        project_store.save_shot(state["project_id"], shot)

    state["shots"] = shots
    state["current_shot_index"] = 0
    add_event(state, "Action", "plan_shots", {"shot_count": len(shots)})
    add_event(state, "Observation", "Storyboard 已生成。", {"shots": shots})
    return persist(state)
