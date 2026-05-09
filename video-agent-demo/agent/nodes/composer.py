from __future__ import annotations

from agent.state import AgentState, add_event, persist
from mcp_servers import video_tools_server as tools


def compose_video(state: AgentState) -> AgentState:
    video_paths = [shot["video_path"] for shot in state.get("shots", []) if shot.get("video_path")]
    add_event(state, "Action", "MCP.compose_video", {"video_paths": video_paths})
    result = tools.compose_video(video_paths, state["project_id"])
    state["final_video_path"] = result["final_video_path"]
    add_event(state, "Observation", "最终视频合成完成。", result)
    return persist(state)


def finish(state: AgentState) -> AgentState:
    state["status"] = "completed"
    add_event(state, "Decision", "Agent pipeline 完成，输出 storyboard、镜头资源和最终视频。")
    return persist(state)
