"""Composition: assemble subtitles across shots, burn them, make the cover."""

from __future__ import annotations

from agent.state import AgentState, add_event, persist
from mcp_servers import video_tools_server as tools
from mcp_servers.videobench_eval import evaluate_episode


def compose_video(state: AgentState) -> AgentState:
    shots = state.get("shots", [])
    clips = [
        {"video_path": shot["video_path"], "audio_path": shot.get("audio_path", ""), "duration_s": shot["duration_s"]}
        for shot in shots
        if shot.get("video_path")
    ]
    if not clips:
        raise ValueError("no generated clips to compose")

    subtitle_lines: list[dict] = []
    cursor = 0.0
    for shot in shots:
        for line in shot.get("subtitle_lines") or []:
            subtitle_lines.append(
                {
                    "text": line["text"],
                    "start_s": round(cursor + float(line["start_s"]), 2),
                    "end_s": round(cursor + float(line["end_s"]), 2),
                }
            )
        cursor += float(shot["duration_s"])

    add_event(
        state,
        "Action",
        "MCP.compose_video",
        {"clips": len(clips), "subtitle_lines": len(subtitle_lines), "duration_s": round(cursor, 2)},
    )
    result = tools.compose_video(clips, state["project_id"], title=state.get("logline", ""), subtitle_lines=subtitle_lines)
    state["final_video_path"] = result["final_video_path"]
    state["subtitle_path"] = result.get("subtitle_path", "")

    report = evaluate_episode(
        [
            {"shot_id": shot["id"], "score": shot.get("multidim_score", 0.0), "passed": shot.get("multidim_score", 0.0) >= 0.62,
             "dimensions": shot.get("dims_detail", {})}
            for shot in shots
        ]
    )
    state["episode_report"] = report
    add_event(state, "Observation", f"全片多维评分：{report['score']:.2f}。", report["dimensions"])

    cover = tools.make_cover(state["project_id"], title=state.get("logline", ""), subtitle=state.get("genre", ""))
    state["cover_image_path"] = cover["cover_path"]
    add_event(state, "Observation", "成片与封面生成完成。", result)
    return persist(state)


def finish(state: AgentState) -> AgentState:
    state["status"] = "completed"
    add_event(
        state,
        "Decision",
        "Pipeline 完成：设定集、分镜、关键帧、片段、字幕与成片均已落地。",
        {"final_video_path": state.get("final_video_path", ""), "usage": state.get("usage", {})},
    )
    return persist(state)
