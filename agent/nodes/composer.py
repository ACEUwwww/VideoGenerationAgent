"""Composition: assemble subtitles across shots, burn them, make the cover."""

from __future__ import annotations

from agent.state import AgentState, add_event, persist
from mcp_servers import video_tools_server as tools
from mcp_servers.providers import get_vision_judge
from mcp_servers.videobench_eval import evaluate_episode, evaluate_sequence


def compose_video(state: AgentState) -> AgentState:
    shots = state.get("shots", [])
    # Subtitle times stay relative to each shot; compose_video re-bases them
    # onto the encoded timeline, which shifts once transitions overlap and a
    # title card is prepended.
    clips = [
        {
            "video_path": shot["video_path"],
            "audio_path": shot.get("audio_path", ""),
            "duration_s": shot["duration_s"],
            "transition": shot.get("transition", ""),
            "subtitle_lines": shot.get("subtitle_lines") or [],
        }
        for shot in shots
        if shot.get("video_path")
    ]
    if not clips:
        raise ValueError("no generated clips to compose")

    subtitle_count = sum(len(shot.get("subtitle_lines") or []) for shot in shots)
    planned_duration = sum(float(shot["duration_s"]) for shot in shots)

    add_event(
        state,
        "Action",
        "MCP.compose_video",
        {"clips": len(clips), "subtitle_lines": subtitle_count, "duration_s": round(planned_duration, 2),
         "transitions": [clip["transition"] for clip in clips if clip["transition"]]},
    )
    result = tools.compose_video(clips, state["project_id"], title=state.get("logline", ""))
    state["final_video_path"] = result["final_video_path"]
    state["subtitle_path"] = result.get("subtitle_path", "")

    report = evaluate_episode(
        [
            {"shot_id": shot["id"], "score": shot.get("multidim_score", 0.0), "passed": shot.get("multidim_score", 0.0) >= 0.62,
             "dimensions": shot.get("dims_detail", {})}
            for shot in shots
        ]
    )
    # Sequence-level judgement: per-clip scoring cannot see whether the cuts
    # contradict each other, whether the story lands, or whether the pacing
    # collapses. That needs the ordered keyframes together in one vision call.
    root = tools.ensure_project_dirs(state["project_id"])
    keyframes = [
        root / shot["keyframe_path"]
        for shot in shots
        if shot.get("keyframe_path") and (root / shot["keyframe_path"]).exists()
    ]
    sequence = evaluate_sequence(keyframes, shots, mllm_judge=get_vision_judge())
    report["sequence"] = sequence
    if sequence.get("score") is not None:
        add_event(
            state,
            "Observation",
            f"序列级质检：{sequence['score']:.2f}。",
            {"dimensions": {k: v["score_1to5"] for k, v in sequence["dimensions"].items()},
             "reason": sequence.get("reason", "")},
        )
    else:
        add_event(state, "Observation", f"序列级质检未执行：{sequence.get('reason', '')}")

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
