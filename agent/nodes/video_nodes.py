"""Video nodes: motion prompt, clip generation, QA, dialogue VO, shot advance."""

from __future__ import annotations

import json

from agent.llm_client import get_llm
from agent.prompts import MOTION_PROMPT_SYSTEM
from agent.state import (
    AgentState,
    add_event,
    current_shot,
    persist,
    track_media_usage,
    track_usage,
    update_current_shot,
)
from agent.settings import get_settings
from mcp_servers import keyframe_memory
from mcp_servers import video_tools_server as tools
from mcp_servers.evaluators import evaluate_video as qa_video
from mcp_servers.providers import get_vision_judge
from mcp_servers.videobench_eval import evaluate_clip


def generate_video_prompt(state: AgentState) -> AgentState:
    shot = current_shot(state)
    add_event(state, "Thought", f"为 {shot['id']} 设计运动提示词。")
    llm = get_llm()
    request = (
        f"分镜：{json.dumps(shot, ensure_ascii=False)}\n"
        f"关键帧提示词：{shot.get('image_prompt', '')}\n"
        f"时长：{shot['duration_s']} 秒\n"
        f"整体调性：{state.get('tone', '')}"
    )
    data, reply = llm.generate_json(MOTION_PROMPT_SYSTEM, request, required_keys=("motion_prompt",))
    track_usage(state, f"{shot['id']}/motion_prompt", reply)
    add_event(state, "Decision", data.get("rationale", "使用生成的运镜描述。"), {"prompt": data["motion_prompt"]})
    return update_current_shot(
        state,
        {"status": "video_prompt_ready", "motion_prompt": data["motion_prompt"]},
    )


def _previous_tail_frame(state: AgentState, shot: dict) -> str:
    """Last frame of the preceding shot, or "" when there is nothing to chain.

    Chaining clips this way removes the jump at the cut: the next clip starts
    from (approximately) where the previous one ended instead of from a
    freshly drawn keyframe.
    """
    if not get_settings().media.first_last_frame:
        return ""
    index = shot.get("index", 0)
    if index <= 0:
        return ""
    previous = state["shots"][index - 1]
    if not previous.get("video_path"):
        return ""
    root = tools.ensure_project_dirs(state["project_id"])
    target = root / "keyframes" / f"{previous['id']}_last.png"
    keyframe_memory.extract_last_frame(root / previous["video_path"], target)
    return str(target.relative_to(root))


def call_generate_video(state: AgentState) -> AgentState:
    shot = current_shot(state)
    media = get_settings().media
    last_frame = _previous_tail_frame(state, shot)
    add_event(
        state,
        "Action",
        "MCP.generate_video",
        {"shot_id": shot["id"], "provider": media.video_provider, "duration_s": shot["duration_s"],
         "last_frame": last_frame or "none"},
    )
    result = tools.generate_video(
        shot["image_path"],
        shot["motion_prompt"],
        state["project_id"],
        shot["id"],
        duration_s=shot["duration_s"],
        caption=shot.get("description", ""),
        seed=shot["index"] + 1,
        last_frame_path=last_frame,
    )
    if media.video_provider != "mock":
        track_media_usage(state, "generate_video", media.video_provider, result.get("usage"))
    add_event(
        state,
        "Observation",
        "视频工具返回结果。",
        {k: result[k] for k in ("video_path", "provider") if k in result},
    )
    state["generated_video_path"] = result["video_path"]
    return update_current_shot(state, {"status": "video_generated", "video_path": result["video_path"]})


def call_evaluate_video(state: AgentState) -> AgentState:
    shot = current_shot(state)
    root = tools.ensure_project_dirs(state["project_id"])
    media = get_settings().media
    add_event(state, "Action", "QA.evaluate_video", {"video_path": shot["video_path"]})
    result = qa_video(
        root / shot["video_path"],
        duration_s=shot["duration_s"],
        width=media.width,
        height=media.height,
        provider=media.video_provider,
    )
    add_event(state, "Observation", "视频质检完成。", {"score": result["score"], "issues": result["issues"]})
    state["evaluation_score"] = result["score"]
    return update_current_shot(
        state,
        {
            "status": "video_passed" if result["passed"] else "video_failed",
            "video_score": result["score"],
            "feedback": result["feedback"],
            "video_passed": result["passed"],
            "video_mock": result.get("mock", False),
        },
    )


def should_retry_video(state: AgentState) -> str:
    shot = current_shot(state)
    if shot.get("video_mock"):
        add_event(state, "Decision", f"{shot['id']} 使用 mock 片段，返工无法改善，直接进入评估。")
        return "next"
    if shot.get("video_passed"):
        add_event(state, "Decision", f"{shot['id']} 片段通过质检。")
        return "next"
    if shot.get("video_retry_count", 0) < get_settings().media.video_retries:
        add_event(state, "Decision", f"{shot['id']} 片段未通过，重写运动提示词后重试。")
        return "retry"
    add_event(state, "Decision", f"{shot['id']} 达到最大重试次数，接受当前片段。")
    return "next"


def evaluate_clip_multidim(state: AgentState) -> AgentState:
    """Extract the shot keyframe, update memory, and score every dimension."""
    shot = current_shot(state)
    root = tools.ensure_project_dirs(state["project_id"])
    add_event(state, "Action", "KeyframeMemory.extract", {"shot_id": shot["id"]})

    keyframe = keyframe_memory.extract_keyframe(root / shot["video_path"], root / "keyframes" / f"{shot['id']}.png")
    memory_anchor = keyframe_memory.anchor_frame(root)
    reference = memory_anchor if memory_anchor is not None else keyframe_memory.load_reference(keyframe)
    anchor_entry = keyframe_memory.anchor_entry(root) or {}
    anchor_size = anchor_entry.get("shot_size") or _first_shot_size(state)
    report = evaluate_clip(
        root / shot["video_path"],
        shot=shot,
        reference_frame=reference,
        reference_shot_size=anchor_size,
        keyframe_path=keyframe,
        mllm_judge=get_vision_judge(),
    )

    entry = keyframe_memory.record(
        root, shot["id"], keyframe, report["score"], shot_size=shot.get("shot_size", "")
    )
    state["keyframe_memory"] = keyframe_memory.compact(keyframe_memory.read_memory(root))
    add_event(
        state,
        "Observation",
        f"{shot['id']} 多维评估完成，综合 {report['score']:.2f}。",
        {
            "scores": {
                name: item["score"] for name, item in report["dimensions"].items() if item.get("score") is not None
            },
            "skipped": [name for name, item in report["dimensions"].items() if item.get("score") is None],
            "drift_vs_anchor": entry.get("drift_vs_anchor"),
        },
    )
    return update_current_shot(
        state,
        {
            "keyframe_path": str(keyframe.relative_to(root)),
            "dims": {
                name: item["score"] for name, item in report["dimensions"].items() if item.get("score") is not None
            },
            "dims_notes": {
                name: item["note"] for name, item in report["dimensions"].items() if item.get("note")
            },
            "dims_detail": report["dimensions"],
            "multidim_score": report["score"],
            "drift_vs_anchor": entry.get("drift_vs_anchor"),
        },
    )


def _first_shot_size(state: AgentState) -> str:
    """Framing of the anchor shot; memory written before shot_size existed
    falls back to the first shot's recorded size."""
    shots = state.get("shots", [])
    return shots[0].get("shot_size", "") if shots else ""


def synthesize_dialogue(state: AgentState) -> AgentState:
    """Record the shot's line with the configured TTS provider."""
    shot = current_shot(state)
    if not shot.get("dialogue"):
        return update_current_shot(state, {"audio_path": "", "subtitle_lines": []})
    add_event(state, "Action", "MCP.synthesize_speech", {"shot_id": shot["id"], "speaker": shot.get("speaker", "")})
    result = tools.synthesize_speech(
        shot["dialogue"],
        state["project_id"],
        shot["id"],
        speaker=shot.get("speaker", ""),
        duration_s=shot["duration_s"],
    )
    if result.get("provider") != "mock":
        track_media_usage(state, "synthesize_speech", result["provider"], {})
    add_event(
        state,
        "Observation",
        f"台词配音完成，原始 {result.get('duration_s', 0):.2f}s。",
        {"audio_path": result["audio_path"], "provider": result["provider"]},
    )

    # Speech length and shot length are decided independently: the line comes
    # from the LLM, the duration from the shot list. Fit one to the other so a
    # long line never truncates the picture and a short one never leaves the
    # mux waiting on silence.
    fitted = tools.fit_audio(
        result["audio_path"],
        state["project_id"],
        shot["id"],
        target_duration=shot["duration_s"],
    )
    add_event(
        state,
        "Observation",
        f"配音适配到 {fitted['target_duration_s']}s（{fitted['action']}）。",
        fitted,
    )
    return update_current_shot(
        state,
        {
            "audio_path": fitted["audio_path"],
            "audio_raw_path": result["audio_path"],
            "audio_fit": fitted,
            "subtitle_lines": split_subtitle(shot["dialogue"], fitted["target_duration_s"]),
        },
    )


def split_subtitle(text: str, duration_s: float, max_chars: int = 14) -> list[dict]:
    """Deterministic subtitle timing: proportional to character count per line."""
    lines: list[str] = []
    current = ""
    for char in text:
        if char in "，。！？,.!? " and current:
            lines.append(current + char.strip())
            current = ""
        else:
            current += char
    if current:
        lines.append(current)
    if not lines:
        return []

    weighted = [(line, max(len(line.strip()), 1)) for line in lines]
    total = sum(weight for _, weight in weighted)
    result: list[dict] = []
    cursor = 0.0
    for line, weight in weighted:
        span = duration_s * weight / total
        result.append({"text": line.strip(), "start_s": round(cursor, 2), "end_s": round(cursor + span, 2)})
        cursor += span
    return result


def advance_shot(state: AgentState) -> AgentState:
    next_index = state.get("current_shot_index", 0) + 1
    state["current_shot_index"] = next_index
    if next_index < len(state.get("shots", [])):
        add_event(state, "Decision", f"进入下一个镜头：shot_{next_index + 1:03d}。")
    else:
        add_event(state, "Decision", "所有镜头处理完成，进入字幕与合成阶段。")
    return persist(state)


def has_more_shots(state: AgentState) -> str:
    return "more" if state.get("current_shot_index", 0) < len(state.get("shots", [])) else "compose"
