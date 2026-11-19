"""Video nodes: motion prompt, clip generation, QA, dialogue VO, shot advance."""

from __future__ import annotations

import json

from agent.llm_client import get_llm
from agent.prompts import MOTION_PROMPT_SYSTEM, T2V_PROMPT_SYSTEM
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
from mcp_servers.providers import get_video_provider, get_vision_judge
from mcp_servers.providers.ark_media import ContentModerationError
from mcp_servers.videobench_eval import evaluate_clip


def generate_video_prompt(state: AgentState) -> AgentState:
    shot = current_shot(state)
    text_to_video = not get_video_provider().frame_roles
    add_event(
        state,
        "Thought",
        f"为 {shot['id']} 设计{'文生视频' if text_to_video else '运动'}提示词。",
        {"mode": "text-to-video" if text_to_video else "image-to-video"},
    )
    llm = get_llm()
    following = (state.get("shots") or [])[shot.get("index", 0) + 1 : shot.get("index", 0) + 2]
    ending = ""
    if following and not text_to_video:
        nxt = following[0]
        ending = (
            f"\n结束帧（本镜必须在这里收尾）：{nxt.get('shot_size', '')} · {nxt.get('scene', '')} · "
            f"{nxt.get('description', '')}"
        )
    # A text-to-video model never sees a keyframe, so the prompt has to carry the
    # picture itself: who, where, what light, what framing. Handing it the
    # image-to-video template produces a description of a *change* with no scene
    # attached, and the model invents a stranger standing in an unknown place.
    # Measured: that is exactly what the first Tier 3a pass scored badly on.
    character_text = json.dumps(state.get("characters") or [], ensure_ascii=False)
    if text_to_video:
        media = get_settings().media
        orientation = "横版 landscape" if media.width > media.height else "竖版 portrait"
        request = (
            f"分镜：{json.dumps(shot, ensure_ascii=False)}\n"
            f"角色外形锚点（必须逐字写进提示词）：{character_text}\n"
            f"本次渲染画布：{media.width}x{media.height}，{orientation}\n"
            f"场景设定：{state.get('setting', '')}\n"
            f"本集梗概：{state.get('logline', '')}\n"
            f"时长：{shot['duration_s']} 秒\n"
            f"整体调性：{state.get('tone', '')}"
        )
    else:
        request = (
            f"分镜：{json.dumps(shot, ensure_ascii=False)}\n"
            f"关键帧提示词：{shot.get('image_prompt', '')}\n"
            f"时长：{shot['duration_s']} 秒"
            f"{ending}\n"
            f"整体调性：{state.get('tone', '')}"
        )
    system = T2V_PROMPT_SYSTEM if text_to_video else MOTION_PROMPT_SYSTEM
    data, reply = llm.generate_json(system, request, required_keys=("motion_prompt",))
    track_usage(state, f"{shot['id']}/motion_prompt", reply)
    add_event(state, "Decision", data.get("rationale", "使用生成的运镜描述。"), {"prompt": data["motion_prompt"]})
    return update_current_shot(
        state,
        {"status": "video_prompt_ready", "motion_prompt": data["motion_prompt"]},
    )


def _previous_tail_frame(state: AgentState, shot: dict) -> str:
    """Real last frame of the clip that plays immediately before this one.

    Used as this clip's first frame, so the cut lands on an identical picture
    and the two clips splice without a jump. This is the only joining mechanism
    available on models that reject the ``last_frame`` role (Seedance 1.0
    pro-fast does), and it needs nothing but a first frame.

    The trade-off is real: each clip is drawn from the previous clip's output,
    so it inherits that clip's small errors. It is still the right trade against
    a hard cut, and drift is now bounded from the other side too: the keyframe
    each clip is meant to start from is re-drawn from one fixed anchor, so the
    authored reference never walks away over a long sequence.
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


def _next_keyframe(state: AgentState, shot: dict) -> str:
    """Authored keyframe of the following shot, served as this clip's last frame.

    Shot N runs keyframe[N] -> keyframe[N+1]. Both ends are drawn by the image
    model from the bible's appearance anchors, so drift cannot accumulate: every
    clip returns to an authored picture. Requires a model that accepts the
    ``last_frame`` role (Seedance 2.x / 1.0 pro).
    """
    if not get_settings().media.first_last_frame:
        return ""
    shots = state.get("shots", [])
    index = shot.get("index", 0)
    if index + 1 >= len(shots):
        return ""
    path = shots[index + 1].get("image_path") or ""
    if not path:
        return ""
    return path if (tools.ensure_project_dirs(state["project_id"]) / path).exists() else ""


def call_generate_video(state: AgentState) -> AgentState:
    shot = current_shot(state)
    media = get_settings().media
    provider = get_video_provider()
    roles = provider.frame_roles
    authored_last = ""
    first_frame: str | None = None
    join = "none"

    if not roles:
        # Text-to-video: the model consumes no image at all. Recording
        # "authored" here would claim a conditioning step that never happened.
        join = "none"
    elif "last_frame" in roles:
        # Both ends authored by the image stage, so drift cannot accumulate.
        first_frame = shot["image_path"]
        join = "authored"
        authored_last = _next_keyframe(state, shot)
    else:
        # First frame only: join on the previous clip's real tail, falling back
        # to this shot's own keyframe for the opening shot.
        join = "tail-to-head"
        first_frame = _previous_tail_frame(state, shot) or shot["image_path"]

    add_event(
        state,
        "Action",
        "MCP.generate_video",
        {"shot_id": shot["id"], "provider": media.video_provider, "duration_s": shot["duration_s"],
         "join": join, "first_frame": first_frame or "none", "last_frame": authored_last or "none"},
    )
    try:
        result = tools.generate_video(
            first_frame,
            shot["motion_prompt"],
            state["project_id"],
            shot["id"],
            duration_s=shot["duration_s"],
            caption=shot.get("description", ""),
            seed=shot["index"] + 1,
            last_frame_path=authored_last,
        )
    except ContentModerationError as exc:
        # Output moderation rejects the pixels, so the same request would be
        # re-billed and rejected again. Skipping the shot is the only recovery:
        # one rejected clip used to discard the whole episode, including every
        # shot that had already rendered and been paid for.
        add_event(
            state,
            "Observation",
            f"{shot['id']} 被输出审核拒绝，跳过该镜，其余镜头继续。",
            {"shot_id": shot["id"], "error": str(exc)},
        )
        return update_current_shot(
            state,
            {"status": "video_skipped", "video_path": "", "video_skip_reason": str(exc)},
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


def clip_was_generated(state: AgentState) -> str:
    """Route a moderated shot past QA and straight to the next shot.

    There is no clip to score, and ``evaluate_clip`` extracts a keyframe from
    it. ``compose_video`` already drops any shot without a ``video_path``, so
    the episode is delivered one shot short instead of not at all.
    """
    return "skipped" if not current_shot(state).get("video_path") else "generated"


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


def start_video_phase(state: AgentState) -> AgentState:
    """Rewind the shot cursor once every keyframe exists.

    The image phase runs to completion first so that shot N can be handed
    keyframe[N+1] as its final frame. Rendering the two phases interleaved would
    mean keyframe[N+1] did not exist yet when shot N is rendered.
    """
    state["current_shot_index"] = 0
    add_event(
        state,
        "Decision",
        "全部关键帧就绪，进入视频阶段；第 N 镜以 keyframe[N] 起、以 keyframe[N+1] 收。",
    )
    return persist(state)


def advance_keyframe(state: AgentState) -> AgentState:
    next_index = state.get("current_shot_index", 0) + 1
    state["current_shot_index"] = next_index
    if next_index < len(state.get("shots", [])):
        add_event(state, "Decision", f"关键帧完成，进入 shot_{next_index + 1:03d}。")
    else:
        add_event(state, "Decision", "全部关键帧生成完毕。")
    return persist(state)


def has_more_keyframes(state: AgentState) -> str:
    return "more" if state.get("current_shot_index", 0) < len(state.get("shots", [])) else "all_done"


def needs_image_phase(state: AgentState) -> str:
    """Route past the whole image stage when the video model takes no image.

    A text-to-video model has nothing to condition on, so rendering keyframes
    would spend a stage's worth of calls producing pictures no shot consumes —
    and every metric derived from them (anchor drift, keyframe quality) would
    then be scored against a placeholder. Skipping the stage entirely keeps the
    run honest: the shot list goes straight to the video prompt.

    ``start_video_phase`` is still the entry point in both cases, because it is
    what rewinds the shot cursor after planning.
    """
    if not get_video_provider().frame_roles:
        add_event(
            state,
            "Decision",
            "视频模型不接受任何图像输入（T2V），跳过整个关键帧阶段；"
            "本层不产出锚点，所有依赖关键帧的指标标记为不适用。",
            {"reason": "text-to-video", "provider": get_settings().media.video_provider},
        )
        return "text_to_video"
    return "image"


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
